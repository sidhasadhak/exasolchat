"""Core ExasolChat engine.

Connects schema introspection, LLM generation, knowledge base retrieval,
safety validation, query execution, and chart suggestion into a single
.ask() interface.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from exachat.builder import QueryBuilder
from exachat.charts import auto_chart
from exachat.connection import ConnectionConfig, DatabaseConnection
from exachat.kb import KnowledgeBase
from exachat.llm import LLMBackend, LLMResponse, OllamaBackend
from exachat.metrics import MetricsCatalog
from exachat.safety import RiskLevel, SafetyVerdict, sanitize_sql, validate_sql
from exachat.schema import (
    SchemaContext,
    introspect_duckdb,
    introspect_exasol,
    introspect_sqlalchemy,
)


@dataclass
class QueryResult:
    """Result of a natural language query."""
    question: str
    sql: str
    safety: SafetyVerdict
    data: Optional[pd.DataFrame] = None
    summary: Optional[str] = None
    chart_config: Optional[dict] = None
    chart_obj: Optional[object] = None  # ("plotly", fig) or ("altair", chart)
    error: Optional[str] = None
    explanation: Optional[str] = None
    kb_patterns_used: int = 0
    column_warnings: list[str] = None
    followups: list[str] = None

    def __post_init__(self):
        if self.column_warnings is None:
            self.column_warnings = []
        if self.followups is None:
            self.followups = []


class ExasolChat:
    """Main interface. Connect a database + LLM, ask questions, get answers.

    Usage:
        chat = ExasolChat("duckdb:///data.duckdb")
        result = chat.ask("What are the top 10 customers by revenue?")
        print(result.data)
    """

    def __init__(
        self,
        connection: str | ConnectionConfig,
        llm: Optional[LLMBackend] = None,
        schema: Optional[str] = None,
        include_tables: Optional[list[str]] = None,
        exclude_tables: Optional[list[str]] = None,
        allowed_schemas: Optional[list[str]] = None,
        allowed_tables: Optional[list[str]] = None,
        extra_context: str = "",
        max_rows: int = 5000,
        kb_path: Optional[str] = None,
        chart_library: str = "auto",
        metrics_path: Optional[str] = None,
    ):
        # Connection
        if isinstance(connection, str):
            self._config = ConnectionConfig.from_url(connection)
        else:
            self._config = connection

        self._db = DatabaseConnection(self._config)
        self._db.connect()

        # LLM
        self.llm = llm or OllamaBackend()

        # Settings
        self.max_rows = max_rows
        self.chart_library = chart_library
        self._allowed_schemas = set(allowed_schemas) if allowed_schemas else None
        self._allowed_tables = set(allowed_tables) if allowed_tables else None

        # Knowledge base (built-ins always loaded; extra dir optional)
        self.kb = KnowledgeBase()
        if kb_path:
            self.kb.load_dir(kb_path)

        # Metrics catalog (persisted JSON; always initialised)
        self.metrics_catalog = MetricsCatalog(metrics_path)

        # Merge explicit include_tables with allowed_tables: if allowed_tables
        # is set and include_tables is not, restrict introspection to the
        # allowed set so the schema prompt, field palette, and sidebar only
        # surface tables the user is actually permitted to query.
        effective_include = include_tables
        if not effective_include and self._allowed_tables:
            effective_include = list(self._allowed_tables)

        # Schema introspection
        if self._db.is_exasol:
            self.schema_context = introspect_exasol(
                self._db.pyexasol_conn,
                schema=schema or self._config.exasol_schema,
                include_tables=effective_include,
                exclude_tables=exclude_tables,
            )
        elif self._db.is_duckdb:
            self.schema_context = introspect_duckdb(
                self._db.duckdb_conn,
                schema=schema,
                include_tables=effective_include,
                exclude_tables=exclude_tables,
            )
        else:
            self.schema_context = introspect_sqlalchemy(
                self._db.sqla_engine,
                schema=schema,
                include_tables=include_tables,
                exclude_tables=exclude_tables,
            )

        if extra_context:
            self.schema_context.extra_context = extra_context

        # Visual query builder — exposes schema introspection to the UI
        self.builder = QueryBuilder(self.schema_context)

        self._history: list[QueryResult] = []

    @property
    def schema_prompt(self) -> str:
        return self.schema_context.to_prompt()

    @property
    def history(self) -> list[QueryResult]:
        return list(self._history)

    def add_context(self, context: str) -> None:
        """Add extra context (business rules, DDL, column descriptions)."""
        if self.schema_context.extra_context:
            self.schema_context.extra_context += "\n" + context
        else:
            self.schema_context.extra_context = context

    def ask(self, question: str) -> QueryResult:
        """Ask a natural language question. Returns SQL + data + chart."""

        # 1. KB retrieval — find relevant SQL patterns
        kb_patterns = []
        kb_context = None
        try:
            kb_patterns = self.kb.search(question)
            if kb_patterns:
                dialect = self.schema_context.dialect or ""
                kb_context = self.kb.format_for_prompt(kb_patterns, dialect=dialect)
        except Exception:
            pass  # KB failure should never block a query

        # 2. LLM generates SQL — include recent session history for follow-up resolution
        recent_history = [
            {"question": r.question, "sql": r.sql}
            for r in self._history[-3:]
            if r.sql and not r.error
        ]

        # Build schema prompt; append metrics catalog if any are defined
        schema_prompt = self.schema_prompt
        if self.metrics_catalog and self.metrics_catalog.count > 0:
            schema_prompt += "\n\n" + self.metrics_catalog.format_for_prompt()

        # Reinforce access control so the LLM doesn't hallucinate unauthorised tables
        if self._allowed_tables:
            schema_prompt += (
                "\n\nACCESS CONTROL — CRITICAL: You may ONLY reference these tables: "
                + ", ".join(sorted(self._allowed_tables))
                + ". Never use any other table in your SQL, even if the question implies one."
            )

        try:
            llm_resp: LLMResponse = self.llm.generate_sql(
                schema_prompt, question, kb_context,
                history=recent_history or None,
            )
        except Exception as e:
            return self._error_result(question, "", f"LLM generation failed: {e}")

        sql = sanitize_sql(llm_resp.sql)
        column_warnings = _check_column_ambiguity(sql, self.schema_context)

        # 3. Safety validation — NEVER skip
        verdict = validate_sql(
            sql,
            allowed_schemas=self._allowed_schemas,
            allowed_tables=self._allowed_tables,
        )
        if verdict.level == RiskLevel.BLOCKED:
            result = QueryResult(
                question=question, sql=sql, safety=verdict,
                error=f"Query blocked: {verdict.reason}",
                explanation=llm_resp.explanation,
                kb_patterns_used=len(kb_patterns),
                column_warnings=column_warnings,
            )
            self._history.append(result)
            return result

        # 4. Rewrite normalized column names back to quoted originals, then execute
        exec_sql = self.schema_context.denormalize_sql(sql)

        # For PostgreSQL: guard timestamp/date casts against empty-string values
        # that are common in CSV-loaded data and cause InvalidDatetimeFormat errors.
        _dialect = (self.schema_context.dialect or "").lower()
        if "postgres" in _dialect or "postgresql" in _dialect:
            exec_sql = _pg_fix_timestamp_casts(exec_sql)

        try:
            df = self._db.execute_query(exec_sql, self.max_rows)
        except Exception as e:
            result = QueryResult(
                question=question, sql=exec_sql, safety=verdict,
                error=f"Query execution failed: {e}",
                explanation=llm_resp.explanation,
                kb_patterns_used=len(kb_patterns),
                column_warnings=column_warnings,
            )
            self._history.append(result)
            return result

        # 5. Generate summary
        summary = None
        try:
            preview = df.head(20).to_string(index=False)
            summary = self.llm.generate_summary(question, sql, preview)
        except Exception:
            summary = f"Returned {len(df)} rows, {len(df.columns)} columns."

        # 6. Suggest chart
        chart_config = None
        chart_obj = None
        if len(df) > 0 and len(df.columns) >= 2:
            try:
                chart_config = self.llm.suggest_chart(
                    question, list(df.columns), len(df)
                )
                chart_obj = auto_chart(df, chart_config, self.chart_library)
            except Exception:
                chart_config = {"chart_type": "table_only"}

        # 7. Suggest follow-up questions based on this result
        followups = []
        try:
            preview = df.head(5).to_string(index=False)
            followups = self.llm.suggest_followups(question, sql, preview)
        except Exception:
            pass

        result = QueryResult(
            question=question, sql=exec_sql, safety=verdict,
            data=df, summary=summary,
            chart_config=chart_config, chart_obj=chart_obj,
            explanation=llm_resp.explanation,
            kb_patterns_used=len(kb_patterns),
            column_warnings=column_warnings,
            followups=followups,
        )
        self._history.append(result)
        return result

    def generate_explore_questions(self) -> list[str]:
        """Generate 5 starter questions, served from cache when schema is unchanged."""
        fp = self._schema_fingerprint()
        cached = self._load_question_cache(fp)
        if cached:
            return cached
        try:
            profile = self._build_profile()
            questions = self.llm.generate_explore_questions(self.schema_prompt, profile)
            if questions:
                self._save_question_cache(fp, questions)
            return questions
        except Exception:
            return []

    # ── Schema fingerprint + question cache ───────────────────────────

    _CACHE_FILE = Path.home() / ".exachat" / "explore_cache.json"
    _CACHE_MAX  = 30  # keep at most this many schema fingerprints

    def _schema_fingerprint(self) -> str:
        key = "|".join(
            f"{t.name}:{len(t.columns)}:{t.row_count or 0}"
            for t in self.schema_context.tables
        )
        return hashlib.md5(key.encode()).hexdigest()[:16]

    def _load_question_cache(self, fp: str) -> list[str] | None:
        try:
            if self._CACHE_FILE.exists():
                data = json.loads(self._CACHE_FILE.read_text())
                return data.get(fp)
        except Exception:
            pass
        return None

    def _save_question_cache(self, fp: str, questions: list[str]) -> None:
        try:
            self._CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            data: dict = {}
            if self._CACHE_FILE.exists():
                data = json.loads(self._CACHE_FILE.read_text())
            data[fp] = questions
            if len(data) > self._CACHE_MAX:
                for old in list(data)[: len(data) - self._CACHE_MAX]:
                    del data[old]
            self._CACHE_FILE.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    def _build_profile(self) -> str:
        """Lightweight data profile sent alongside the schema to generate questions.

        For DuckDB: uses SUMMARIZE (min/max/unique counts).
        For everything else: schema_prompt already contains all column info,
        so we only add row counts to avoid a redundant LLM context.
        """
        lines = []
        for table in self.schema_context.tables[:5]:
            row_count = table.row_count or "?"
            lines.append(f"\nTable: {table.name} ({row_count} rows)")

            if self._db.is_duckdb:
                try:
                    df = self._db.duckdb_conn.execute(
                        f'SELECT * FROM (SUMMARIZE "{table.name}") LIMIT 30'
                    ).df()
                    for _, row in df.iterrows():
                        col    = row.get("column_name", "")
                        ctype  = row.get("column_type", "")
                        mn     = row.get("min", "")
                        mx     = row.get("max", "")
                        unique = row.get("approx_unique", "")
                        nulls  = row.get("null_percentage", "")
                        lines.append(
                            f"  - {col} ({ctype}): min={mn}, max={mx}, "
                            f"~{unique} unique, {nulls}% null"
                        )
                    continue
                except Exception:
                    pass

            # Non-DuckDB: column names/types are already in schema_prompt;
            # just confirm which columns exist (first 8 to keep prompt small).
            for col in table.columns[:8]:
                lines.append(f"  - {col.name}: {col.type}")

        return "\n".join(lines)

    def clear_history(self) -> None:
        """Clear session conversation history (for new chat)."""
        self._history.clear()

    def close(self):
        """Close database connection."""
        self._db.close()

    def _error_result(self, question: str, sql: str, error: str) -> QueryResult:
        result = QueryResult(
            question=question, sql=sql,
            safety=SafetyVerdict(RiskLevel.BLOCKED, sql, "error"),
            error=error,
        )
        self._history.append(result)
        return result

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _normalise(name: str) -> str:
    return re.sub(r"[\s_]+", "", name.lower())


def _pg_fix_timestamp_casts(sql: str) -> str:
    """Rewrite unsafe timestamp/date casts for PostgreSQL.

    Data loaded from CSV often stores missing dates as '' (empty string) rather
    than NULL.  PostgreSQL's CAST / :: operator raises InvalidDatetimeFormat on
    empty strings.  Wrapping the expression in NULLIF(expr, '') makes the cast
    return NULL instead of erroring.

    Handles both forms the LLM commonly emits:
        CAST(col AS TIMESTAMP)  →  NULLIF(col, '')::TIMESTAMP
        col::TIMESTAMP          →  NULLIF(col, '')::TIMESTAMP
    """
    _TS = r'(TIMESTAMP(?:TZ|(?:\s+WITH(?:OUT)?\s+TIME\s+ZONE))?|DATE)'
    # Already-wrapped expressions — don't touch
    _already = re.compile(r"NULLIF\s*\(", re.IGNORECASE)

    def _wrap(expr: str, typ: str) -> str:
        expr = expr.strip()
        if _already.match(expr):
            return f"{expr}::{typ}"
        return f"NULLIF({expr}, '')::{typ.strip()}"

    # 1. CAST(expr AS TIMESTAMP/DATE)
    sql = re.sub(
        rf'CAST\(\s*(\w+)\s+AS\s+{_TS}\s*\)',
        lambda m: _wrap(m.group(1), m.group(2)),
        sql, flags=re.IGNORECASE,
    )

    # 2. plain_identifier::TIMESTAMP/DATE  (skip function names / keywords)
    _SKIP = {"NOW", "CURRENT_TIMESTAMP", "CURRENT_DATE", "INTERVAL",
             "NULLIF", "COALESCE", "CAST", "TRUE", "FALSE"}
    def _replace_bare(m: re.Match) -> str:
        ident = m.group(1)
        if ident.upper() in _SKIP:
            return m.group(0)
        return _wrap(ident, m.group(2))

    sql = re.sub(
        rf'\b(\w+)::{_TS}\b',
        _replace_bare,
        sql, flags=re.IGNORECASE,
    )
    return sql


def _check_column_ambiguity(sql: str, schema: "SchemaContext") -> list[str]:
    """Detect column names in SQL that fuzzy-match schema columns but don't match exactly."""
    all_cols = {c.name for t in schema.tables for c in t.columns}
    norm_to_exact: dict[str, str] = {_normalise(c): c for c in all_cols}

    tokens = set(re.findall(r'"([^"]+)"|([A-Za-z_]\w*)', sql))
    used_names = {q or u for q, u in tokens if (q or u)}

    warnings = []
    for name in used_names:
        if name.upper() in ("SELECT", "FROM", "WHERE", "JOIN", "ON", "AND", "OR",
                            "GROUP", "BY", "ORDER", "HAVING", "LIMIT", "AS",
                            "WITH", "INNER", "LEFT", "RIGHT", "OUTER", "COUNT",
                            "SUM", "AVG", "MIN", "MAX", "DISTINCT", "NULL", "NOT",
                            "IN", "LIKE", "CASE", "WHEN", "THEN", "ELSE", "END",
                            "OVER", "PARTITION", "ROW_NUMBER", "RANK", "ALL",
                            "CAST", "INTERVAL", "DATE", "TIMESTAMP", "TRUE", "FALSE"):
            continue
        if name in all_cols:
            continue
        norm = _normalise(name)
        if norm in norm_to_exact:
            exact = norm_to_exact[norm]
            if exact != name:
                warnings.append(
                    f"⚠️ Column **'{name}'** not found — did you mean **'{exact}'**?"
                )
    return warnings
