"""Core TalonSight engine.

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

from talonsight.builder import QueryBuilder
from talonsight.charts import auto_chart
from talonsight.connection import ConnectionConfig, DatabaseConnection
from talonsight.kb import KnowledgeBase, SchemaIndex, build_embedding_fn
from talonsight.llm import LLMBackend, LLMResponse, OllamaBackend
from talonsight.metrics import MetricsCatalog
from talonsight.safety import RiskLevel, SafetyVerdict, sanitize_sql, validate_sql
from talonsight.schema import (
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
    # Auto-correction fields — set when a query failed then recovered
    auto_corrected: bool = False
    original_sql: Optional[str] = None      # the SQL that first failed
    original_error: Optional[str] = None    # the error from the first attempt
    correction_explanation: Optional[str] = None  # what the LLM says it fixed
    # Agent fields — populated when ask_agent() is used; None in classic mode
    agent_steps: Optional[list] = None      # list[AgentStep]
    agent_plan: Optional[list] = None       # list[str] from create_plan step

    def __post_init__(self):
        if self.column_warnings is None:
            self.column_warnings = []
        if self.followups is None:
            self.followups = []


class TalonSight:
    """Main interface. Connect a database + LLM, ask questions, get answers.

    Usage:
        chat = TalonSight("duckdb:///data.duckdb")
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
        embedding_backend: str = "bow",
        embedding_url: str = "",
        embedding_model: str = "nomic-embed-text",
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

        # Build the embedding function once and share it between the KB and the
        # schema index so both collections use the same model and vector space.
        # Falls back to bag-of-words silently if the server is unreachable.
        _ef = build_embedding_fn(
            backend=embedding_backend,
            url=embedding_url,
            model=embedding_model,
        )

        # Knowledge base (built-ins always loaded; extra dir optional)
        self.kb = KnowledgeBase(ef=_ef)
        if kb_path:
            self.kb.load_dir(kb_path)

        # Schema index — built after introspection below; declared here so the
        # attribute always exists even if introspection fails.
        self._schema_index = SchemaIndex(ef=_ef)

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

        # Build schema index for large-schema retrieval
        # Uses the same fingerprint as the question cache so both stay in sync.
        try:
            fp = self._schema_fingerprint()
            self._schema_index.index(self.schema_context.tables, fp)
        except Exception:
            pass  # index failure must never block a connection

        # Visual query builder — exposes schema introspection to the UI
        self.builder = QueryBuilder(self.schema_context)

        # Business model — persists confirmed findings across sessions
        from talonsight.memory import BusinessModel
        _conn_id = (
            self._config.duckdb_path
            or self._config.sqlalchemy_url
            or getattr(self._config, "exasol_host", None)
            or "default"
        )
        self.business_model = BusinessModel(_conn_id)

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

    def ask(
        self,
        question: str,
        on_attempt: Optional[callable] = None,
        max_attempts: int = 3,
    ) -> QueryResult:
        """Ask a natural language question. Returns SQL + data + chart.

        Args:
            question:    Natural language question.
            on_attempt:  Optional callback(attempt: int, total: int) called
                         before each retry attempt (not called on attempt 1).
            max_attempts: Maximum execution attempts (default 3).
        """

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

        # Build schema prompt — use retrieved subset for large schemas, full
        # schema for small ones.  The index is a transparent pass-through when
        # the schema has <= SchemaIndex.THRESHOLD tables.
        if self._schema_index.active:
            relevant_tables = self._schema_index.retrieve(question)
            _filtered_ctx = SchemaContext(
                tables=relevant_tables,
                dialect=self.schema_context.dialect,
                extra_context=self.schema_context.extra_context,
            )
            schema_prompt = _filtered_ctx.to_prompt()
        else:
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
            exec_sql = _pg_post_process(exec_sql)

        # ── Execute with auto-correction loop (up to max_attempts) ──────────
        current_sql = exec_sql
        current_verdict = verdict
        original_sql = exec_sql          # the first SQL that was tried
        original_error: Optional[str] = None
        auto_corrected = False
        correction_explanation: Optional[str] = None
        df = None

        for attempt in range(1, max_attempts + 1):
            # Notify UI on retry attempts (not on the first attempt)
            if attempt > 1 and on_attempt:
                try:
                    on_attempt(attempt, max_attempts)
                except Exception:
                    pass

            try:
                df = self._db.execute_query(current_sql, self.max_rows)
                if attempt > 1:
                    auto_corrected = True
                break  # success — exit retry loop

            except Exception as e:
                err_str = str(e)
                if original_error is None:
                    original_error = err_str  # capture first failure only

                if attempt == max_attempts:
                    # All attempts exhausted — return final error
                    result = QueryResult(
                        question=question, sql=current_sql, safety=current_verdict,
                        error=f"Query execution failed: {err_str}",
                        explanation=llm_resp.explanation,
                        kb_patterns_used=len(kb_patterns),
                        column_warnings=column_warnings,
                        original_sql=original_sql if attempt > 1 else None,
                        original_error=original_error,
                    )
                    self._history.append(result)
                    return result

                # Ask the LLM to diagnose and fix the SQL — provide schema so
                # the LLM can verify table/column names when fixing references.
                try:
                    fix_resp = self.llm.fix_sql(
                        question, current_sql, err_str, schema=schema_prompt
                    )
                    fixed_sql = sanitize_sql(fix_resp.sql)

                    # Re-validate safety before running the fixed SQL
                    fixed_verdict = validate_sql(
                        fixed_sql,
                        allowed_schemas=self._allowed_schemas,
                        allowed_tables=self._allowed_tables,
                    )
                    if fixed_verdict.level == RiskLevel.BLOCKED:
                        result = QueryResult(
                            question=question, sql=fixed_sql, safety=fixed_verdict,
                            error=f"Fixed query blocked: {fixed_verdict.reason}",
                            explanation=fix_resp.explanation,
                            kb_patterns_used=len(kb_patterns),
                            column_warnings=column_warnings,
                        )
                        self._history.append(result)
                        return result

                    # Apply the same post-processing pipeline to fixed SQL
                    current_sql = self.schema_context.denormalize_sql(fixed_sql)
                    if "postgres" in _dialect or "postgresql" in _dialect:
                        current_sql = _pg_post_process(current_sql)
                    current_verdict = fixed_verdict
                    correction_explanation = fix_resp.explanation or f"Auto-corrected on attempt {attempt + 1}"
                except Exception:
                    pass  # fix_sql failed — retry loop will try again or exhaust

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
            question=question, sql=current_sql, safety=current_verdict,
            data=df, summary=summary,
            chart_config=chart_config, chart_obj=chart_obj,
            explanation=llm_resp.explanation,
            kb_patterns_used=len(kb_patterns),
            column_warnings=column_warnings,
            followups=followups,
            auto_corrected=auto_corrected,
            original_sql=original_sql if auto_corrected else None,
            original_error=original_error if auto_corrected else None,
            correction_explanation=correction_explanation,
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

    # ── Agent mode ────────────────────────────────────────────────────

    def ask_agent(
        self,
        question: str,
        on_step=None,
        max_steps: int = 12,
    ) -> QueryResult:
        """Run a multi-step agentic investigation for a natural-language question.

        The agent calls tools (schema lookup, SQL execution, KB search, etc.)
        iteratively until it reaches a confident answer, then synthesises a
        plain-English narrative alongside the key SQL and chart hint.

        Args:
            question:   Natural-language business question.
            on_step:    Optional callback(AgentStep) called after each tool call.
                        Useful for streaming live progress to the Streamlit UI.
            max_steps:  Hard cap on total tool calls (default 12).

        Returns:
            QueryResult with agent_steps, agent_plan, and summary = narrative.
        """
        from talonsight.agent import AgentLoop

        loop = AgentLoop(
            connector=self._db,
            schema_context=self.schema_context,
            llm=self.llm,
            business_model=self.business_model,
            kb=self.kb,
            dialect=getattr(self.schema_context, "dialect", ""),
            allowed_schemas=list(self._allowed_schemas) if self._allowed_schemas else [],
            allowed_tables=list(self._allowed_tables) if self._allowed_tables else [],
            max_steps=max_steps,
        )

        ar = loop.run_sync(question, on_step=on_step)

        # Build chart from hint + data
        chart_config = None
        chart_obj = None
        if ar.data is not None and not ar.data.empty and ar.chart_hint:
            try:
                chart_config = {
                    "chart_type": ar.chart_hint,
                    "title": question[:80],
                    "x": ar.data.columns[0] if len(ar.data.columns) > 0 else None,
                    "y": ar.data.columns[1] if len(ar.data.columns) > 1 else None,
                }
                chart_obj = auto_chart(ar.data, chart_config, self.chart_library)
            except Exception:
                pass

        result = QueryResult(
            question=question,
            sql=ar.sql or "",
            safety=SafetyVerdict(RiskLevel.SAFE, ar.sql or "", "agent"),
            data=ar.data,
            summary=ar.narrative,
            chart_config=chart_config,
            chart_obj=chart_obj,
            error=ar.error,
            agent_steps=ar.steps,
            agent_plan=ar.plan,
        )
        self._history.append(result)
        return result

    # ── Schema fingerprint + question cache ───────────────────────────

    _CACHE_FILE = Path.home() / ".talonsight" / "explore_cache.json"
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

    # ── Data Quality Scan ─────────────────────────────────────────────

    def run_dq_scan(
        self,
        table_name: str,
        config: Optional[dict] = None,
        config_path: Optional[str] = None,
        on_progress=None,
    ) -> list:
        """Run a data-quality scan against a single table.

        Parameters
        ----------
        table_name:   Name of the table to scan (must exist in the connected schema).
        config:       Optional DQ config dict (overrides built-in dq_config.json).
        config_path:  Optional path to a custom DQ config JSON file.
        on_progress:  Optional callback(current, total, rule, col) for UI updates.

        Returns a list of ``DQResult`` objects sorted by severity × failure rate.
        Raises ``ValueError`` if the table is not found in the connected schema.
        """
        from talonsight.dq import DataQualityScanner

        # Resolve TableInfo
        table_info = next(
            (t for t in self.schema_context.tables
             if t.name.lower() == table_name.lower()
             or f"{t.schema}.{t.name}".lower() == table_name.lower()),
            None,
        )
        if table_info is None:
            raise ValueError(
                f"Table '{table_name}' not found in the connected schema. "
                f"Available: {[t.name for t in self.schema_context.tables]}"
            )

        scanner = DataQualityScanner(config=config, config_path=config_path)
        return scanner.run(
            table_name=table_name,
            table_info=table_info,
            db=self._db,
            dialect=self.schema_context.dialect or "",
            llm=self.llm,
            on_progress=on_progress,
        )

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


def _pg_fix_round(sql: str) -> str:
    """Rewrite ROUND(expr, n) → ROUND((expr)::numeric, n) for PostgreSQL.

    PostgreSQL only defines ROUND(numeric, integer).  AVG(), SUM()/COUNT(),
    and float-column division all return double precision, so
    ROUND(AVG(price), 2) raises "function round(double precision, integer)
    does not exist".  Casting to ::numeric before ROUND fixes it.

    Uses balanced-parenthesis walking so deeply nested expressions such as
    ROUND(AVG(NULLIF(col, 0)), 2) are handled correctly.
    Skips expressions already ending in ::numeric or CAST(... AS NUMERIC).
    """
    out: list[str] = []
    pos = 0

    for m in re.finditer(r'\bROUND\s*\(', sql, re.IGNORECASE):
        open_pos = m.end() - 1          # index of '('

        # Find matching ')' via balanced-paren walk
        depth      = 0
        close_pos  = -1
        for i in range(open_pos, len(sql)):
            if sql[i] == '(':
                depth += 1
            elif sql[i] == ')':
                depth -= 1
                if depth == 0:
                    close_pos = i
                    break

        if close_pos < 0:
            continue                     # unbalanced — leave unchanged

        inner = sql[open_pos + 1: close_pos]

        # Find the last top-level comma (separates expr from precision arg)
        last_comma = -1
        d = 0
        for i, c in enumerate(inner):
            if c == '(':
                d += 1
            elif c == ')':
                d -= 1
            elif c == ',' and d == 0:
                last_comma = i

        if last_comma < 0:
            continue                     # ROUND(expr) single-arg — leave alone

        expr = inner[:last_comma].strip()
        prec = inner[last_comma + 1:].strip()

        # Only rewrite when precision is a bare integer literal
        if not re.match(r'^\d+$', prec):
            continue

        # Skip if already cast to numeric
        el = expr.lower()
        if el.endswith('::numeric') or el.endswith('::decimal') or 'as numeric' in el:
            continue

        out.append(sql[pos: m.start()])
        out.append(f'ROUND(({expr})::numeric, {prec})')
        pos = close_pos + 1

    out.append(sql[pos:])
    return ''.join(out)


def _pg_post_process(sql: str) -> str:
    """Run the full PostgreSQL post-processing pipeline on a generated SQL string.

    Applied to every query before first execution and to every auto-corrected
    query before retry execution.  Order matters:
      1. Timestamp/date casts   — alias.col::timestamp  →  NULLIF(alias.col,'')::timestamp
      2. Interval-to-days       — expr::numeric/86400   →  EXTRACT(EPOCH FROM expr)/86400
      3. ROUND precision cast   — ROUND(expr, n)        →  ROUND(expr::numeric, n)
    """
    sql = _pg_fix_timestamp_casts(sql)
    sql = _pg_fix_round(sql)
    return sql


def _pg_rewrite_interval_to_days(sql: str) -> str:
    """Rewrite incorrect interval arithmetic to EXTRACT(EPOCH FROM ...) / 86400.

    ``timestamp - timestamp`` in PostgreSQL yields an ``interval``.  The LLM
    commonly tries two broken patterns to turn that into a number of days:

    * ``expr::numeric / 86400``            — interval cannot be cast to numeric
    * ``expr::interval / INTERVAL '1 day'``— interval / interval unsupported

    Both are rewritten to ``EXTRACT(EPOCH FROM expr) / 86400``, which extracts
    the interval as seconds and divides by 86400 to give fractional days.

    Uses balanced-parenthesis matching to find the exact expression boundary,
    so nested NULLIF / COALESCE calls are handled correctly.
    """

    def _find_open_paren(s: str, close_idx: int) -> int:
        """Return the index of the '(' that balances the ')' at close_idx."""
        depth = 0
        for i in range(close_idx, -1, -1):
            if s[i] == ')':
                depth += 1
            elif s[i] == '(':
                depth -= 1
                if depth == 0:
                    return i
        return -1  # unbalanced — leave SQL untouched

    # Suffix patterns that follow a closing ')' and indicate misused intervals
    _INTERVAL_SUFFIX = re.compile(
        r'\)::(?:numeric|integer|float\d*|double\s+precision)\s*/\s*86400'
        r'|'
        r"\)::interval\s*/\s*INTERVAL\s*'1\s+days?'",
        re.IGNORECASE,
    )

    out: list[str] = []
    pos = 0

    for m in _INTERVAL_SUFFIX.finditer(sql):
        close_pos = m.start()               # index of the ')' before ::numeric etc.
        open_pos  = _find_open_paren(sql, close_pos)
        if open_pos < 0:
            continue                         # unbalanced — skip

        inner = sql[open_pos + 1: close_pos]

        # Only rewrite when the expression contains a timestamp subtraction
        il = inner.lower()
        if '::timestamp' not in il and '::date' not in il:
            continue

        # Walk back from open_pos to include any function keyword (e.g. NULLIF)
        func_start = open_pos
        while func_start > 0 and (sql[func_start - 1].isalnum() or sql[func_start - 1] == '_'):
            func_start -= 1

        full_expr = sql[func_start: close_pos + 1]   # e.g. NULLIF(inner_expr)

        out.append(sql[pos: func_start])
        out.append(f'EXTRACT(EPOCH FROM {full_expr}) / 86400')
        pos = m.end()

    out.append(sql[pos:])
    return ''.join(out)


def _pg_fix_timestamp_casts(sql: str) -> str:
    """Rewrite unsafe timestamp/date casts for PostgreSQL.

    Data loaded from CSV often stores missing dates as '' (empty string) rather
    than NULL.  PostgreSQL's CAST / :: operator raises InvalidDatetimeFormat on
    empty strings.  Wrapping the expression in NULLIF(expr, '') makes the cast
    return NULL instead of erroring.

    Handles the three forms the LLM commonly emits:
        CAST(col AS TIMESTAMP)        →  NULLIF(col, '')::TIMESTAMP
        col::TIMESTAMP                →  NULLIF(col, '')::TIMESTAMP
        alias.col::TIMESTAMP          →  NULLIF(alias.col, '')::TIMESTAMP

    The third form is the critical fix: the previous regex ``(word+)::TYPE`` only
    captured the bare column name and left the ``alias.`` prefix outside the
    match, producing ``alias.NULLIF(col, '')::TYPE`` — which PostgreSQL reads as
    a schema-qualified function call and raises InvalidSchemaName.
    """
    _TS = r'(TIMESTAMP(?:TZ|(?:\s+WITH(?:OUT)?\s+TIME\s+ZONE))?|DATE)'
    # Already-wrapped expressions — don't touch
    _already = re.compile(r"NULLIF\s*\(", re.IGNORECASE)

    def _wrap(expr: str, typ: str) -> str:
        expr = expr.strip()
        if _already.match(expr):
            return f"{expr}::{typ}"
        return f"NULLIF({expr}, '')::{typ.strip()}"

    # 1. CAST(expr AS TIMESTAMP/DATE) — bare column or alias.col
    sql = re.sub(
        rf'CAST\(\s*(\w+(?:\.\w+)?)\s+AS\s+{_TS}\s*\)',
        lambda m: _wrap(m.group(1), m.group(2)),
        sql, flags=re.IGNORECASE,
    )

    # 2. [alias.]col::TIMESTAMP/DATE
    #
    # Pattern: optional `alias.` prefix + column name + ::TYPE
    # The alias. group is captured so the full `alias.col` expression
    # is passed into NULLIF — preventing the orphaned-prefix bug where
    # `oi.col::timestamp` became `oi.NULLIF(col,'')::timestamp`.
    #
    # Negative lookbehind (?<!\w) anchors the match without consuming
    # a preceding word character (replaces \b which fired at the dot).
    _SKIP = {"NOW", "CURRENT_TIMESTAMP", "CURRENT_DATE", "INTERVAL",
             "NULLIF", "COALESCE", "CAST", "TRUE", "FALSE"}

    def _replace_bare(m: re.Match) -> str:
        alias   = m.group(1)  # e.g. "oi" — may be None
        col     = m.group(2)  # e.g. "shipping_limit_date"
        typ     = m.group(3)  # e.g. "TIMESTAMP"
        if col.upper() in _SKIP:
            return m.group(0)
        full_expr = f"{alias}.{col}" if alias else col
        return _wrap(full_expr, typ)

    sql = re.sub(
        rf'(?<!\w)(?:(\w+)\.)?(\w+)::{_TS}',
        _replace_bare,
        sql, flags=re.IGNORECASE,
    )

    # 3. Rewrite interval-to-days patterns that the LLM commonly gets wrong.
    #
    # timestamp - timestamp  →  interval  (PostgreSQL type system)
    #
    # Two broken forms the LLM emits:
    #   a) expr::numeric / 86400          — interval cannot be cast to numeric
    #   b) expr::interval / INTERVAL '1 day' — interval / interval unsupported
    #
    # Both are rewritten to: EXTRACT(EPOCH FROM expr) / 86400
    # which extracts seconds from the interval and converts to fractional days.
    sql = _pg_rewrite_interval_to_days(sql)

    return sql


_SQL_KEYWORDS = frozenset({
    "SELECT", "FROM", "WHERE", "JOIN", "ON", "AND", "OR", "NOT", "IN", "IS",
    "GROUP", "BY", "ORDER", "HAVING", "LIMIT", "OFFSET", "AS", "WITH",
    "INNER", "LEFT", "RIGHT", "FULL", "OUTER", "CROSS", "NATURAL",
    "UNION", "INTERSECT", "EXCEPT", "ALL", "DISTINCT", "EXISTS",
    "CASE", "WHEN", "THEN", "ELSE", "END", "BETWEEN", "LIKE", "ILIKE",
    "NULL", "TRUE", "FALSE", "ASC", "DESC", "NULLS", "FIRST", "LAST",
    "OVER", "PARTITION", "ROW_NUMBER", "RANK", "DENSE_RANK", "NTILE",
    "LAG", "LEAD", "FIRST_VALUE", "LAST_VALUE", "ROWS", "RANGE",
    "UNBOUNDED", "PRECEDING", "FOLLOWING", "CURRENT", "ROW",
    "COUNT", "SUM", "AVG", "MIN", "MAX", "STDDEV", "VARIANCE",
    "COALESCE", "NULLIF", "CAST", "TRY_CAST", "EXTRACT", "DATE_TRUNC",
    "DATE_DIFF", "DATE_PART", "INTERVAL", "DATE", "TIMESTAMP", "TIME",
    "CONCAT", "TRIM", "LOWER", "UPPER", "LENGTH", "REPLACE", "SUBSTR",
    "GENERATE_SERIES", "UNNEST", "FILTER", "WITHIN", "PERCENTILE_CONT",
    "PERCENTILE_DISC", "QUALIFY", "PIVOT", "UNPIVOT", "INSERT", "UPDATE",
    "DELETE", "CREATE", "DROP", "ALTER", "TABLE", "VIEW", "INDEX",
    "RETURNING", "SET", "VALUES", "INTO", "PRIMARY", "FOREIGN", "KEY",
    "REFERENCES", "CONSTRAINT", "UNIQUE", "CHECK", "DEFAULT",
})


def _check_column_ambiguity(sql: str, schema: "SchemaContext") -> list[str]:
    """Warn about:
    1. Unqualified columns that exist in 2+ tables referenced by the query
       (these cause 'column is ambiguous' errors at runtime).
    2. Column names used in SQL that fuzzy-match schema columns but differ in case/spelling.
    """
    warnings: list[str] = []

    # ── Build per-table column maps ──────────────────────────────────────────
    # col_name (lower) → set of table names that own it
    col_to_tables: dict[str, set[str]] = {}
    for t in schema.tables:
        for c in t.columns:
            col_to_tables.setdefault(c.name.lower(), set()).add(t.name)

    all_col_names = {c.name for t in schema.tables for c in t.columns}
    norm_to_exact: dict[str, str] = {_normalise(c): c for c in all_col_names}

    # ── Find tables referenced in this query (simple heuristic) ─────────────
    table_names_lower = {t.name.lower() for t in schema.tables}
    sql_upper = sql.upper()
    referenced_tables: set[str] = set()
    for t in schema.tables:
        # Match table name appearing after FROM / JOIN keyword
        if re.search(
            r'\b(?:FROM|JOIN)\s+' + re.escape(t.name),
            sql,
            re.IGNORECASE,
        ):
            referenced_tables.add(t.name)

    has_join = bool(re.search(r'\bJOIN\b', sql, re.IGNORECASE))

    # ── Tokenise SQL to find bare (unqualified) identifiers ─────────────────
    # Qualified: alias.col or table.col — skip these entirely
    qualified = {m.group(2).lower() for m in re.finditer(
        r'\b([A-Za-z_]\w*)\s*\.\s*([A-Za-z_]\w*)', sql
    )}

    tokens = re.findall(r'"([^"]+)"|\'([^\']+)\'|([A-Za-z_]\w*)', sql)
    used_bare: set[str] = set()
    for q1, q2, bare in tokens:
        name = q1 or q2 or bare
        if not name:
            continue
        if name.upper() in _SQL_KEYWORDS:
            continue
        if name.lower() in table_names_lower:
            continue  # it's a table name, not a column
        if name.lower() in qualified:
            continue  # already qualified elsewhere — skip
        used_bare.add(name)

    # ── Check 1: ambiguous columns (exist in 2+ referenced tables) ──────────
    if has_join and referenced_tables:
        for name in used_bare:
            tables_with_col = col_to_tables.get(name.lower(), set())
            # Only flag if the column appears in 2+ of the REFERENCED tables
            colliding = tables_with_col & referenced_tables
            if len(colliding) >= 2:
                warnings.append(
                    f"⚠️ Column **'{name}'** exists in multiple joined tables "
                    f"({', '.join(sorted(colliding))}) — qualify it as "
                    f"e.g. **`{sorted(colliding)[0]}.{name}`** to avoid an ambiguity error."
                )

    # ── Check 2: fuzzy-match misspellings ────────────────────────────────────
    for name in used_bare:
        if name in all_col_names:
            continue
        norm = _normalise(name)
        if norm in norm_to_exact:
            exact = norm_to_exact[norm]
            if exact != name:
                warnings.append(
                    f"⚠️ Column **'{name}'** not found — did you mean **'{exact}'**?"
                )

    return warnings
