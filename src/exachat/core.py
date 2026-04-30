"""Core ExasolChat engine.

Connects schema introspection, LLM generation, RAG memory, safety
validation, query execution, and chart suggestion into a single
`.ask()` interface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from exachat.charts import auto_chart
from exachat.connection import ConnectionConfig, DatabaseConnection
from exachat.llm import LLMBackend, LLMResponse, OllamaBackend
from exachat.rag import NoopRAGMemory, RAGMemory
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
    rag_examples_used: int = 0
    column_warnings: list[str] = None  # ambiguous column name hints

    def __post_init__(self):
        if self.column_warnings is None:
            self.column_warnings = []


class ExasolChat:
    """Main interface. Connect a database + LLM, ask questions, get answers.

    Usage:
        chat = ExasolChat("exa+pyexasol://user:pass@host:8563/schema")
        result = chat.ask("What are the top 10 customers by revenue?")
        print(result.data)

    Or with explicit config:
        from exachat.connection import ConnectionConfig
        config = ConnectionConfig.exasol("host:8563", "user", "pass", "MY_SCHEMA")
        chat = ExasolChat(config)
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
        rag: Optional[RAGMemory] = None,
        rag_enabled: bool = True,
        chart_library: str = "auto",
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

        # RAG memory
        if rag_enabled and rag is None:
            self.rag = RAGMemory()
        elif rag is not None:
            self.rag = rag
        else:
            self.rag = NoopRAGMemory()

        # Schema introspection
        if self._db.is_exasol:
            self.schema_context = introspect_exasol(
                self._db.pyexasol_conn,
                schema=schema or self._config.exasol_schema,
                include_tables=include_tables,
                exclude_tables=exclude_tables,
            )
        elif self._db.is_duckdb:
            self.schema_context = introspect_duckdb(
                self._db.duckdb_conn,
                schema=schema,
                include_tables=include_tables,
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

        # 1. RAG retrieval — find similar past queries
        rag_examples = []
        rag_prompt = None
        try:
            rag_examples = self.rag.search(question)
            if rag_examples:
                rag_prompt = self.rag.format_for_prompt(rag_examples)
        except Exception:
            pass  # RAG failure should never block a query

        # 2. LLM generates SQL
        try:
            llm_resp: LLMResponse = self.llm.generate_sql(
                self.schema_prompt, question, rag_prompt,
            )
        except Exception as e:
            return self._error_result(
                question, "", f"LLM generation failed: {e}"
            )

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
                rag_examples_used=len(rag_examples),
                column_warnings=column_warnings,
            )
            self._history.append(result)
            return result

        # 4. Execute query
        try:
            df = self._db.execute_query(sql, self.max_rows)
        except Exception as e:
            result = QueryResult(
                question=question, sql=sql, safety=verdict,
                error=f"Query execution failed: {e}",
                explanation=llm_resp.explanation,
                rag_examples_used=len(rag_examples),
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

        # 7. Store in RAG memory (only if query succeeded)
        try:
            self.rag.add(question, sql)
        except Exception:
            pass  # RAG write failure should never surface

        result = QueryResult(
            question=question, sql=sql, safety=verdict,
            data=df, summary=summary,
            chart_config=chart_config, chart_obj=chart_obj,
            explanation=llm_resp.explanation,
            rag_examples_used=len(rag_examples),
            column_warnings=column_warnings,
        )
        self._history.append(result)
        return result

    def train(self, question: str, sql: str) -> None:
        """Manually add a Q&A pair to RAG memory."""
        self.rag.add(question, sql)

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
    """Normalise a column name for fuzzy comparison: lowercase, strip spaces/underscores."""
    return re.sub(r"[\s_]+", "", name.lower())


def _check_column_ambiguity(sql: str, schema: "SchemaContext") -> list[str]:
    """Detect column names in SQL that fuzzy-match schema columns but don't match exactly.

    Returns warning strings like: "Did you mean 'Order Date'? (used as 'order_date')"
    """
    all_cols = {c.name for t in schema.tables for c in t.columns}
    norm_to_exact: dict[str, str] = {_normalise(c): c for c in all_cols}

    # Extract bare identifiers and quoted identifiers from SQL
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
            continue  # exact match — fine
        norm = _normalise(name)
        if norm in norm_to_exact:
            exact = norm_to_exact[norm]
            if exact != name:
                warnings.append(
                    f"⚠️ Column **'{name}'** not found — did you mean **'{exact}'**?"
                )
    return warnings
