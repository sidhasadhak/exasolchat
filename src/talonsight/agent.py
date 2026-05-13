"""Hermes Agent — autonomous multi-step data analyst for talonsight.

The AgentLoop drives an iterative investigation: the LLM calls tools, reads
results, forms hypotheses, runs more queries, and ultimately calls final_answer
with a plain-English narrative and the key SQL used.

Architecture principles baked in for long-term evolution:
  - async run() / sync run_sync() — scheduler in v2 calls run() directly
  - explicit create_plan as mandatory step 0 — visible, auditable reasoning
  - capability layer (AgentCapabilities) — business-level tools, not raw SQL
  - BusinessModel store — findings persist across sessions from day one
  - schema_context.semantic() interface — v2 enriches without touching agent logic
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import pandas as pd

from talonsight.capabilities import AgentCapabilities
from talonsight.memory import BusinessModel, Finding

if TYPE_CHECKING:
    from talonsight.connection import DatabaseConnection
    from talonsight.kb import KnowledgeBase
    from talonsight.llm import LLMBackend
    from talonsight.schema import SchemaContext

logger = logging.getLogger(__name__)

# ── Tool schemas (OpenAI function-calling format) ────────────────────────────
# Hermes3, llama3.1, qwen2.5, mistral-nemo all understand this format via Ollama.

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "create_plan",
            "description": (
                "ALWAYS call this first before any other tool. "
                "Declare your investigation plan as an ordered list of steps. "
                "No data access until the plan is recorded."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Ordered investigation steps in plain English.",
                    }
                },
                "required": ["steps"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_tables",
            "description": "List all available tables with their row counts.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_schema",
            "description": "Get column definitions (name, type, nullable) for one or more tables.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Table names. Empty list returns all tables.",
                    }
                },
                "required": ["tables"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sample_data",
            "description": (
                "Fetch a small sample of rows from a table. "
                "Use before filtering on categorical columns — never assume value formats."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string", "description": "Table name."},
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columns to include. Empty = all (capped at 6).",
                    },
                    "n": {"type": "integer", "description": "Number of rows (default 5, max 20)."},
                },
                "required": ["table"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_column_stats",
            "description": (
                "Get min, max, avg, null count, distinct count, and top 5 values for a column. "
                "Use before aggregating numeric columns or filtering on categoricals."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string", "description": "Table name."},
                    "column": {"type": "string", "description": "Column name."},
                },
                "required": ["table", "column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": (
                "Execute a SELECT query and return results as a markdown table. "
                "On error, the full error message is returned so you can fix and retry. "
                "Only SELECT is allowed — writes are blocked."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "The SELECT query to execute."},
                    "limit": {
                        "type": "integer",
                        "description": "Max rows to return (default 200).",
                    },
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the SQL pattern knowledge base for relevant techniques. "
                "Use before writing window functions, CTEs, date arithmetic, or "
                "any pattern you are not fully confident about."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language description of the SQL pattern needed.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_drivers",
            "description": (
                "Decompose what is driving movement in a metric across given dimensions. "
                "Use for 'what is causing X?' style questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "description": "Column expression to aggregate, e.g. 'SUM(revenue)'.",
                    },
                    "date_range": {
                        "type": "string",
                        "description": "Time window, e.g. 'last 7 days'.",
                    },
                    "dimensions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columns to segment by, e.g. ['region', 'product_tier'].",
                    },
                },
                "required": ["metric", "date_range", "dimensions"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_change",
            "description": (
                "Statistically verify whether a metric genuinely shifted or is noise. "
                "Use for 'did X change?' style questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "description": "Metric to inspect."},
                    "timeframe": {
                        "type": "string",
                        "description": "Period to inspect, e.g. 'last 7 days'.",
                    },
                    "comparison": {
                        "type": "string",
                        "description": "Baseline to compare against, e.g. 'prior 7 days'.",
                    },
                },
                "required": ["metric", "timeframe"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": (
                "Call this when you have a complete, verified answer. "
                "Terminates the investigation. "
                "narrative must be plain English — 2-4 sentences stating what you found "
                "and what it means. No markdown, no bullet points."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "narrative": {
                        "type": "string",
                        "description": "2-4 sentence plain-English finding.",
                    },
                    "sql": {
                        "type": "string",
                        "description": "The most important SQL query from the investigation.",
                    },
                    "chart_hint": {
                        "type": "string",
                        "enum": ["bar", "line", "scatter", "pie", "area", "table_only"],
                        "description": "Best chart type for the result. Omit if not applicable.",
                    },
                },
                "required": ["narrative"],
            },
        },
    },
]

# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class AgentStep:
    """A single tool call made during an agent investigation."""
    step_num: int
    tool_name: str
    tool_input: dict
    tool_output: str          # truncated to 2000 chars for large result sets
    elapsed_ms: int
    error: Optional[str] = None

    @property
    def is_sql_step(self) -> bool:
        return self.tool_name == "run_sql"

    @property
    def is_plan_step(self) -> bool:
        return self.tool_name == "create_plan"


@dataclass
class AgentResult:
    """The complete output of an agent investigation."""
    question: str
    narrative: str
    sql: Optional[str]              # last successful run_sql (Option A)
    data: Optional[pd.DataFrame]    # last successful run_sql DataFrame
    steps: list[AgentStep]
    plan: list[str]                 # from create_plan step
    chart_hint: Optional[str]
    error: Optional[str]
    total_elapsed_ms: int


# ── Agent loop ────────────────────────────────────────────────────────────────

class AgentLoop:
    """Drives the multi-step tool-calling investigation loop.

    Call run_sync() from synchronous code (the Streamlit UI).
    Call run() directly from async code (the v2 scheduler).
    """

    MAX_STEPS = 12
    # Truncate tool outputs larger than this before appending to message history.
    # Keeps context windows from blowing up on large query results.
    _MAX_OUTPUT_CHARS = 3000

    def __init__(
        self,
        connector: "DatabaseConnection",
        schema_context: "SchemaContext",
        llm: "LLMBackend",
        business_model: BusinessModel,
        kb: Optional["KnowledgeBase"] = None,
        dialect: str = "",
        allowed_schemas: Optional[list[str]] = None,
        allowed_tables: Optional[list[str]] = None,
        max_steps: int = MAX_STEPS,
        schema_str: str = "",
    ) -> None:
        self._connector = connector
        self._schema = schema_context
        self._llm = llm
        self._bm = business_model
        self._kb = kb
        self._dialect = dialect
        self._allowed_schemas = allowed_schemas or []
        self._allowed_tables = allowed_tables or []
        self._max_steps = max_steps
        self._caps = AgentCapabilities(self)
        # Pre-built schema string injected into every system prompt.
        # Empty string = agent must discover schema via tools (cold start fallback).
        self._schema_str = schema_str

        # Investigation state — reset on each run()
        self._done: bool = False
        self._plan: list[str] = []
        self._final_narrative: str = ""
        self._final_chart_hint: Optional[str] = None
        self._last_sql: Optional[str] = None
        self._last_df: Optional[pd.DataFrame] = None

    # ── Public entry points ───────────────────────────────────────────

    async def run(
        self,
        question: str,
        on_step: Optional[Any] = None,
    ) -> AgentResult:
        """Async investigation loop. Scheduler calls this directly in v2."""
        t0 = time.time()
        self._reset_state()

        messages: list[dict] = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": question},
        ]
        steps: list[AgentStep] = []
        agent_error: Optional[str] = None

        for step_num in range(1, self._max_steps + 1):
            try:
                response = self._llm.chat_with_tools(messages, TOOLS)
            except Exception as exc:
                agent_error = f"LLM error on step {step_num}: {exc}"
                logger.error(agent_error)
                break

            # Model returned plain text (no tool calls) — treat as narrative
            if not response.tool_calls:
                if response.content:
                    self._final_narrative = response.content
                break

            # Execute each tool call in this response
            for tc in response.tool_calls:
                ts = time.time()
                tool_name = tc["name"]
                tool_args = tc["arguments"]

                tool_output = ""
                tool_error = None
                try:
                    tool_output = self._execute_tool(tool_name, tool_args)
                except Exception as exc:
                    tool_error = str(exc)
                    tool_output = f"TOOL ERROR: {exc}"
                    logger.warning("Tool %s raised: %s", tool_name, exc)

                elapsed = int((time.time() - ts) * 1000)
                step = AgentStep(
                    step_num=step_num,
                    tool_name=tool_name,
                    tool_input=tool_args,
                    tool_output=tool_output[:self._MAX_OUTPUT_CHARS],
                    elapsed_ms=elapsed,
                    error=tool_error,
                )
                steps.append(step)

                if on_step:
                    try:
                        if asyncio.iscoroutinefunction(on_step):
                            await on_step(step)
                        else:
                            on_step(step)
                    except Exception:
                        pass  # never let the callback crash the loop

                # Append the assistant message (verbatim) + tool result to history
                messages.append(response.raw_assistant_msg)
                messages.append(
                    self._llm.build_tool_result_msg(tc, tool_output[:self._MAX_OUTPUT_CHARS])
                )

                if self._done:
                    break

            if self._done:
                break

        total_ms = int((time.time() - t0) * 1000)

        # Persist confirmed finding to BusinessModel
        if self._final_narrative and not agent_error:
            self._bm.record_finding(Finding(
                question=question,
                narrative=self._final_narrative,
                sql=self._last_sql or "",
                tables_used=self._extract_tables(self._last_sql or ""),
            ))

        if not self._final_narrative:
            self._final_narrative = (
                "Investigation reached the step limit without a definitive conclusion. "
                "The partial results above represent the best available analysis."
            ) if not agent_error else agent_error

        return AgentResult(
            question=question,
            narrative=self._final_narrative,
            sql=self._last_sql,
            data=self._last_df,
            steps=steps,
            plan=self._plan,
            chart_hint=self._final_chart_hint,
            error=agent_error,
            total_elapsed_ms=total_ms,
        )

    def run_sync(
        self,
        question: str,
        on_step: Optional[Any] = None,
    ) -> AgentResult:
        """Synchronous entry point used by the Streamlit UI and ask_agent()."""
        return asyncio.run(self.run(question, on_step=on_step))

    # ── Tool dispatch ─────────────────────────────────────────────────

    def _execute_tool(self, name: str, args: dict) -> str:
        dispatch = {
            "create_plan":           self._tool_create_plan,
            "list_tables":           self._tool_list_tables,
            "get_schema":            self._tool_get_schema,
            "get_sample_data":       self._tool_get_sample_data,
            "get_column_stats":      self._tool_get_column_stats,
            "run_sql":               self._tool_run_sql,
            "search_knowledge_base": self._tool_search_kb,
            "find_drivers":          self._tool_find_drivers,
            "detect_change":         self._tool_detect_change,
            "final_answer":          self._tool_final_answer,
        }
        fn = dispatch.get(name)
        if fn is None:
            return f"Unknown tool: {name}. Available: {list(dispatch.keys())}"
        return fn(args)

    # ── Tool implementations ──────────────────────────────────────────

    def _tool_create_plan(self, args: dict) -> str:
        steps = args.get("steps", [])
        if not isinstance(steps, list):
            steps = [str(steps)]
        self._plan = [str(s) for s in steps]
        numbered = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(self._plan))
        return f"Plan recorded ({len(self._plan)} steps):\n{numbered}"

    def _tool_list_tables(self, _args: dict) -> str:
        try:
            tables = self._schema.tables
            if not tables:
                return "No tables found in schema."
            lines = []
            for tbl in tables:
                try:
                    count_sql = f'SELECT COUNT(*) AS n FROM "{tbl.name}"'
                    df = self._connector.execute_query(count_sql)
                    n = int(df.iloc[0, 0]) if df is not None and len(df) > 0 else "?"
                except Exception:
                    n = "?"
                lines.append(f"- {tbl.name} ({n} rows)")
            return "\n".join(lines)
        except Exception as exc:
            return f"Could not list tables: {exc}"

    def _tool_get_schema(self, args: dict) -> str:
        tables_filter = args.get("tables", [])
        try:
            all_tables = self._schema.tables
            if tables_filter:
                tl = [t.lower() for t in tables_filter]
                all_tables = [t for t in all_tables if t.name.lower() in tl]
            if not all_tables:
                return f"No tables found matching: {tables_filter}"
            lines = []
            for tbl in all_tables:
                lines.append(f"\nTable: {tbl.name}")
                for col in tbl.columns:
                    # ColumnInfo is a dataclass — use attribute access, not dict
                    nullable = "nullable" if col.nullable else "not null"
                    lines.append(f"  {col.name}  {col.type}  ({nullable})")
            return "\n".join(lines)
        except Exception as exc:
            return f"Could not get schema: {exc}"

    def _tool_get_sample_data(self, args: dict) -> str:
        table = args.get("table", "")
        columns = args.get("columns") or []
        n = min(int(args.get("n", 5)), 20)

        if not table:
            return "table argument is required."
        try:
            if columns:
                cols_sql = ", ".join(f'"{c}"' for c in columns[:6])
            else:
                # pick first 6 columns from schema to avoid wide result sets
                tbl_info = next(
                    (t for t in self._schema.tables if t.name.lower() == table.lower()), None
                )
                if tbl_info:
                    cols_sql = ", ".join(
                        f'"{c.name}"' for c in tbl_info.columns[:6]
                    )
                else:
                    cols_sql = "*"
            sql = f'SELECT {cols_sql} FROM "{table}" LIMIT {n}'
            df = self._connector.execute_query(sql)
            if df is None or df.empty:
                return f"No rows returned from {table}."
            return df.to_markdown(index=False)
        except Exception as exc:
            return f"Could not sample {table}: {exc}"

    def _tool_get_column_stats(self, args: dict) -> str:
        table = args.get("table", "")
        column = args.get("column", "")
        if not table or not column:
            return "Both table and column are required."

        dialect = self._dialect.lower()
        try:
            # Determine column type from schema
            tbl_info = next(
                (t for t in self._schema.tables if t.name.lower() == table.lower()), None
            )
            col_type = ""
            if tbl_info:
                col_info = next(
                    (c for c in tbl_info.columns if c.name.lower() == column.lower()), None
                )
                col_type = (col_info.type if col_info else "").lower()

            is_numeric = any(k in col_type for k in (
                "int", "float", "double", "numeric", "decimal", "real", "number"
            ))
            is_text = any(k in col_type for k in ("char", "text", "string", "varchar"))

            lines = []

            # Null stats — always
            null_sql = (
                f'SELECT COUNT(*) AS total, '
                f'SUM(CASE WHEN "{column}" IS NULL THEN 1 ELSE 0 END) AS null_count, '
                f'COUNT(DISTINCT "{column}") AS distinct_count '
                f'FROM "{table}"'
            )
            df = self._connector.execute_query(null_sql)
            if df is not None and len(df) > 0:
                total = int(df.iloc[0]["total"])
                nulls = int(df.iloc[0]["null_count"])
                distinct = int(df.iloc[0]["distinct_count"])
                null_pct = round(100 * nulls / total, 1) if total else 0
                lines.append(
                    f"total={total}, nulls={nulls} ({null_pct}%), distinct={distinct}"
                )

            # Numeric stats
            if is_numeric:
                num_sql = (
                    f'SELECT MIN("{column}") AS min_val, MAX("{column}") AS max_val, '
                    f'AVG("{column}") AS avg_val FROM "{table}"'
                )
                df2 = self._connector.execute_query(num_sql)
                if df2 is not None and len(df2) > 0:
                    r = df2.iloc[0]
                    lines.append(
                        f"min={r['min_val']}, max={r['max_val']}, "
                        f"avg={round(float(r['avg_val']), 2) if r['avg_val'] is not None else 'N/A'}"
                    )

            # Top values — text or low-cardinality
            if is_text or distinct <= 50:
                top_sql = (
                    f'SELECT "{column}" AS value, COUNT(*) AS freq '
                    f'FROM "{table}" WHERE "{column}" IS NOT NULL '
                    f'GROUP BY "{column}" ORDER BY freq DESC LIMIT 5'
                )
                df3 = self._connector.execute_query(top_sql)
                if df3 is not None and len(df3) > 0:
                    top = ", ".join(
                        f'"{row["value"]}" ({row["freq"]})' for _, row in df3.iterrows()
                    )
                    lines.append(f"top values: {top}")

            return f"Column {table}.{column}:\n" + "\n".join(lines) if lines else "No stats available."
        except Exception as exc:
            return f"Could not get stats for {table}.{column}: {exc}"

    def _tool_run_sql(self, args: dict) -> str:
        sql = args.get("sql", "").strip()
        limit = min(int(args.get("limit", 200)), 500)

        if not sql:
            return "sql argument is required."

        # Safety check
        from talonsight.safety import validate_sql, RiskLevel
        verdict = validate_sql(
            sql,
            allowed_schemas=self._allowed_schemas,
            allowed_tables=self._allowed_tables,
        )
        if verdict.level == RiskLevel.BLOCKED:
            return f"BLOCKED: {verdict.reason}"

        # Apply PostgreSQL post-processing if needed
        from talonsight.core import TalonSight
        if "postgresql" in self._dialect.lower() or "postgres" in self._dialect.lower():
            sql = TalonSight._pg_post_process(sql)

        # Inject LIMIT if not present
        if re.search(r'\bLIMIT\b', sql, re.IGNORECASE) is None:
            sql = sql.rstrip(";") + f"\nLIMIT {limit}"

        try:
            df = self._connector.execute_query(sql)
            if df is None or df.empty:
                return "Query executed successfully but returned no rows."

            # Store for Option A — last successful result is the agent's main answer
            self._last_sql = args.get("sql", sql)  # store the original (no injected LIMIT)
            self._last_df = df

            row_note = (
                f"\n\n*Showing {len(df)} rows (LIMIT {limit} applied)*"
                if len(df) >= limit else ""
            )
            return df.to_markdown(index=False) + row_note

        except Exception as exc:
            return f"SQL ERROR: {exc}\n\nFailed SQL:\n{sql}"

    def _tool_search_kb(self, args: dict) -> str:
        query = args.get("query", "")
        if not query:
            return "query argument is required."
        if self._kb is None:
            return "Knowledge base not available."
        try:
            results = self._kb.search(query, n_results=4)
            return self._kb.format_for_prompt(results) if results else "No relevant patterns found."
        except Exception as exc:
            return f"KB search error: {exc}"

    def _tool_find_drivers(self, args: dict) -> str:
        return self._caps.find_drivers(
            metric=args.get("metric", ""),
            date_range=args.get("date_range", ""),
            dimensions=args.get("dimensions", []),
        )

    def _tool_detect_change(self, args: dict) -> str:
        return self._caps.detect_change(
            metric=args.get("metric", ""),
            timeframe=args.get("timeframe", ""),
            comparison=args.get("comparison", "prior period"),
        )

    def _tool_final_answer(self, args: dict) -> str:
        self._final_narrative = args.get("narrative", "")
        self._final_chart_hint = args.get("chart_hint")
        # If agent provided a better SQL, prefer it — otherwise keep _last_sql (Option A)
        if args.get("sql"):
            self._last_sql = args["sql"]
        self._done = True
        return "Answer recorded. Investigation complete."

    # ── System prompt ─────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        past_context = self._bm.get_context(max_findings=5)
        past_block = (
            f"\nCONFIRMED FINDINGS FROM PAST INVESTIGATIONS:\n{past_context}\n"
            "Use these as trusted baselines. Do not re-derive what is already known.\n"
            if past_context else ""
        )
        dialect_name = self._dialect or "SQL"

        # Schema is pre-built from introspection — inject it so the agent starts
        # knowing all tables/columns and can skip list_tables / get_schema entirely.
        if self._schema_str:
            schema_block = f"\n{self._schema_str}\n"
            schema_instruction = (
                "The full schema is provided above — you already know every table and column. "
                "Do NOT call list_tables or get_schema. "
                "Start with create_plan, then go straight to get_sample_data or run_sql."
            )
        else:
            schema_block = ""
            schema_instruction = (
                "Call list_tables or get_schema to understand the available data."
            )

        return f"""You are an autonomous data analyst with access to a database via tools.
Investigate questions step-by-step using the provided tools, then call final_answer with your findings.
DATABASE DIALECT: {dialect_name}
{schema_block}{past_block}
STEPS:
1. Call create_plan with your investigation steps first.
2. {schema_instruction}
3. Call get_sample_data before filtering on text/category columns to understand value formats.
4. Call run_sql to execute queries — read errors and retry on failure.
5. Call final_answer with a 2-4 sentence plain-English narrative when done.

RULES: Only SELECT in run_sql. Max {self._max_steps} tool calls. Use only real table/column names."""

    # ── Helpers ───────────────────────────────────────────────────────

    def _reset_state(self) -> None:
        self._done = False
        self._plan = []
        self._final_narrative = ""
        self._final_chart_hint = None
        self._last_sql = None
        self._last_df = None

    @staticmethod
    def _extract_tables(sql: str) -> list[str]:
        """Best-effort extraction of table names from a SQL string."""
        return list({
            m.strip('"').strip("'")
            for m in re.findall(r'\bFROM\s+"?(\w+)"?|\bJOIN\s+"?(\w+)"?', sql, re.IGNORECASE)
            for m in m if m
        })
