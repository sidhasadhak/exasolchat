"""LLM backends for text-to-SQL generation.

Supports Ollama (default), any OpenAI-compatible API (LM Studio, vLLM, etc.),
and Apple Silicon MLX-LM server.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    sql: str
    explanation: str
    raw: str


@dataclass
class ToolResponse:
    """Response from chat_with_tools() — either tool calls or a plain text reply.

    tool_calls items are normalised across backends:
        {"name": str, "arguments": dict, "id": str | None}

    raw_assistant_msg is the verbatim assistant message from the API response.
    Append it directly to the message history before adding tool-result messages.
    """
    content: str             # non-empty when model replies with text (no tool calls)
    tool_calls: list[dict]   # normalised tool calls
    raw_assistant_msg: dict  # verbatim — append to history as-is


class LLMBackend(ABC):
    """Abstract LLM backend."""

    def ping(self) -> tuple[bool, str]:
        """Return (reachable, message). Override in each backend."""
        return False, "ping not implemented"

    def supports_tool_calling(self) -> tuple[bool, str]:
        """Return (supported, message). Override in backends that can detect this."""
        return True, ""

    def build_tool_result_msg(self, tool_call: dict, result: str) -> dict:
        """Build the tool-result message to append after executing a tool call.

        Ollama needs no tool_call_id; OpenAI-compatible backends do.
        Subclasses override when the backend requires tool_call_id.
        """
        return {"role": "tool", "content": str(result)}

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> "ToolResponse":
        """Multi-turn conversation with tool-calling support.

        Subclasses must override this to support agent mode.
        Default raises NotImplementedError so classic mode still works fine.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement chat_with_tools(). "
            "Agent mode requires an Ollama or OpenAI-compatible backend."
        )

    @abstractmethod
    def generate_sql(
        self, schema_prompt: str, question: str,
        kb_context: Optional[str] = None,
        history: Optional[list[dict]] = None,
    ) -> LLMResponse:
        ...

    @abstractmethod
    def generate_summary(self, question: str, sql: str, data_preview: str) -> str:
        ...

    @abstractmethod
    def suggest_chart(self, question: str, columns: list[str], row_count: int) -> dict:
        ...

    @abstractmethod
    def suggest_followups(self, question: str, sql: str, data_preview: str) -> list[str]:
        ...

    @abstractmethod
    def generate_explore_questions(self, schema_prompt: str, profile: str) -> list[str]:
        ...

    _DUCKDB_DIALECT_HINTS = """
DuckDB SQL dialect — apply these rules when the dialect is duckdb:
- PostgreSQL-based. Use LIMIT not TOP.
- Casting: CAST(x AS TYPE) or x::TYPE. TRY_CAST returns NULL on failure instead of an error.
- Date/time:
    date_trunc('month', col), date_diff('day', start, end), date_part('year', col),
    EXTRACT(year FROM col), strftime(col, '%Y-%m-%d'), strptime(str, '%Y-%m-%d'),
    today(), now(), current_date, col + INTERVAL '1 day'
    Parts: year, month, day, quarter, week, weekday, hour, minute, second, epoch
- CRITICAL — VARCHAR date columns: date_trunc(), date_diff(), EXTRACT() and all date functions
    require a DATE or TIMESTAMP argument. If the schema shows a date column typed as
    VARCHAR / TEXT / CHARACTER VARYING, you MUST cast it first:
        date_trunc('month', CAST("Order Date" AS DATE))
        date_trunc('month', "Order Date"::DATE)
        date_diff('day', "Start Date"::DATE, "End Date"::DATE)
    Calling date_trunc('month', varchar_column) raises:
      "No function matches date_trunc(STRING_LITERAL, VARCHAR)"
    Any column whose name contains 'date', 'time', 'at', 'created', 'updated', 'day',
    'month', 'year' and whose schema type is VARCHAR must be cast to DATE or TIMESTAMP
    before use in any date function. When the schema marks a column with ⚠ Cast required,
    always wrap it with CAST(col AS DATE) or CAST(col AS TIMESTAMP).
- Identifiers are case-insensitive but preserve their stored case.
  ALWAYS use the exact column names from the schema — if schema shows "Order Date", write "Order Date";
  if it shows order_date, write order_date. Never guess or transform column names.
- GROUP BY ALL — groups by all non-aggregated SELECT columns automatically.
- ORDER BY ALL — sorts by all columns.
- SELECT * EXCLUDE(col1, col2) — wildcard minus named columns.
- SELECT * REPLACE(expr AS col) — wildcard with column overrides.
- QUALIFY — filter window function results without a subquery:
    SELECT * FROM t QUALIFY ROW_NUMBER() OVER (PARTITION BY x ORDER BY y) = 1
  ⚠ QUALIFY is ONLY for window functions. Use HAVING to filter aggregates (SUM, AVG, COUNT, etc.).
  ⚠ QUALIFY cannot be combined with GROUP BY ALL — use explicit GROUP BY columns if QUALIFY is needed.
- HAVING — filter aggregate results (always prefer over QUALIFY for non-window filters):
    SELECT customer, SUM(sales) AS total FROM t GROUP BY customer HAVING total > 1000
- Nested aggregates (AVG of COUNT) require a CTE — you cannot nest aggregate functions directly:
    WITH counts AS (SELECT customer, COUNT(*) AS n FROM t GROUP BY customer)
    SELECT AVG(n) FROM counts
- PIVOT / UNPIVOT — transpose rows to columns and back.
- UNION BY NAME — match union sides by column name, not position.
- string_split(col, delim), regexp_matches(col, pattern), string_agg(col, sep)
- list_agg(), array_agg(), unnest(col) for list/array columns.
- Nested types: STRUCT (dot access), LIST, MAP.
- Trailing commas in SELECT/FROM lists are valid syntax.
"""

    _POSTGRESQL_DIALECT_HINTS = """
PostgreSQL SQL dialect — apply these rules when the dialect is postgresql:
- Use LIMIT not TOP.
- Casting: CAST(x AS TYPE) or x::TYPE.
- EMPTY STRINGS vs NULL: Data loaded from CSV often stores missing values as '' (empty string)
  instead of NULL. For date/timestamp columns ALWAYS guard against empty strings:
    NULLIF(col, '')::timestamp        -- safe cast, returns NULL for '' instead of erroring
    WHERE col <> '' AND col IS NOT NULL  -- safe filter
  Never cast a date/timestamp column directly without NULLIF if the data came from CSV/flat files.
- Date/time arithmetic:
    Subtracting two timestamps returns an INTERVAL: ts2 - ts1
    To get days as a number: EXTRACT(epoch FROM (ts2 - ts1)) / 86400
    Or use: DATE_PART('day', ts2 - ts1)
    AGE(ts2, ts1) returns a human-readable interval.
    DATE_TRUNC('month', col), EXTRACT(year FROM col), NOW(), CURRENT_DATE
- String functions: CONCAT, ||, LOWER, UPPER, TRIM, SPLIT_PART, REGEXP_REPLACE
- NULL-safe aggregation: use FILTER (WHERE col IS NOT NULL) or COALESCE.
- Window functions: standard OVER (PARTITION BY ... ORDER BY ...) syntax.
- CTEs: WITH name AS (...) SELECT ...
- Use double-quotes for identifiers with spaces or mixed case; lowercase is case-insensitive.
"""

    def _build_sql_prompt(
        self, schema_prompt: str, question: str,
        kb_context: Optional[str] = None,
        history: Optional[list[dict]] = None,
    ) -> str:
        kb_section = ""
        if kb_context:
            kb_section = f"""
RELEVANT SQL PATTERNS (apply these techniques where appropriate):
{kb_context}
"""
        history_section = ""
        if history:
            turns = []
            for h in history:
                turns.append(f"Q: {h['question']}\nSQL:\n```sql\n{h['sql']}\n```")
            history_section = "\nCONVERSATION HISTORY (the user may be refining or following up on these):\n" + "\n\n".join(turns) + "\n"

        _sp_lower = schema_prompt.lower()
        if "duckdb" in _sp_lower:
            dialect_section = self._DUCKDB_DIALECT_HINTS
        elif "postgresql" in _sp_lower or "postgres" in _sp_lower:
            dialect_section = self._POSTGRESQL_DIALECT_HINTS
        else:
            dialect_section = "- For Exasol: use LIMIT, double-quote identifiers only if mixed case."

        return f"""You are a SQL expert. Given the database schema below, write a SQL query that answers the user's question.

RULES:
- Write ONLY a SELECT query. Never write INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, EXEC, CALL, EXPORT, IMPORT, or any DDL/DML.
- Use ONLY the exact table and column names from the schema below. Do not invent or rename columns.
- Column name matching: when the user refers to a column informally (e.g. "order date"), map it to the closest schema column (e.g. "Order Date" or "order_date"). Use the schema name exactly.
- Return SQL inside a ```sql code block.
- After the SQL, write 1-2 sentences explaining what it does.
- If the question cannot be answered from the schema, say so clearly instead of guessing.
- If the question is a follow-up (e.g. "add actual numbers", "also show X", "filter by Y"), modify the most recent SQL from conversation history to address it.
- Use appropriate aggregations, JOINs, filtering, and window functions.
- Alias columns for human readability.
- ALWAYS qualify column names with their table name or alias when the query contains any JOIN (e.g. orders.order_id, not just order_id). Ambiguous unqualified column references cause "column is ambiguous" errors at runtime.
SQL FORMATTING (important):
- Put each clause on its own line: SELECT, FROM, JOIN, WHERE, GROUP BY, HAVING, ORDER BY, LIMIT.
- Indent column lists and conditions with 4 spaces.
- Do NOT use -- inline comments anywhere in the SQL. They break execution when the query is minified.
- Do NOT use /* */ block comments either.
- Each selected column on its own line, comma at the end of the line (not the start).
{dialect_section}
{schema_prompt}
{kb_section}{history_section}
USER QUESTION: {question}"""

    def _build_summary_prompt(self, question: str, sql: str, data_preview: str) -> str:
        return f"""The user asked: "{question}"

This SQL was executed:
```sql
{sql}
```

Results (first rows):
{data_preview}

Write a concise plain-text summary. Rules: no markdown, no bold (**), no italics (*), no backticks (`), no bullet points. Be specific with numbers. 2-3 sentences max."""

    def _build_followups_prompt(self, question: str, sql: str, data_preview: str) -> str:
        return f"""A business analyst asked: "{question}"

SQL executed:
{sql}

Result preview:
{data_preview}

Suggest 3 specific, actionable follow-up questions they would naturally ask next — drilling deeper, comparing, or exploring anomalies in this data.
Return ONLY a JSON array of 3 strings. No markdown, no explanation.
["Question 1?", "Question 2?", "Question 3?"]"""

    def _build_explore_prompt(self, schema_prompt: str, profile: str) -> str:
        return f"""You are a data analyst. Given the database schema and data profile below, generate 5 insightful business questions a user would want to explore first.

{schema_prompt}

DATA PROFILE:
{profile}

Make questions specific, business-relevant, and varied — cover trends, top/bottom rankings, comparisons, and anomalies.
Return ONLY a JSON array of 5 strings. No markdown, no explanation.
["Question 1?", "Question 2?", "Question 3?", "Question 4?", "Question 5?"]"""

    def _build_chart_prompt(self, question: str, columns: list[str], row_count: int) -> str:
        return f"""Given a query result with columns {columns} and {row_count} rows,
for the question "{question}", suggest the best chart type.

Respond with ONLY a JSON object (no markdown fences, no explanation):
{{
  "chart_type": "bar" | "line" | "scatter" | "pie" | "area" | "heatmap" | "table_only",
  "x": "column_name",
  "y": "column_name_or_list",
  "color": "column_name_or_null",
  "title": "Chart Title"
}}
Use "table_only" if the data isn't well-suited for charting (e.g., single row, text-heavy)."""

    @staticmethod
    def _classify_sql_error(error: str) -> str:  # noqa: C901 (intentionally long)
        """Return a targeted diagnostic hint based on the database error message.

        These hints are injected into the fix prompt so the LLM knows exactly
        what class of fix is needed without having to infer it from the raw
        error string (which varies across databases).

        Checks are ordered most-specific → least-specific so that a narrow
        pattern wins over a broad one (e.g. "cannot cast type" fires before
        the generic invalid-syntax fallback).
        """
        e = error.lower()

        # ── Reference / name errors ───────────────────────────────────────────

        # PostgreSQL: alias mistaken for a schema name — common when LLM writes
        # alias.FUNCTION(col) instead of FUNCTION(alias.col)
        if "schema" in e and "does not exist" in e:
            return (
                "DIAGNOSIS: A table alias is being interpreted as a PostgreSQL schema name. "
                "This happens when a function call is prefixed with a table alias, e.g. "
                "`alias.NULLIF(col, '')` — SQL parsers read `alias` as a schema, not an alias. "
                "FIX: Move the alias inside the function argument: `NULLIF(alias.col, '')`."
            )

        # Missing JOIN / alias not declared (PG: "missing FROM-clause entry for table X")
        if "missing from-clause entry" in e or ("unknown table" in e and "from" not in e):
            return (
                "DIAGNOSIS: A table alias or table name is used in the query but was never "
                "declared in a FROM or JOIN clause. "
                "FIX: Add the missing JOIN, or correct the alias to match one already in the query."
            )

        if ("column" in e or "field" in e) and "does not exist" in e:
            return (
                "DIAGNOSIS: A column referenced in the query does not exist. "
                "Cross-check every column name against the DATABASE SCHEMA provided. "
                "Check for typos, wrong table alias, or a column that belongs to a different table."
            )

        if ("relation" in e or "table" in e) and "does not exist" in e:
            return (
                "DIAGNOSIS: A table or view name is wrong. "
                "Use only table names that appear in the DATABASE SCHEMA provided. "
                "Check capitalisation and schema prefix."
            )

        # PostgreSQL ROUND precision — must come before generic function-not-found check
        if ("round" in e and "does not exist" in e and
                ("double precision" in e or "float" in e or "real" in e)):
            return (
                "DIAGNOSIS: PostgreSQL's ROUND() only accepts NUMERIC for the two-argument form — "
                "ROUND(double precision, integer) does not exist. "
                "AVG(), SUM()/COUNT() division, and most arithmetic return double precision. "
                "FIX: Cast to numeric before rounding: ROUND(AVG(col)::numeric, 2) or "
                "ROUND(CAST(AVG(col) AS NUMERIC), 2). "
                "Apply this to every ROUND(expr, n) in the query where expr may be double precision."
            )

        if "function" in e and "does not exist" in e:
            return (
                "DIAGNOSIS: A function name or its argument types are wrong for this SQL dialect. "
                "FIX: Check the correct function name (e.g. STRFTIME vs TO_CHAR vs FORMAT), "
                "or add an explicit CAST on the argument to match the expected type."
            )

        if "ambiguous" in e and ("column" in e or "field" in e):
            return (
                "DIAGNOSIS: A column name is ambiguous — it exists in more than one joined table. "
                "FIX: Qualify every ambiguous column with its table alias (e.g. `t.column_name`)."
            )

        # ── Aggregation / grouping errors ─────────────────────────────────────

        if (
            "must appear in the group by" in e          # PostgreSQL
            or ("not in aggregate" in e and "group by" in e)  # MySQL / DuckDB
            or "is not in aggregate function and not in group by" in e
            or "not contained in either an aggregate function or the group by" in e
        ):
            return (
                "DIAGNOSIS: A non-aggregated column appears in the SELECT (or ORDER BY) "
                "but is missing from the GROUP BY clause. "
                "FIX: Either add the column to GROUP BY, or wrap it in an aggregate function "
                "such as MAX(), MIN(), or ANY_VALUE() if the value is the same per group."
            )

        if "cannot be nested" in e and ("window" in e or "aggregate" in e):
            return (
                "DIAGNOSIS: Window functions or aggregate functions cannot be directly nested. "
                "FIX: Split into two query levels — compute the inner aggregate in a subquery "
                "or CTE, then apply the outer window/aggregate to that result."
            )

        if ("window function" in e or "aggregate function" in e) and (
            "where clause" in e or "not allowed in" in e or "not permitted" in e
        ):
            return (
                "DIAGNOSIS: Window functions and aggregate functions are not allowed in a WHERE clause. "
                "FIX: For aggregates, move the filter to HAVING. "
                "For window functions, wrap the query in a subquery or CTE and filter in the outer query."
            )

        # ── Type / cast errors ────────────────────────────────────────────────

        # interval / interval — must come before the generic operator check
        # because "operator does not exist: interval / interval" matches both
        if "interval" in e and (
            "/ interval" in e or "interval /" in e
            or ("operator does not exist" in e and "interval" in e)
        ):
            return (
                "DIAGNOSIS: PostgreSQL cannot divide an interval by another interval, "
                "and cannot cast an interval to numeric or integer. "
                "timestamp - timestamp produces an INTERVAL — to convert to days you must use: "
                "EXTRACT(EPOCH FROM (end_ts - start_ts)) / 86400 "
                "(EXTRACT returns seconds; dividing by 86400 gives fractional days). "
                "NEVER write (ts - ts)::numeric, (ts - ts)::integer, "
                "or (ts - ts)::interval / INTERVAL '1 day' — all will fail. "
                "Correct pattern: EXTRACT(EPOCH FROM NULLIF(col_a, '')::timestamp "
                "- NULLIF(col_b, '')::timestamp) / 86400"
            )

        # interval cannot be cast to numeric/integer (separate from above in case
        # the word "interval" doesn't appear in the operator-does-not-exist message)
        if ("cannot cast type interval" in e or "cannot coerce" in e and "interval" in e
                or ("coerce" in e and "interval" in e)):
            return (
                "DIAGNOSIS: An interval (the result of subtracting two timestamps) "
                "cannot be cast directly to numeric or integer. "
                "FIX: Replace `(ts2 - ts1)::numeric / 86400` with "
                "`EXTRACT(EPOCH FROM (ts2 - ts1)) / 86400`. "
                "EXTRACT returns the duration as seconds; dividing by 86400 gives days."
            )

        # Specific explicit-cast failure before generic syntax fallback
        if ("cannot cast type" in e or "cannot be cast" in e or
                ("cast" in e and "does not exist" in e)):
            return (
                "DIAGNOSIS: An explicit type cast is impossible between these two types. "
                "FIX: Use an intermediate cast (e.g. cast to TEXT first, then to the target type), "
                "or restructure to avoid the cast (e.g. use EXTRACT instead of casting a timestamp "
                "to integer, or DATE_PART instead of ::int on an interval)."
            )

        if "could not determine data type" in e or "indeterminate type" in e:
            return (
                "DIAGNOSIS: The database cannot infer the data type of an expression — "
                "typically a bare NULL, an empty string literal, or a CASE branch "
                "where all arms are NULL. "
                "FIX: Add an explicit CAST, e.g. CAST(NULL AS TEXT), CAST(NULL AS NUMERIC), "
                "or CAST('' AS DATE) to tell the planner the intended type."
            )

        if "operator does not exist" in e:
            return (
                "DIAGNOSIS: A comparison or arithmetic operator is being applied to "
                "incompatible types (e.g. integer = text). "
                "FIX: Add an explicit CAST on one side so both operands share the same type, "
                "e.g. CAST(col AS TEXT) = 'value'  or  col = CAST('123' AS INTEGER)."
            )

        if "invalid input syntax" in e or "invalid datetime" in e:
            return (
                "DIAGNOSIS: A string value cannot be parsed into the target type "
                "(e.g. an empty string or non-numeric text being cast to a number or date). "
                "FIX: Guard with NULLIF(col, '') before casting, "
                "use TRY_CAST / TRY_TO_DATE where the dialect supports it, "
                "or filter with WHERE col ~ '^[0-9]+$' before casting."
            )

        if ("date" in e or "timestamp" in e) and ("out of range" in e or "invalid" in e or "format" in e):
            return (
                "DIAGNOSIS: A date or timestamp literal is in the wrong format or outside the "
                "valid range for this database. "
                "FIX: Use ISO-8601 format (YYYY-MM-DD or YYYY-MM-DD HH:MI:SS), "
                "or wrap with TO_DATE(col, 'format') / TO_TIMESTAMP(col, 'format') to parse explicitly."
            )

        if ("overflow" in e or "out of range" in e) and (
            "integer" in e or "numeric" in e or "bigint" in e or "smallint" in e
        ):
            return (
                "DIAGNOSIS: A numeric computation overflows the column's data type. "
                "FIX: CAST intermediate values or the column to BIGINT or NUMERIC(precision, scale) "
                "before arithmetic to avoid overflow."
            )

        # ── Subquery errors ───────────────────────────────────────────────────

        if "more than one row" in e or "subquery used as an expression" in e:
            return (
                "DIAGNOSIS: A scalar subquery (used in SELECT or WHERE with =) "
                "returned more than one row. "
                "FIX: Add LIMIT 1 inside the subquery, use an aggregate (MAX/MIN/AVG), "
                "or replace `= (subquery)` with `IN (subquery)`."
            )

        # ── UNION errors ──────────────────────────────────────────────────────

        if "union" in e and ("number of columns" in e or "types" in e or "cannot be matched" in e):
            return (
                "DIAGNOSIS: The branches of a UNION / UNION ALL are incompatible. "
                "FIX: Ensure every SELECT branch has the same number of columns "
                "and that corresponding columns have compatible types "
                "(add explicit CASTs where needed, e.g. CAST(col AS TEXT))."
            )

        # ── Structural / syntax errors ────────────────────────────────────────

        if "duplicate column" in e or "duplicate alias" in e or "specified more than once" in e:
            return (
                "DIAGNOSIS: Two columns in the result set share the same name or alias. "
                "FIX: Add or change the AS alias on one of the duplicate columns "
                "so every output column has a unique name."
            )

        if "syntax error" in e or "parse error" in e or "unterminated" in e:
            return (
                "DIAGNOSIS: SQL syntax error. "
                "Check for: unbalanced parentheses, missing commas, keywords used as identifiers "
                "(quote them with double-quotes), or function calls written as "
                "`alias.FUNCTION()` instead of `FUNCTION(alias.col)`."
            )

        # ── Resource / performance errors (not always fixable, but LLM can optimise) ──

        if any(p in e for p in (
            "memory exhausted", "out of memory", "memory limit exceeded",
            "disk spill", "work_mem", "not enough memory",
            "exceeded memory", "query exceeded", "resources exceeded",
        )):
            return (
                "DIAGNOSIS: The query consumed too much memory or caused a disk spill. "
                "FIX: Reduce the result set before aggregating — add a WHERE filter, "
                "replace a CROSS JOIN with a filtered JOIN, add a LIMIT clause, "
                "or pre-aggregate in a CTE/subquery to reduce row count before the final join."
            )

        # ── Arithmetic ────────────────────────────────────────────────────────

        if "divide by zero" in e or "division by zero" in e:
            return (
                "DIAGNOSIS: Division by zero. "
                "FIX: Wrap the denominator with NULLIF(denominator, 0) to return NULL instead of erroring."
            )

        return ""  # no specific hint — let the LLM reason from the raw error

    def _build_fix_prompt(
        self,
        question: str,
        failed_sql: str,
        error: str,
        schema: str = "",
    ) -> str:
        hint = self._classify_sql_error(error)
        hint_block = f"\n{hint}\n" if hint else ""
        schema_block = (
            f"\nDATABASE SCHEMA (authoritative — use only tables/columns listed here):\n"
            f"{schema}\n"
            if schema else ""
        )
        return f"""You are a SQL expert. A query failed. Diagnose the root cause and return a corrected query.

ORIGINAL QUESTION: {question}

FAILED SQL:
```sql
{failed_sql}
```

ERROR MESSAGE:
{error}
{hint_block}{schema_block}
RULES:
- Return ONLY a SELECT query. No INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, or DDL.
- Fix the exact error. Do not change the intent or structure of the query beyond what is necessary.
- Return corrected SQL inside a ```sql code block.
- After the SQL block, write ONE sentence explaining what was wrong and what you changed.
- Do NOT invent columns or tables that are not in the schema or original query."""

    def fix_sql(
        self,
        question: str,
        failed_sql: str,
        error: str,
        schema: str = "",
    ) -> "LLMResponse":
        """Ask the LLM to diagnose and fix a failed SQL query.

        Pass ``schema`` (the full schema prompt string) so the LLM can verify
        table and column names when fixing reference errors.

        Subclasses inherit this — they only need to implement _chat().
        """
        prompt = self._build_fix_prompt(question, failed_sql, error, schema=schema)
        raw = self._chat(prompt, temperature=0.0)
        sql = self._extract_sql(raw)
        explanation = raw.split("```")[-1].strip() if "```" in raw else raw.strip()
        return LLMResponse(sql=sql, explanation=explanation, raw=raw)

    def _extract_json_list(self, text: str) -> list[str]:
        """Extract a JSON string array from LLM output."""
        try:
            cleaned = re.sub(r"```json\s*|\s*```", "", text).strip()
            match = re.search(r"\[.*\]", cleaned, re.DOTALL)
            if match:
                result = json.loads(match.group(0))
                if isinstance(result, list):
                    return [str(s) for s in result if s]
        except Exception:
            pass
        return []

    def _extract_sql(self, text: str) -> str:
        """Extract SQL from LLM response."""
        match = re.search(r"```sql\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        match = re.search(r"```\s*(SELECT.*?)```", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        match = re.search(r"((?:WITH|SELECT)\s+.+?)(?:;|\Z)", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return text.strip()


class OllamaBackend(LLMBackend):
    """Ollama local LLM."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        timeout: float = 180.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def ping(self) -> tuple[bool, str]:
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=3.0)
            return True, f"Ollama reachable ({self.model})"
        except httpx.ConnectError:
            return False, f"Ollama not running at {self.base_url} — start it with: ollama serve"
        except Exception as e:
            return False, str(e)

    def _chat(self, prompt: str, temperature: float = 0.1) -> str:
        try:
            resp = self._client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature},
                },
            )
            resp.raise_for_status()
            return resp.json()["response"]
        except httpx.ConnectError:
            raise ConnectionError(
                f"Ollama server not reachable at {self.base_url}.\n"
                f"Start it with:  ollama serve\n"
                f"Then ensure the model is pulled:  ollama pull {self.model}"
            )

    def generate_sql(
        self, schema_prompt: str, question: str,
        kb_context: Optional[str] = None,
        history: Optional[list[dict]] = None,
    ) -> LLMResponse:
        prompt = self._build_sql_prompt(schema_prompt, question, kb_context, history)
        raw = self._chat(prompt)
        sql = self._extract_sql(raw)
        explanation = raw.split("```")[-1].strip() if "```" in raw else ""
        return LLMResponse(sql=sql, explanation=explanation, raw=raw)

    def generate_summary(self, question: str, sql: str, data_preview: str) -> str:
        return self._chat(self._build_summary_prompt(question, sql, data_preview), 0.3)

    def suggest_chart(self, question: str, columns: list[str], row_count: int) -> dict:
        raw = self._chat(self._build_chart_prompt(question, columns, row_count), 0.0)
        try:
            cleaned = re.sub(r"```json\s*|\s*```", "", raw).strip()
            return json.loads(cleaned)
        except (json.JSONDecodeError, KeyError):
            return {"chart_type": "table_only"}

    def suggest_followups(self, question: str, sql: str, data_preview: str) -> list[str]:
        raw = self._chat(self._build_followups_prompt(question, sql, data_preview), 0.4)
        return self._extract_json_list(raw)

    def generate_explore_questions(self, schema_prompt: str, profile: str) -> list[str]:
        raw = self._chat(self._build_explore_prompt(schema_prompt, profile), 0.4)
        return self._extract_json_list(raw)

    # ── Agent / tool-calling ──────────────────────────────────────────

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> ToolResponse:
        """Multi-turn chat with tool-calling via Ollama /api/chat."""
        try:
            resp = self._client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "tools": tools,
                    "stream": False,
                },
            )
            resp.raise_for_status()
            msg = resp.json()["message"]
        except httpx.ConnectError:
            raise ConnectionError(
                f"Ollama server not reachable at {self.base_url}.\n"
                f"Start it with:  ollama serve\n"
                f"Then ensure the model is pulled:  ollama pull {self.model}"
            )

        raw_calls = msg.get("tool_calls") or []
        content = msg.get("content") or ""
        fallback_used = False

        # ── Fallback parsers for models whose tool calls land in content ──
        # Some Ollama model versions don't convert native tool-call formats
        # into the tool_calls field — we parse them out of content instead.
        if not raw_calls and content:
            raw_calls = self._parse_content_tool_calls(content, tools)
            if raw_calls:
                content = ""        # consumed — don't treat as narrative
                fallback_used = True

        normalised = []
        for tc in raw_calls:
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            normalised.append({
                "name": fn.get("name", ""),
                "arguments": args,
                "id": None,  # Ollama does not use tool_call_id
            })

        # When fallback parsing fired, rebuild raw_assistant_msg in the
        # Ollama tool_calls format so the next conversation turn is valid.
        if fallback_used and normalised:
            msg = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": n["name"], "arguments": n["arguments"]}}
                    for n in normalised
                ],
            }

        return ToolResponse(
            content=content,
            tool_calls=normalised,
            raw_assistant_msg=msg,
        )

    @staticmethod
    def _parse_content_tool_calls(content: str, tools: list[dict]) -> list[dict]:
        """Extract tool calls from content when Ollama doesn't populate tool_calls.

        Handles two formats that models emit when native tool-call conversion fails:

        1. Hermes3 / ChatML XML format:
               <tool_call>{"name": "fn", "arguments": {...}}</tool_call>

        2. JSON object where keys are (partial) tool names and values are arguments
           (seen when models improvise a response plan rather than using tool_calls):
               {"create_plan": {"steps": [...]}, "list_tables": {}}
        """
        calls = []
        known_names = {t["function"]["name"] for t in tools}

        # ── Format 1: <tool_call> XML blocks (hermes3 native) ────────────
        xml_matches = re.findall(
            r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, re.DOTALL
        )
        for raw in xml_matches:
            try:
                parsed = json.loads(raw)
                name = parsed.get("name", "")
                if name in known_names:
                    calls.append({
                        "function": {
                            "name": name,
                            "arguments": parsed.get("arguments", {}),
                        }
                    })
            except Exception:
                pass
        if calls:
            return calls

        # ── Format 2: JSON object with tool-name keys ─────────────────────
        # e.g. {"plan": {"steps": [...]}, "tables": {}} where keys approximate
        # tool names. Match by exact name first, then by substring.
        try:
            cleaned = re.sub(r"```json\s*|```", "", content).strip()
            obj = json.loads(cleaned)
            if isinstance(obj, dict):
                # Build a lookup: full_tool_name → any key that is a substring match
                for key, val in obj.items():
                    key_lower = key.lower().replace("_", "").replace(" ", "")
                    for tool_name in known_names:
                        tool_lower = tool_name.lower().replace("_", "")
                        if key_lower == tool_lower or key_lower in tool_lower or tool_lower in key_lower:
                            args = val if isinstance(val, dict) else {}
                            calls.append({
                                "function": {"name": tool_name, "arguments": args}
                            })
                            break
        except Exception:
            pass

        # Deduplicate: keep first occurrence of each tool name
        seen: set[str] = set()
        deduped = []
        for c in calls:
            n = c["function"]["name"]
            if n not in seen:
                seen.add(n)
                deduped.append(c)
        return deduped

    def supports_tool_calling(self) -> tuple[bool, str]:
        """Probe the model with a minimal tool schema to verify support."""
        _probe_tool = [{
            "type": "function",
            "function": {
                "name": "probe",
                "description": "Connectivity probe — ignore.",
                "parameters": {"type": "object", "properties": {}},
            },
        }]
        try:
            resp = self._client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": _probe_tool,
                    "stream": False,
                },
                timeout=httpx.Timeout(10.0),
            )
            resp.raise_for_status()
            return True, f"{self.model} supports tool calling"
        except Exception as exc:
            msg = str(exc).lower()
            if any(k in msg for k in ("does not support", "tool", "function")):
                return (
                    False,
                    f"{self.model} does not support tool calling. "
                    "Recommended models: hermes3, llama3.1, qwen2.5",
                )
            return False, f"Could not verify tool calling support: {exc}"


class OpenAICompatibleBackend(LLMBackend):
    """Any OpenAI-compatible API (LM Studio, vLLM, text-gen-webui, LocalAI, etc.)."""

    # Subclasses may override to customise identity shown in errors.
    _backend_name: str = "OpenAI-compatible"

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model: str = "local-model",
        api_key: str = "not-needed",
        timeout: float = 180.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self._client = httpx.Client(timeout=timeout)

    def ping(self) -> tuple[bool, str]:
        try:
            r = httpx.get(f"{self.base_url}/models", timeout=3.0,
                          headers={"Authorization": f"Bearer {self.api_key}"})
            return True, f"{self._backend_name} reachable ({self.model})"
        except httpx.ConnectError:
            return False, f"{self._backend_name} server not running at {self.base_url}"
        except Exception as e:
            return False, str(e)

    def _chat(self, prompt: str, temperature: float = 0.1) -> str:
        try:
            resp = self._client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except httpx.ConnectError:
            raise ConnectionError(
                f"{self._backend_name} server not reachable at {self.base_url}.\n"
                f"Make sure the server is running and the URL is correct."
            )

    def generate_sql(
        self, schema_prompt: str, question: str,
        kb_context: Optional[str] = None,
        history: Optional[list[dict]] = None,
    ) -> LLMResponse:
        prompt = self._build_sql_prompt(schema_prompt, question, kb_context, history)
        raw = self._chat(prompt)
        sql = self._extract_sql(raw)
        explanation = raw.split("```")[-1].strip() if "```" in raw else ""
        return LLMResponse(sql=sql, explanation=explanation, raw=raw)

    def generate_summary(self, question: str, sql: str, data_preview: str) -> str:
        return self._chat(self._build_summary_prompt(question, sql, data_preview), 0.3)

    def suggest_chart(self, question: str, columns: list[str], row_count: int) -> dict:
        raw = self._chat(self._build_chart_prompt(question, columns, row_count), 0.0)
        try:
            cleaned = re.sub(r"```json\s*|\s*```", "", raw).strip()
            return json.loads(cleaned)
        except (json.JSONDecodeError, KeyError):
            return {"chart_type": "table_only"}

    def suggest_followups(self, question: str, sql: str, data_preview: str) -> list[str]:
        raw = self._chat(self._build_followups_prompt(question, sql, data_preview), 0.4)
        return self._extract_json_list(raw)

    def generate_explore_questions(self, schema_prompt: str, profile: str) -> list[str]:
        raw = self._chat(self._build_explore_prompt(schema_prompt, profile), 0.4)
        return self._extract_json_list(raw)

    # ── Agent / tool-calling ──────────────────────────────────────────

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> ToolResponse:
        """Multi-turn chat with tool-calling via OpenAI-compatible /chat/completions."""
        try:
            resp = self._client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "auto",
                },
            )
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
        except httpx.ConnectError:
            raise ConnectionError(
                f"{self._backend_name} server not reachable at {self.base_url}.\n"
                "Make sure the server is running and the URL is correct."
            )

        raw_calls = msg.get("tool_calls") or []
        normalised = []
        for tc in raw_calls:
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            normalised.append({
                "name": fn.get("name", ""),
                "arguments": args,
                "id": tc.get("id"),  # OpenAI always provides tool_call_id
            })

        return ToolResponse(
            content=msg.get("content") or "",
            tool_calls=normalised,
            raw_assistant_msg=msg,
        )

    def build_tool_result_msg(self, tool_call: dict, result: str) -> dict:
        """OpenAI-compatible backends require tool_call_id on tool-result messages."""
        msg: dict = {"role": "tool", "content": str(result)}
        if tool_call.get("id"):
            msg["tool_call_id"] = tool_call["id"]
        return msg


class MLXBackend(OpenAICompatibleBackend):
    """Apple Silicon MLX-LM server backend.

    MLX-LM runs models natively on Apple Silicon via Metal and exposes an
    OpenAI-compatible HTTP server, so this backend is a thin wrapper around
    OpenAICompatibleBackend with MLX-appropriate defaults.

    The server is started automatically on demand — no manual terminal command
    needed. If the server is not running when a query is made, this backend
    will spawn it, wait for it to load the model into memory, and then proceed.

    Any model from the mlx-community HuggingFace organisation works, e.g.:
        mlx-community/Qwen3-8B-4bit                      (default, ~5 GB)
        mlx-community/Qwen3-8B-8bit                      (~9 GB, higher quality)
        mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit  (~18 GB, MoE code-specialist)

    Note: Qwen3 is a thinking model. This backend appends /no_think to every
    prompt to disable chain-of-thought reasoning — you get direct SQL answers
    instead of long reasoning traces, which is faster and more reliable for
    text-to-SQL tasks.
    """

    _backend_name: str = "MLX"
    # One start attempt at a time across all instances (class-level lock)
    _start_lock: threading.Lock = threading.Lock()

    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1",
        model: str = "mlx-community/Qwen3-8B-4bit",
        api_key: str = "not-needed",
        timeout: float = 180.0,
    ):
        super().__init__(base_url=base_url, model=model, api_key=api_key, timeout=timeout)
        self._proc: Optional[subprocess.Popen] = None

    # ── Server lifecycle ──────────────────────────────────────────────

    def _port(self) -> int:
        try:
            return urlparse(self.base_url).port or 8080
        except Exception:
            return 8080

    def _is_up(self, timeout: float = 2.0) -> bool:
        try:
            httpx.get(f"{self.base_url}/models", timeout=timeout)
            return True
        except Exception:
            return False

    @staticmethod
    def _mlx_lm_available() -> bool:
        """Return True if mlx_lm is importable in the current Python environment."""
        import importlib.util
        return importlib.util.find_spec("mlx_lm") is not None

    _MLX_INSTALL_HINT = (
        "mlx-lm is not installed in this Python environment.\n\n"
        "  If you installed via pipx:\n"
        "      pipx inject talonsight mlx-lm\n\n"
        "  If you installed via pip:\n"
        "      pip install mlx-lm\n\n"
        "  Or reinstall with the mlx extra:\n"
        "      pip install talonsight[mlx]"
    )

    def _start_and_wait(self, wait_secs: int = 180) -> bool:
        """Ensure the MLX server is running, starting it if needed.

        Blocks until the server responds or *wait_secs* elapses.
        Returns True when the server is reachable.
        Uses a class-level lock so only one spawn attempt runs at a time.
        Raises RuntimeError immediately if mlx_lm is not installed — no waiting.
        """
        with self.__class__._start_lock:
            # Fast path — already up (maybe started by cli.py or another thread)
            if self._is_up():
                return True

            # Fail fast — don't wait 180 s just to discover the module is missing
            if not self._mlx_lm_available():
                raise RuntimeError(self._MLX_INSTALL_HINT)

            port = self._port()
            logger.info("MLX server not running — spawning on port %d …", port)

            try:
                self._proc = subprocess.Popen(
                    [sys.executable, "-m", "mlx_lm", "server",
                     "--model", self.model, "--port", str(port)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,   # capture so early crashes are logged
                )
            except Exception as exc:
                logger.warning("Could not spawn MLX server: %s", exc)
                return False

            deadline = time.time() + wait_secs
            while time.time() < deadline:
                # Process died unexpectedly — log stderr so the user knows why
                if self._proc.poll() is not None:
                    try:
                        err_out = (self._proc.stderr.read() or b"").decode(errors="replace").strip()
                    except Exception:
                        err_out = ""
                    logger.warning(
                        "MLX server process exited early (rc=%s)%s",
                        self._proc.returncode,
                        f": {err_out}" if err_out else "",
                    )
                    return False
                if self._is_up(timeout=1.5):
                    logger.info("MLX server ready at %s", self.base_url)
                    return True
                time.sleep(1)

            logger.warning("MLX server did not become ready within %ds", wait_secs)
            return False

    # ── LLM protocol ─────────────────────────────────────────────────

    def ping(self) -> tuple[bool, str]:
        if self._is_up():
            return True, f"MLX server reachable ({self.model})"
        try:
            started = self._start_and_wait()
        except RuntimeError as exc:
            return False, str(exc)
        if started:
            return True, f"MLX server started automatically ({self.model})"
        return (
            False,
            f"MLX server could not be started at {self.base_url}. "
            f"Check logs for the exact error, or start it manually:\n"
            f"  python3 -m mlx_lm.server --model {self.model} --port {self._port()}",
        )

    def _chat(self, prompt: str, temperature: float = 0.1) -> str:
        # Append /no_think to disable Qwen3's chain-of-thought reasoning mode.
        # Without this, the model outputs a long <think>...</think> block before
        # the actual answer, which breaks SQL extraction and wastes time.
        no_think_prompt = prompt + " /no_think"

        def _do_request() -> str:
            resp = self._client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": no_think_prompt}],
                    "temperature": temperature,
                },
            )
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            # Qwen3 thinking models may return content in "reasoning_content" or
            # "reasoning" when in thinking mode. Prefer "content", fall back gracefully.
            return (
                msg.get("content")
                or msg.get("reasoning_content")
                or msg.get("reasoning")
                or ""
            )

        try:
            return _do_request()
        except httpx.ConnectError:
            # Server not reachable — attempt automatic start, then retry once.
            logger.info("MLX server unreachable — attempting auto-start…")
            try:
                started = self._start_and_wait()
            except RuntimeError as exc:
                # mlx_lm not installed — surface the install hint immediately
                raise ConnectionError(str(exc)) from None
            if started:
                try:
                    return _do_request()
                except httpx.ConnectError:
                    pass
            raise ConnectionError(
                f"MLX server at {self.base_url} is not running and could not be "
                f"started automatically.\n"
                f"Start it manually: python3 -m mlx_lm.server --model {self.model} --port {self._port()}"
            )
