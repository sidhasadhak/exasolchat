"""LLM backends for text-to-SQL generation.

Supports Ollama (default) and any OpenAI-compatible API (LM Studio, vLLM, etc.).
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import httpx


@dataclass
class LLMResponse:
    sql: str
    explanation: str
    raw: str


class LLMBackend(ABC):
    """Abstract LLM backend."""

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
- Identifiers are case-insensitive but preserve their stored case.
  ALWAYS use the exact column names from the schema — if schema shows "Order Date", write "Order Date";
  if it shows order_date, write order_date. Never guess or transform column names.
- GROUP BY ALL — groups by all non-aggregated SELECT columns automatically.
- ORDER BY ALL — sorts by all columns.
- SELECT * EXCLUDE(col1, col2) — wildcard minus named columns.
- SELECT * REPLACE(expr AS col) — wildcard with column overrides.
- QUALIFY — filter window function results without a subquery:
    SELECT * FROM t QUALIFY ROW_NUMBER() OVER (PARTITION BY x ORDER BY y) = 1
- PIVOT / UNPIVOT — transpose rows to columns and back.
- UNION BY NAME — match union sides by column name, not position.
- string_split(col, delim), regexp_matches(col, pattern), string_agg(col, sep)
- list_agg(), array_agg(), unnest(col) for list/array columns.
- Nested types: STRUCT (dot access), LIST, MAP.
- Trailing commas in SELECT/FROM lists are valid syntax.
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

        dialect_section = self._DUCKDB_DIALECT_HINTS if "duckdb" in schema_prompt.lower() else (
            "- For Exasol: use LIMIT, double-quote identifiers only if mixed case."
        )

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

Write a concise natural-language summary. Be specific with numbers. 2-3 sentences max."""

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

    def _extract_sql(self, text: str) -> str:
        """Extract SQL from LLM response."""
        # Try <sql> tags first (fine-tuned model format)
        match = re.search(r"<sql>\s*(.*?)\s*</sql>", text, re.DOTALL)
        if match:
            return match.group(1).strip().rstrip(";")

        # Fall back to ```sql blocks (generic model format)
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

    def _chat(self, prompt: str, temperature: float = 0.1) -> str:
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


class OpenAICompatibleBackend(LLMBackend):
    """Any OpenAI-compatible API (LM Studio, vLLM, text-gen-webui, LocalAI, etc.)."""

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

    def _chat(self, prompt: str, temperature: float = 0.1) -> str:
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
