"""LLM backends for text-to-SQL generation.

Supports Ollama (default) and any OpenAI-compatible API (LM Studio, vLLM, etc.).
Prompts are RAG-aware: they include relevant past Q&A pairs when available.
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
        rag_examples: Optional[str] = None,
    ) -> LLMResponse:
        ...

    @abstractmethod
    def generate_summary(self, question: str, sql: str, data_preview: str) -> str:
        ...

    @abstractmethod
    def suggest_chart(self, question: str, columns: list[str], row_count: int) -> dict:
        ...

    def _build_sql_prompt(
        self, schema_prompt: str, question: str,
        rag_examples: Optional[str] = None,
    ) -> str:
        rag_section = ""
        if rag_examples:
            rag_section = f"""
SIMILAR PAST QUERIES (use these as reference for style and patterns):
{rag_examples}
"""

        return f"""You are a SQL expert. Given the database schema below, write a SQL query that answers the user's question.

RULES:
- Write ONLY a SELECT query. Never write INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, EXEC, CALL, EXPORT, IMPORT, or any DDL/DML.
- Use only tables and columns that exist in the schema below.
- Return the SQL inside a ```sql code block.
- After the SQL, briefly explain what the query does.
- If the question cannot be answered with the schema, say so clearly.
- Use appropriate aggregations, JOINs, filtering, and window functions.
- Alias columns for human readability.
- For Exasol: use LIMIT (not TOP), double-quote identifiers only if mixed case.
- For DuckDB: supports LIMIT, window functions, list/struct types, UNNEST, strftime for dates. Use DuckDB SQL dialect.

{schema_prompt}
{rag_section}
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
        rag_examples: Optional[str] = None,
    ) -> LLMResponse:
        prompt = self._build_sql_prompt(schema_prompt, question, rag_examples)
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
        rag_examples: Optional[str] = None,
    ) -> LLMResponse:
        prompt = self._build_sql_prompt(schema_prompt, question, rag_examples)
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
