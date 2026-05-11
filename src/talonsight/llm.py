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


class LLMBackend(ABC):
    """Abstract LLM backend."""

    def ping(self) -> tuple[bool, str]:
        """Return (reachable, message). Override in each backend."""
        return False, "ping not implemented"

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

    def _build_fix_prompt(self, question: str, failed_sql: str, error: str) -> str:
        return f"""You are a SQL expert. A SQL query failed with an error. Diagnose the problem and return a corrected query.

ORIGINAL QUESTION: {question}

FAILED SQL:
```sql
{failed_sql}
```

ERROR MESSAGE:
{error}

RULES:
- Return ONLY a SELECT query. No INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, or DDL.
- Fix the exact error described. Do not change the intent of the query.
- Return corrected SQL inside a ```sql code block.
- After the SQL, write 1 sentence explaining what was wrong and what you fixed.
- If the error is about a missing column, check if you used the wrong column name or alias.
- If the error is about a type mismatch, add the appropriate CAST.
- If the error is about an ambiguous column, qualify it with the table name or alias.
- Do NOT invent columns or tables that are not in the original query."""

    def fix_sql(self, question: str, failed_sql: str, error: str) -> "LLMResponse":
        """Ask the LLM to diagnose and fix a failed SQL query.

        Subclasses inherit this — they only need to implement _chat().
        """
        prompt = self._build_fix_prompt(question, failed_sql, error)
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
