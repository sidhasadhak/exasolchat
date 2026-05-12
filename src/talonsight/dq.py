"""Data Quality Scanner — rule-driven anomaly detection for any connected table.

Loads a JSON rule config (built-in ``dq_config.json`` or user-supplied), classifies
every column by naming pattern and SQL type, generates a check SQL per applicable
rule × column pair, executes them all, and returns results ranked by severity.

Supported checks
----------------
null_check               All columns    — NULL rate per column
blank_string_check       String cols    — empty / whitespace-only values
whitespace_anomaly       String cols    — leading/trailing whitespace
invalid_date_check       Date-like cols — values that cannot be parsed as a date
numeric_format_validation Numeric-named  — string columns storing non-numeric text
rare_value_detection     Categorical    — values appearing in < 1% of rows
duplicate_rows           Entire table   — fully duplicated rows
logical_consistency      Inferred pairs — LLM-detected constraint violations
                                          (e.g. start_date > end_date)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from talonsight.schema import TableInfo

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

BUILTIN_CONFIG = Path(__file__).parent / "dq_config.json"

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
SEVERITY_COLOR = {
    "critical": "🔴",
    "high":     "🟠",
    "medium":   "🟡",
    "low":      "🔵",
}

# SQL types that are textual (string checks apply)
_STRING_TYPES = {"char", "text", "varchar", "string", "clob", "nchar", "nvarchar", "bpchar"}
# SQL types that are numeric
_NUMERIC_TYPES = {"int", "float", "double", "decimal", "numeric", "number", "real",
                  "bigint", "smallint", "tinyint", "byteint", "money"}
# SQL types that are date/time
_DATE_TYPES = {"date", "time", "timestamp", "datetime", "interval"}


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class DQResult:
    rule_name:           str
    column_name:         str
    failed_count:        int
    total_count:         int
    failure_rate:        float          # 0.0 – 1.0
    severity:            str
    generated_sql:       str
    sample_failed_values: list[str] = field(default_factory=list)
    error:               Optional[str] = None

    @property
    def severity_rank(self) -> int:
        return SEVERITY_ORDER.get(self.severity.lower(), 99)

    @property
    def failure_pct(self) -> str:
        return f"{self.failure_rate * 100:.1f}%"

    @property
    def severity_icon(self) -> str:
        return SEVERITY_COLOR.get(self.severity.lower(), "⚪")


# ── Scanner ───────────────────────────────────────────────────────────────────

class DataQualityScanner:
    """Generates and executes data-quality checks against a single table."""

    def __init__(
        self,
        config: Optional[dict] = None,
        config_path: Optional[str] = None,
    ) -> None:
        if config:
            self._cfg = config
        elif config_path:
            self._cfg = json.loads(Path(config_path).read_text())
        else:
            self._cfg = json.loads(BUILTIN_CONFIG.read_text())

        self._type_rules: dict[str, list[str]] = (
            self._cfg
            .get("column_profile_rules", {})
            .get("type_inference_rules", {})
        )
        self._assessment: dict = self._cfg.get("assessment_config", {})
        self._checks: list[dict] = self._cfg.get("checks", [])

    # ── Column classification ─────────────────────────────────────────

    def _classify(self, col_name: str, col_type: str) -> set[str]:
        """Return a set of semantic labels for a column."""
        labels: set[str] = set()
        name = col_name.lower()
        base_type = re.split(r"[\s(]", col_type.lower())[0]  # strip "(n)" suffixes

        # Naming-pattern labels from config
        for label, patterns in self._type_rules.items():
            for pat in patterns:
                if fnmatch(name, pat.lower()):
                    labels.add(label)
                    break

        # SQL-type labels — use 'in' so "integer" matches "int", "varchar" matches "char"
        if any(k in base_type for k in _STRING_TYPES):
            labels.add("string_columns")
        if any(k in base_type for k in _NUMERIC_TYPES):
            labels.add("native_numeric")
        if any(k in base_type for k in _DATE_TYPES):
            labels.add("native_date")

        return labels

    # ── SQL builders ──────────────────────────────────────────────────

    @staticmethod
    def _total_cte(table: str) -> str:
        return f"(SELECT COUNT(*) FROM {table})"

    @staticmethod
    def _count_where(table: str, col: str, where: str, total: str) -> str:
        return (
            f"SELECT COUNT(*) AS failed_count, "
            f"COUNT(*) * 1.0 / NULLIF({total}, 0) AS failure_rate "
            f"FROM {table} WHERE {where}"
        )

    @staticmethod
    def _sample_where(table: str, col: str, where: str) -> str:
        return f"SELECT {col} FROM {table} WHERE {where} LIMIT 5"

    def _null_sql(self, table: str, col: str) -> tuple[str, str]:
        w = f"{col} IS NULL"
        total = self._total_cte(table)
        return self._count_where(table, col, w, total), self._sample_where(table, col, w)

    def _blank_sql(self, table: str, col: str) -> tuple[str, str]:
        w = f"TRIM(CAST({col} AS VARCHAR)) = ''"
        total = self._total_cte(table)
        return self._count_where(table, col, w, total), self._sample_where(table, col, w)

    def _whitespace_sql(self, table: str, col: str) -> tuple[str, str]:
        w = f"CAST({col} AS VARCHAR) <> TRIM(CAST({col} AS VARCHAR))"
        total = self._total_cte(table)
        return self._count_where(table, col, w, total), self._sample_where(table, col, w)

    def _invalid_date_sql(self, table: str, col: str, dialect: str) -> tuple[str, str]:
        d = dialect.lower()
        if "postgres" in d or "postgresql" in d:
            # PG has no TRY_CAST; use regex to check ISO-8601 format for string cols
            w = (
                f"{col} IS NOT NULL "
                f"AND TRIM(CAST({col} AS TEXT)) <> '' "
                f"AND CAST({col} AS TEXT) !~ "
                r"'^[0-9]{4}-[0-9]{2}-[0-9]{2}'"
            )
        elif "duckdb" in d:
            w = f"TRY_CAST({col} AS DATE) IS NULL AND {col} IS NOT NULL"
        else:
            w = f"TRY_CAST({col} AS DATE) IS NULL AND {col} IS NOT NULL"
        total = self._total_cte(table)
        return self._count_where(table, col, w, total), self._sample_where(table, col, w)

    def _numeric_fmt_sql(self, table: str, col: str, dialect: str) -> tuple[str, str]:
        d = dialect.lower()
        if "postgres" in d or "postgresql" in d:
            w = (
                f"{col} IS NOT NULL "
                f"AND TRIM(CAST({col} AS TEXT)) <> '' "
                r"AND CAST({col} AS TEXT) !~ "
                r"'^-?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?$'"
            )
            w = w.replace("{col}", col)
        elif "duckdb" in d:
            w = f"TRY_CAST({col} AS DOUBLE) IS NULL AND {col} IS NOT NULL"
        else:
            w = f"TRY_CAST({col} AS FLOAT) IS NULL AND {col} IS NOT NULL"
        total = self._total_cte(table)
        return self._count_where(table, col, w, total), self._sample_where(table, col, w)

    def _rare_value_sql(self, table: str, col: str, threshold: float) -> tuple[str, str]:
        count_sql = (
            f"SELECT COUNT(*) AS failed_count, "
            f"COUNT(*) * 1.0 / NULLIF({self._total_cte(table)}, 0) AS failure_rate "
            f"FROM {table} t "
            f"INNER JOIN ("
            f"  SELECT {col} FROM {table} "
            f"  GROUP BY {col} "
            f"  HAVING COUNT(*) * 1.0 / NULLIF({self._total_cte(table)}, 0) < {threshold} "
            f"  AND {col} IS NOT NULL"
            f") rare ON t.{col} = rare.{col}"
        )
        sample_sql = (
            f"SELECT {col}, COUNT(*) AS cnt FROM {table} "
            f"GROUP BY {col} "
            f"HAVING COUNT(*) * 1.0 / NULLIF({self._total_cte(table)}, 0) < {threshold} "
            f"AND {col} IS NOT NULL "
            f"ORDER BY cnt ASC LIMIT 5"
        )
        return count_sql, sample_sql

    def _duplicate_rows_sql(self, table: str, cols: list[str]) -> tuple[str, str]:
        cols_str = ", ".join(cols)
        count_sql = (
            f"SELECT COUNT(*) AS failed_count, "
            f"COUNT(*) * 1.0 / NULLIF({self._total_cte(table)}, 0) AS failure_rate "
            f"FROM ("
            f"  SELECT {cols_str}, COUNT(*) AS cnt "
            f"  FROM {table} GROUP BY {cols_str} HAVING COUNT(*) > 1"
            f") dups"
        )
        sample_sql = (
            f"SELECT {cols_str} FROM {table} "
            f"GROUP BY {cols_str} HAVING COUNT(*) > 1 LIMIT 3"
        )
        return count_sql, sample_sql

    def _logical_consistency_sql(
        self, table: str, table_info: "TableInfo", dialect: str, llm
    ) -> list[tuple[str, str, str, str]]:
        """Ask LLM to infer date-pair / value constraints and generate check SQLs."""
        if llm is None:
            return []
        col_names = [c.name for c in table_info.columns]
        col_list  = ", ".join(col_names)
        prompt = (
            f"Table: {table}\n"
            f"Columns: {col_list}\n\n"
            f"Identify pairs of columns where a logical constraint should hold "
            f"(e.g. start_date <= end_date, order_date <= ship_date, amount >= 0). "
            f"Only include pairs where both columns exist in the list above. "
            f"For each constraint, return a JSON array item with:\n"
            f'  {{"constraint": "col_a <= col_b", "col_a": "...", "col_b": "..."}}\n\n'
            f"Return ONLY a JSON array. No explanation. "
            f"If no obvious constraints exist, return []."
        )
        try:
            raw = llm._chat(prompt, temperature=0.0)
            # Extract JSON array
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            if not m:
                return []
            constraints = json.loads(m.group(0))
        except Exception as exc:
            logger.warning("Logical consistency LLM call failed: %s", exc)
            return []

        results = []
        total = self._total_cte(table)
        for c in constraints:
            if not isinstance(c, dict):
                continue
            constraint = c.get("constraint", "")
            col_a = c.get("col_a", "")
            col_b = c.get("col_b", "")
            if not constraint or not col_a or not col_b:
                continue
            # Build violation SQL: rows where constraint is BROKEN
            w = f"NOT ({col_a} {constraint.replace(col_a, '').replace(col_b, '').strip()} {col_b})"
            # Simpler: directly negate the constraint
            w = f"({col_a} IS NOT NULL AND {col_b} IS NOT NULL AND NOT ({constraint}))"
            count_sql  = self._count_where(table, f"{col_a},{col_b}", w, total)
            sample_sql = f"SELECT {col_a}, {col_b} FROM {table} WHERE {w} LIMIT 5"
            results.append((f"logical_consistency:{constraint}", f"{col_a},{col_b}", count_sql, sample_sql))
        return results

    # ── Check assembly ────────────────────────────────────────────────

    def _build_checks(
        self,
        table_name: str,
        table_info: "TableInfo",
        dialect: str,
        llm,
    ) -> list[tuple[str, str, str, str, str]]:
        """Return list of (rule_name, col_name, severity, count_sql, sample_sql)."""
        items: list[tuple[str, str, str, str, str]] = []
        null_map = self._cfg.get("assessment_config", {})
        rare_threshold = null_map.get("rare_value_threshold", 0.01)

        col_names = [c.name for c in table_info.columns]

        for col in table_info.columns:
            labels = self._classify(col.name, col.type or "")
            is_str = "string_columns" in labels
            is_native_num = "native_numeric" in labels
            is_native_date = "native_date" in labels

            # ── null_check — all columns ──────────────────────────────
            csql, ssql = self._null_sql(table_name, col.name)
            sev = "high" if "id_columns" in labels else "medium"
            items.append(("null_check", col.name, sev, csql, ssql))

            if is_str:
                # ── blank_string_check ────────────────────────────────
                csql, ssql = self._blank_sql(table_name, col.name)
                items.append(("blank_string_check", col.name, "medium", csql, ssql))

                # ── whitespace_anomaly ────────────────────────────────
                csql, ssql = self._whitespace_sql(table_name, col.name)
                items.append(("whitespace_anomaly", col.name, "low", csql, ssql))

                # ── invalid_date_check (string col with date-like name) ─
                if "date_like_columns" in labels:
                    csql, ssql = self._invalid_date_sql(table_name, col.name, dialect)
                    items.append(("invalid_date_check", col.name, "high", csql, ssql))

                # ── numeric_format_validation (string col with numeric name) ─
                if "numeric_columns" in labels:
                    csql, ssql = self._numeric_fmt_sql(table_name, col.name, dialect)
                    items.append(("numeric_format_validation", col.name, "high", csql, ssql))

            # ── rare_value_detection — categorical cols ───────────────
            if "categorical_columns" in labels:
                csql, ssql = self._rare_value_sql(table_name, col.name, rare_threshold)
                items.append(("rare_value_detection", col.name, "medium", csql, ssql))

        # ── duplicate_rows — entire table ─────────────────────────────
        if self._assessment.get("duplicate_row_check", True):
            csql, ssql = self._duplicate_rows_sql(table_name, col_names)
            items.append(("duplicate_rows", "*", "high", csql, ssql))

        # ── logical_consistency — LLM-inferred pairs ───────────────────
        if self._assessment.get("logical_consistency_check", True) and llm is not None:
            for rule, cols, csql, ssql in self._logical_consistency_sql(
                table_name, table_info, dialect, llm
            ):
                items.append((rule, cols, "critical", csql, ssql))

        return items

    # ── Main entry point ──────────────────────────────────────────────

    def run(
        self,
        table_name: str,
        table_info: "TableInfo",
        db,
        dialect: str,
        llm=None,
        on_progress: Optional[Callable[[int, int, str, str], None]] = None,
    ) -> list[DQResult]:
        """Execute all applicable checks and return results sorted by severity.

        Parameters
        ----------
        table_name:   Qualified table name (schema.table or just table).
        table_info:   TableInfo from schema introspection.
        db:           DatabaseConnection instance (has execute_query()).
        dialect:      SQL dialect string (e.g. "PostgreSQL", "DuckDB").
        llm:          Optional LLM backend — used only for logical_consistency.
        on_progress:  Optional callback(current, total, rule, col) for UI updates.
        """
        checks = self._build_checks(table_name, table_info, dialect, llm)
        results: list[DQResult] = []

        # Total row count — needed for failure_rate denominator
        total_rows = 0
        try:
            df = db.execute_query(f"SELECT COUNT(*) AS n FROM {table_name}", 1)
            total_rows = int(df.iloc[0, 0])
        except Exception:
            pass

        for i, (rule_name, col_name, severity, count_sql, sample_sql) in enumerate(checks):
            if on_progress:
                try:
                    on_progress(i + 1, len(checks), rule_name, col_name)
                except Exception:
                    pass

            failed_count  = 0
            failure_rate  = 0.0
            sample_values: list[str] = []
            error: Optional[str]     = None

            try:
                df = db.execute_query(count_sql, 10)
                failed_count = int(df["failed_count"].iloc[0] or 0)
                raw_rate     = df["failure_rate"].iloc[0]
                failure_rate = float(raw_rate) if raw_rate is not None else 0.0
                if failure_rate != failure_rate:   # NaN guard
                    failure_rate = 0.0
            except Exception as exc:
                error = str(exc)

            # Skip clean checks (no failures, no error worth surfacing)
            if failed_count == 0 and error is None:
                continue

            # Sample values — best-effort
            if failed_count > 0 and sample_sql:
                try:
                    sdf = db.execute_query(sample_sql, 5)
                    sample_values = [
                        str(v) for v in sdf.iloc[:, 0].tolist() if v is not None
                    ][:5]
                except Exception:
                    pass

            results.append(DQResult(
                rule_name=rule_name,
                column_name=col_name,
                failed_count=failed_count,
                total_count=total_rows,
                failure_rate=failure_rate,
                severity=severity,
                generated_sql=count_sql,
                sample_failed_values=sample_values,
                error=error,
            ))

        # Sort: severity rank ascending, then failure_rate descending
        results.sort(key=lambda r: (r.severity_rank, -r.failure_rate))
        return results
