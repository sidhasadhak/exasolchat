"""Visual query builder — translates field-well configurations into SQL.

Converts a structured config dict (dimensions, measures, filters, sort, limit)
into a formatted, safety-compliant SELECT statement. Also provides schema
introspection helpers for the field palette.
"""

from __future__ import annotations

import re
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from talonsight.schema import SchemaContext
    from talonsight.metrics import MetricsCatalog


AGGREGATIONS = ["SUM", "AVG", "COUNT", "COUNT DISTINCT", "MIN", "MAX", "MEDIAN"]

FILTER_OPS = ["=", "!=", ">", "<", ">=", "<=", "LIKE", "IN", "IS NULL", "IS NOT NULL"]

_NUMERIC_TYPES = {
    "INTEGER", "INT", "BIGINT", "SMALLINT", "HUGEINT", "UBIGINT",
    "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC", "REAL",
}
_DATE_TYPES = {"DATE", "TIMESTAMP", "DATETIME", "TIMESTAMPTZ"}


def _is_numeric(col_type: str) -> bool:
    t = col_type.upper()
    return any(x in t for x in _NUMERIC_TYPES)


def _is_date(col_type: str) -> bool:
    t = col_type.upper()
    return any(x in t for x in _DATE_TYPES)


def col_type_icon(col_type: str) -> str:
    t = col_type.upper()
    if any(x in t for x in _DATE_TYPES):
        return "📅"
    if any(x in t for x in _NUMERIC_TYPES):
        return "🔢"
    if "BOOL" in t:
        return "☑️"
    return "🔤"


class QueryBuilder:
    """Builds SQL from a field-well configuration dict."""

    def __init__(self, schema: "SchemaContext"):
        self.schema = schema
        self._tables = {t.name: t for t in schema.tables}

    # ── Field introspection ───────────────────────────────────────

    def table_names(self) -> list[str]:
        return list(self._tables.keys())

    def columns_for(self, table_name: str) -> list:
        """Returns list of column objects for a table."""
        t = self._tables.get(table_name)
        return t.columns if t else []

    def numeric_columns(self, table_name: str) -> list[str]:
        return [c.name for c in self.columns_for(table_name) if _is_numeric(c.type)]

    def dimension_columns(self, table_name: str) -> list[str]:
        return [c.name for c in self.columns_for(table_name) if not _is_numeric(c.type)]

    # ── SQL generation ────────────────────────────────────────────

    def build_sql(
        self,
        config: dict,
        metrics_catalog: Optional["MetricsCatalog"] = None,
    ) -> str:
        """Translate a builder config dict into a SQL query string.

        Config keys:
            table        str            table name
            dimensions   list[str]      GROUP BY columns
            measures     list[dict]     {"field", "agg", "alias"}
            metric_names list[str]      metric names from catalog
            filters      list[dict]     {"field", "op", "value"}
            sort_field   str            column/alias to sort by
            sort_dir     str            "ASC" | "DESC"
            limit        int
        """
        table       = config.get("table", "")
        dimensions  = config.get("dimensions", [])
        measures    = config.get("measures", [])
        metric_names = config.get("metric_names", [])
        filters     = config.get("filters", [])
        sort_field  = config.get("sort_field", "")
        sort_dir    = config.get("sort_dir", "DESC")
        limit       = config.get("limit", 500)

        select_parts = []

        for dim in dimensions:
            select_parts.append(f'    "{dim}"')

        for m in measures:
            field = m["field"]
            agg   = m["agg"]
            alias = m.get("alias") or f"{agg}_{field}"
            if agg == "COUNT DISTINCT":
                select_parts.append(f'    COUNT(DISTINCT "{field}") AS "{alias}"')
            else:
                select_parts.append(f'    {agg}("{field}") AS "{alias}"')

        if metrics_catalog:
            for name in metric_names:
                metric = metrics_catalog.get(name)
                if metric and metric.get("sql"):
                    select_parts.append(f'    {metric["sql"]} AS "{name}"')

        if not select_parts:
            select_parts = ["    *"]

        lines = ["SELECT"]
        lines.append(",\n".join(select_parts))
        # Qualify table name with its schema to avoid "table not found" errors
        _tinfo = self._tables.get(table)
        if _tinfo and _tinfo.schema:
            _qualified = f'"{_tinfo.schema}"."{table}"'
        else:
            _qualified = f'"{table}"'
        lines.append(f"FROM\n    {_qualified}")

        # WHERE
        valid_filters = [
            f for f in filters
            if f.get("field") and (
                f.get("op") in ("IS NULL", "IS NOT NULL") or f.get("value", "")
            )
        ]
        if valid_filters:
            where_clauses = []
            for f in valid_filters:
                clause = _build_filter_clause(f)
                if clause:
                    where_clauses.append(f"    {clause}")
            if where_clauses:
                lines.append("WHERE\n" + "\n    AND ".join(where_clauses))

        # GROUP BY
        if dimensions and (measures or metric_names):
            group_cols = ",\n    ".join(f'"{d}"' for d in dimensions)
            lines.append(f"GROUP BY\n    {group_cols}")

        # ORDER BY
        if sort_field:
            lines.append(f'ORDER BY\n    "{sort_field}" {sort_dir}')

        # LIMIT
        if limit:
            lines.append(f"LIMIT {limit}")

        return "\n".join(lines)

    # ── NL → Builder seeding ──────────────────────────────────────

    def seed_from_sql(self, sql: str) -> dict:
        """Parse a SQL string and return a partial builder config dict.

        Used for the "Open in Builder" NL → visual handoff.
        Best-effort: handles common SELECT / GROUP BY / WHERE patterns.
        """
        config: dict = {
            "table": "",
            "dimensions": [],
            "measures": [],
            "metric_names": [],
            "filters": [],
            "sort_field": "",
            "sort_dir": "DESC",
            "limit": 500,
        }

        # Table
        m = re.search(r"FROM\s+(?:\w+\.)?(\w+)", sql, re.IGNORECASE)
        if m:
            config["table"] = m.group(1)

        # GROUP BY columns
        gm = re.search(
            r"GROUP\s+BY\s+(.*?)(?:HAVING|ORDER|LIMIT|$)", sql,
            re.IGNORECASE | re.DOTALL,
        )
        if gm:
            raw = gm.group(1).strip()
            if raw.upper() != "ALL":
                cols = re.findall(r'"([^"]+)"|\b([A-Za-z_]\w*)\b', raw)
                dims = [q or u for q, u in cols if (q or u)]
                config["dimensions"] = dims

        # Aggregations in SELECT
        for am in re.finditer(
            r'(SUM|AVG|COUNT|MIN|MAX|MEDIAN)\s*\(\s*(?:DISTINCT\s+)?"?([^"(),\n]+?)"?\s*\)\s+AS\s+"?([^",\n]+)"?',
            sql, re.IGNORECASE,
        ):
            agg   = am.group(1).upper()
            field = am.group(2).strip().strip('"')
            alias = am.group(3).strip().strip('"')
            config["measures"].append({"field": field, "agg": agg, "alias": alias})

        # LIMIT
        lm = re.search(r"LIMIT\s+(\d+)", sql, re.IGNORECASE)
        if lm:
            config["limit"] = int(lm.group(1))

        return config


def _build_filter_clause(f: dict) -> str:
    field = f["field"]
    op    = f["op"]
    val   = f.get("value", "")

    if op == "IS NULL":
        return f'"{field}" IS NULL'
    if op == "IS NOT NULL":
        return f'"{field}" IS NOT NULL'
    if op == "IN":
        parts = ", ".join(f"'{v.strip()}'" for v in str(val).split(",") if v.strip())
        return f'"{field}" IN ({parts})'
    if op == "LIKE":
        return f'"{field}" LIKE \'{val}\''
    # Numeric check
    try:
        float(val)
        return f'"{field}" {op} {val}'
    except (ValueError, TypeError):
        return f'"{field}" {op} \'{val}\''
