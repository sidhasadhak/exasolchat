"""Data profiler — lightweight column distributions and ranges for agent context.

Runs once at connect time (or lazily on first ask_agent call).  Injects a
compact profile block into every agent system prompt so the agent starts
knowing what values are in each column and never needs to call
get_sample_data or get_column_stats for basic discovery.

v1 strategy
-----------
DuckDB   : one SUMMARIZE per table — extremely fast, no per-column queries.
Other DBs: selective queries — top-5 values for low-cardinality columns,
           min/max/avg for numeric, date range for timestamp columns.
           Tables with > ROW_LIMIT rows are sampled instead of full-scanned.

Output format (compact, token-efficient)
-----------------------------------------
  Table: orders [112,650 rows]
    order_status       → delivered(96k) · shipped(1.1k) · canceled(625)
    order_purchase_timestamp → 2016-09-04 → 2018-10-17
    freight_value      → 0.00 → 409.68, avg 19.99
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from talonsight.connection import DatabaseConnection
    from talonsight.schema import SchemaContext
    from talonsight.graph import SchemaGraph, TableNode

logger = logging.getLogger(__name__)

ROW_LIMIT   = 500_000   # use TABLESAMPLE above this threshold
MAX_TABLES  = 20        # profile at most this many tables
MAX_COLS    = 30        # profile at most this many columns per table
TOP_N       = 6         # top-N values for categorical columns
LOW_CARD    = 50        # treat as categorical if distinct ≤ this


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ColumnProfile:
    name: str
    col_type: str
    null_pct: float = 0.0
    n_distinct: int = 0
    min_val:  Optional[str] = None
    max_val:  Optional[str] = None
    avg_val:  Optional[str] = None
    top_values: list[tuple[str, int]] = field(default_factory=list)  # [(val, count)]


@dataclass
class TableProfile:
    name: str
    fqn: str
    row_count: int
    columns: list[ColumnProfile] = field(default_factory=list)

    def to_context_lines(self) -> list[str]:
        """Compact lines for agent prompt injection."""
        lines = [f"\nTable: {self.fqn} [{self.row_count:,} rows]"]
        for cp in self.columns:
            line = _format_column(cp)
            if line:
                lines.append(f"  {cp.name:<32} {line}")
        return lines


# ── Main class ────────────────────────────────────────────────────────────────

class DataProfiler:
    """Profiles every table in the connected schema.

    Usage:
        profiler = DataProfiler(connector, schema_context, schema_graph)
        profiler.run()                      # executes DB queries
        prompt_block = profiler.to_agent_context()
    """

    def __init__(
        self,
        connector: "DatabaseConnection",
        schema_context: "SchemaContext",
        schema_graph: "SchemaGraph",
    ) -> None:
        self._db     = connector
        self._ctx    = schema_context
        self._graph  = schema_graph
        self.profiles: dict[str, TableProfile] = {}
        self._done   = False

    def run(self) -> "DataProfiler":
        """Execute profiling queries. Safe to call multiple times (no-op after first)."""
        if self._done:
            return self
        tables = self._ctx.tables[:MAX_TABLES]
        is_duck = getattr(self._db, "is_duckdb", False)
        for tbl in tables:
            try:
                fqn = f'"{tbl.schema}"."{tbl.name}"' if tbl.schema else f'"{tbl.name}"'
                if is_duck:
                    tp = self._profile_duckdb(tbl.name, fqn, tbl.row_count or 0)
                else:
                    tp = self._profile_generic(tbl.name, fqn, tbl.columns, tbl.row_count or 0)
                if tp:
                    self.profiles[tbl.name.lower()] = tp
            except Exception as exc:
                logger.debug("Profiler: skipped %s — %s", tbl.name, exc)
        self._done = True
        return self

    # ── DuckDB fast path ──────────────────────────────────────────────────────

    def _profile_duckdb(self, name: str, fqn: str, row_count: int) -> Optional[TableProfile]:
        """Use SUMMARIZE for a single-query full profile."""
        df = self._db.execute_query(f"SELECT * FROM (SUMMARIZE {fqn}) LIMIT 60")
        if df is None or df.empty:
            return None

        tp = TableProfile(name=name, fqn=fqn, row_count=row_count)
        col_map: dict[str, dict] = {}
        for _, row in df.iterrows():
            col_name = str(row.get("column_name", ""))
            col_map[col_name] = row.to_dict()

        # Enrich low-cardinality columns with top-N values
        low_card_cols = [
            c for c, r in col_map.items()
            if _safe_int(r.get("approx_unique")) is not None
            and _safe_int(r.get("approx_unique")) <= LOW_CARD
        ]
        top_vals: dict[str, list[tuple[str, int]]] = {}
        for col in low_card_cols[:10]:
            try:
                vdf = self._db.execute_query(
                    f'SELECT "{col}" AS v, COUNT(*) AS n FROM {fqn} '
                    f'WHERE "{col}" IS NOT NULL GROUP BY "{col}" '
                    f'ORDER BY n DESC LIMIT {TOP_N}'
                )
                if vdf is not None and not vdf.empty:
                    top_vals[col] = [(str(r["v"]), int(r["n"])) for _, r in vdf.iterrows()]
            except Exception:
                pass

        graph_node = self._graph.nodes.get(name.lower()) if self._graph else None

        for col in self._ctx.tables[
            next((i for i, t in enumerate(self._ctx.tables)
                  if t.name.lower() == name.lower()), 0)
        ].columns[:MAX_COLS]:
            row = col_map.get(col.name, {})
            n_dist = _safe_int(row.get("approx_unique")) or 0
            null_pct = _safe_float(row.get("null_percentage")) or 0.0

            cp = ColumnProfile(
                name=col.name,
                col_type=col.type or "",
                null_pct=null_pct,
                n_distinct=n_dist,
                min_val=_clean_val(row.get("min")),
                max_val=_clean_val(row.get("max")),
                avg_val=_clean_val(row.get("mean")),
                top_values=top_vals.get(col.name, []),
            )
            # Only include columns that add signal
            if _has_signal(cp, graph_node, col.name):
                tp.columns.append(cp)

        return tp

    # ── Generic path (Postgres, SQLAlchemy, Exasol) ───────────────────────────

    def _profile_generic(
        self, name: str, fqn: str, columns, row_count: int
    ) -> Optional[TableProfile]:
        """Selective profiling — one query per interesting column, capped."""
        tp = TableProfile(name=name, fqn=fqn, row_count=row_count)
        graph_node = self._graph.nodes.get(name.lower()) if self._graph else None

        for col in columns[:MAX_COLS]:
            try:
                cp = self._profile_column_generic(fqn, col, row_count)
                if cp and _has_signal(cp, graph_node, col.name):
                    tp.columns.append(cp)
            except Exception as exc:
                logger.debug("Profiler: skipped %s.%s — %s", name, col.name, exc)

        return tp

    def _profile_column_generic(self, fqn: str, col, row_count: int) -> Optional[ColumnProfile]:
        cname  = col.name
        ctype  = (col.type or "").lower()
        is_num = any(k in ctype for k in ("int","float","double","numeric","decimal","real","number"))
        is_ts  = any(k in ctype for k in ("date","time","timestamp"))
        is_txt = any(k in ctype for k in ("char","text","varchar","string","enum"))

        if not (is_num or is_ts or is_txt):
            return None

        # Null + distinct count
        df = self._db.execute_query(
            f'SELECT COUNT(*) AS total, '
            f'SUM(CASE WHEN "{cname}" IS NULL THEN 1 ELSE 0 END) AS nulls, '
            f'COUNT(DISTINCT "{cname}") AS dist FROM {fqn}'
        )
        if df is None or df.empty:
            return None
        total  = int(df.iloc[0]["total"]) or 1
        nulls  = int(df.iloc[0]["nulls"])
        n_dist = int(df.iloc[0]["dist"])
        null_pct = round(100 * nulls / total, 1)

        cp = ColumnProfile(name=cname, col_type=col.type or "",
                           null_pct=null_pct, n_distinct=n_dist)

        if is_num:
            r2 = self._db.execute_query(
                f'SELECT MIN("{cname}") AS mn, MAX("{cname}") AS mx, '
                f'AVG("{cname}") AS av FROM {fqn}'
            )
            if r2 is not None and not r2.empty:
                cp.min_val = _clean_val(r2.iloc[0]["mn"])
                cp.max_val = _clean_val(r2.iloc[0]["mx"])
                cp.avg_val = _clean_val(r2.iloc[0]["av"])

        elif is_ts:
            r2 = self._db.execute_query(
                f'SELECT MIN("{cname}") AS mn, MAX("{cname}") AS mx FROM {fqn}'
            )
            if r2 is not None and not r2.empty:
                cp.min_val = _clean_val(r2.iloc[0]["mn"])
                cp.max_val = _clean_val(r2.iloc[0]["mx"])

        if is_txt and n_dist <= LOW_CARD:
            r3 = self._db.execute_query(
                f'SELECT "{cname}" AS v, COUNT(*) AS n FROM {fqn} '
                f'WHERE "{cname}" IS NOT NULL '
                f'GROUP BY "{cname}" ORDER BY n DESC LIMIT {TOP_N}'
            )
            if r3 is not None and not r3.empty:
                cp.top_values = [(str(row["v"]), int(row["n"])) for _, row in r3.iterrows()]

        return cp

    # ── Output ────────────────────────────────────────────────────────────────

    def to_agent_context(self) -> str:
        """Compact block for the agent system prompt."""
        if not self.profiles:
            return ""
        lines = ["DATA PROFILE (distributions and ranges — use these instead of get_sample_data):"]
        for tp in self.profiles.values():
            lines.extend(tp.to_context_lines())
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialisable for UI display."""
        return {
            name: {
                "name": tp.name,
                "fqn": tp.fqn,
                "row_count": tp.row_count,
                "columns": [
                    {
                        "name": cp.name,
                        "type": cp.col_type,
                        "null_pct": cp.null_pct,
                        "n_distinct": cp.n_distinct,
                        "min": cp.min_val,
                        "max": cp.max_val,
                        "avg": cp.avg_val,
                        "top_values": cp.top_values,
                    }
                    for cp in tp.columns
                ],
            }
            for name, tp in self.profiles.items()
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_column(cp: ColumnProfile) -> str:
    """Format a ColumnProfile as a compact single line."""
    parts: list[str] = []

    if cp.top_values:
        formatted = " · ".join(
            f"{v}({_fmt_n(n)})" for v, n in cp.top_values[:TOP_N]
        )
        parts.append(formatted)
    elif cp.min_val is not None and cp.max_val is not None:
        rng = f"{cp.min_val} → {cp.max_val}"
        if cp.avg_val is not None:
            rng += f", avg {cp.avg_val}"
        parts.append(rng)

    if cp.null_pct > 10:
        parts.append(f"{cp.null_pct:.0f}% null")

    return "  ·  ".join(parts) if parts else ""


def _has_signal(cp: ColumnProfile, node: Optional["TableNode"], col_name: str) -> bool:
    """Only include columns that carry useful agent signal."""
    col_l = col_name.lower()
    # Always include measure, date, and FK columns
    if node:
        if col_name in node.measures:        return True
        if col_name in node.date_columns:    return True
        if col_name in node.fk_columns:      return False  # FK IDs add noise not signal
    # Include if it has top-values distribution or a numeric range
    if cp.top_values:                        return True
    if cp.min_val is not None:               return True
    # Include high-null columns (quality signal)
    if cp.null_pct > 20:                     return True
    return False


def _safe_int(val) -> Optional[int]:
    try:
        return int(float(str(val))) if val is not None and str(val) not in ("", "None") else None
    except (ValueError, TypeError):
        return None


def _safe_float(val) -> Optional[float]:
    try:
        return float(str(val)) if val is not None and str(val) not in ("", "None") else None
    except (ValueError, TypeError):
        return None


def _clean_val(val) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip()
    if s in ("", "None", "NaT", "nan"):
        return None
    # Trim timestamps to date precision for readability
    s = re.sub(r"(\d{4}-\d{2}-\d{2})[ T]\d{2}:\d{2}:\d{2}.*", r"\1", s)
    # Round long decimals
    try:
        f = float(s)
        return f"{f:,.2f}" if abs(f) < 1e9 else f"{f:,.0f}"
    except ValueError:
        return s[:40]  # cap string length


def _fmt_n(n: int) -> str:
    """Format a count compactly: 96478 → 96k, 1234567 → 1.2M."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n//1_000}k"
    return str(n)
