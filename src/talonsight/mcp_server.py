"""TalonSight MCP Server — exposes database tools to Hermes Agent.

Run automatically by Hermes Agent via stdio transport:
    talonsight-mcp

Or start manually for testing:
    talonsight-mcp --connection-id <id>

Connection config is read from ~/.talonsight/preferences.json
(last_connection key), written there whenever the user connects a
database in the Streamlit UI.

Tools exposed:
  list_tables        → all tables with row counts
  get_schema         → column definitions for one or more tables
  run_sql            → execute a SELECT query (safety-checked)
  get_sample_data    → preview rows from a table
  find_drivers       → metric decomposition across dimensions
  detect_change      → statistical change detection for a metric
  schema_summary     → compact schema + data profile for context
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── Lazy DB connection ────────────────────────────────────────────────────────

_connector = None
_schema_ctx = None
_core_inst = None


def _get_core():
    """Lazily initialise a TalonSight core instance from stored preferences.

    TalonSight.__init__ establishes the DB connection internally — there is
    no separate connect() method.  We pass a minimal LLM stub so the
    constructor doesn't try to reach an Ollama server at import time.
    """
    global _connector, _schema_ctx, _core_inst

    if _core_inst is not None:
        return _core_inst

    from talonsight.preferences import Preferences
    prefs = Preferences.load()
    conn_cfg = prefs.last_connection

    if not conn_cfg:
        raise RuntimeError(
            "No database connection found. "
            "Open TalonSight, connect to a database, then try again."
        )

    from talonsight.connection import ConnectionConfig
    from talonsight.core import TalonSight
    from talonsight.llm import LLMBackend, LLMResponse

    # Minimal no-op LLM — MCP server only needs the DB connection, not LLM calls
    class _NoOpLLM(LLMBackend):
        def generate_sql(self, *a, **kw) -> LLMResponse:
            return LLMResponse(sql="", explanation="")
        def generate_summary(self, *a, **kw) -> str:
            return ""
        def suggest_chart(self, *a, **kw) -> dict:
            return {}
        def suggest_followups(self, *a, **kw) -> list:
            return []
        def generate_explore_questions(self, *a, **kw) -> list:
            return []

    cfg = ConnectionConfig(**conn_cfg)
    # TalonSight connects to the DB in __init__ — no separate connect() needed
    ts = TalonSight(cfg, llm=_NoOpLLM())

    _core_inst = ts
    return ts


# ── Tool implementations ──────────────────────────────────────────────────────

def _list_tables() -> str:
    ts = _get_core()
    tables = ts.schema_context.tables
    if not tables:
        return "No tables found."
    lines = []
    for t in tables:
        rc = t.row_count
        rc_str = f"{rc:,}" if rc else "?"
        lines.append(f"- {t.name} ({rc_str} rows)")
    return "\n".join(lines)


def _get_schema(tables: list[str] = []) -> str:
    """Return schema with ACTUAL column names as they exist in the database.

    We query information_schema directly rather than using the introspected
    (normalised) names, because the introspector lowercases / de-spaces names
    which then fail in SQL (e.g. 'customer_id' vs 'Customer ID').
    Columns with spaces or special characters are shown pre-quoted.
    """
    ts = _get_core()
    try:
        tbl_filter = ""
        if tables:
            quoted = ", ".join(f"'{t}'" for t in tables)
            tbl_filter = f" AND table_name IN ({quoted})"
        df = ts._db.execute_query(
            f"SELECT table_name, column_name, data_type "
            f"FROM information_schema.columns "
            f"WHERE table_schema NOT IN ('information_schema', 'pg_catalog'){tbl_filter} "
            f"ORDER BY table_name, ordinal_position"
        )
        if df is None or df.empty:
            raise ValueError("empty result")

        lines: list[str] = []
        for tname, grp in df.groupby("table_name", sort=False):
            lines.append(f"\nTable: {tname}")
            for _, row in grp.iterrows():
                col = row["column_name"]
                dtype = row["data_type"]
                safe = f'"{col}"' if any(c in col for c in (' ', '-', '.')) else col
                lines.append(f"  {safe}  {dtype}")
        return "\n".join(lines)

    except Exception:
        all_tables = ts.schema_context.tables
        if tables:
            tl = [t.lower() for t in tables]
            all_tables = [t for t in all_tables if t.name.lower() in tl]
        if not all_tables:
            return f"No tables found matching: {tables}"
        lines = []
        for tbl in all_tables:
            lines.append(f"\nTable: {tbl.name}")
            for col in tbl.columns:
                lines.append(f"  {col.name}  {col.type}")
        return "\n".join(lines)


def _run_sql(sql: str, limit: int = 200) -> str:
    ts = _get_core()
    from talonsight.safety import validate_sql, RiskLevel
    import re

    verdict = validate_sql(sql)
    if verdict.level == RiskLevel.BLOCKED:
        return f"BLOCKED: {verdict.reason}"

    sql = sql.rstrip("; \n\t")
    if not re.search(r'\bLIMIT\b', sql, re.IGNORECASE):
        sql = sql + f"\nLIMIT {min(limit, 500)}"

    df = ts._db.execute_query(sql)
    if df is None or df.empty:
        return "Query returned no rows."
    return df.to_markdown(index=False)


def _get_sample_data(table: str, n: int = 5) -> str:
    ts = _get_core()
    tbl_info = next(
        (t for t in ts.schema_context.tables if t.name.lower() == table.lower()), None
    )
    fqn = f'"{tbl_info.schema}"."{tbl_info.name}"' if (tbl_info and tbl_info.schema) else f'"{table}"'
    cols_sql = ", ".join(f'"{c.name}"' for c in tbl_info.columns[:6]) if tbl_info else "*"
    df = ts._db.execute_query(f"SELECT {cols_sql} FROM {fqn} LIMIT {min(n, 20)}")
    if df is None or df.empty:
        return f"No data from {table}."
    return df.to_markdown(index=False)


def _find_drivers(metric: str, dimensions: list[str]) -> str:
    """Run GROUP BY queries to find which dimension values drive a metric most."""
    ts = _get_core()
    results = []
    for dim in dimensions[:4]:  # cap at 4 dimensions
        # Try to build a reasonable query: aggregate metric grouped by dimension
        # metric may be "orders.ordertotal" or "SUM(freight)" etc.
        col_expr = metric.split(".")[-1] if "." in metric else metric
        # Find a table that has both columns
        table_name = None
        for tbl in ts.schema_context.tables:
            col_names = [c.name.lower() for c in tbl.columns]
            if dim.lower() in col_names and col_expr.lower() in col_names:
                table_name = tbl.name
                schema_prefix = f'"{tbl.schema}".' if tbl.schema else ""
                break
        if not table_name:
            results.append(f"dimension '{dim}': could not find a table with both '{dim}' and '{col_expr}'")
            continue
        sql = (
            f'SELECT "{dim}", SUM("{col_expr}") AS metric_total, COUNT(*) AS row_count '
            f'FROM {schema_prefix}"{table_name}" '
            f'GROUP BY "{dim}" ORDER BY metric_total DESC LIMIT 20'
        )
        df = ts._db.execute_query(sql)
        if df is None or df.empty:
            results.append(f"dimension '{dim}': no data")
        else:
            results.append(f"## {dim} breakdown\n{df.to_markdown(index=False)}")
    return "\n\n".join(results) if results else "No results."


def _detect_change(metric: str, timeframe: str, comparison: str = "prior period") -> str:
    """Return a prompt hint — actual period-over-period SQL should use run_sql."""
    return (
        f"To detect change in '{metric}' for '{timeframe}' vs '{comparison}', "
        f"use run_sql to compare aggregates for each period directly. "
        f"Example pattern: SELECT period, SUM(metric) ... GROUP BY period ORDER BY period."
    )


def _schema_summary() -> str:
    ts = _get_core()
    return ts._build_agent_schema_str() if hasattr(ts, "_build_agent_schema_str") else _get_schema([])


# ── MCP Server (FastMCP) ──────────────────────────────────────────────────────

def _build_server():
    """Build and return the FastMCP server instance."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        raise ImportError(
            "The `mcp` package is required for Analyst mode.\n"
            "Install it with:  pip install mcp"
        )

    mcp = FastMCP(
        "TalonSight",
        instructions=(
            "You are a data analyst connected to a SQL database via TalonSight. "
            "ALWAYS answer by calling run_sql with valid SQL. "
            "Step 1: if you don't know the schema, call get_schema([]) first. "
            "Step 2: write a SELECT query using EXACT column names from the schema. "
            "Step 3: call run_sql and use the results to answer the question. "
            "Only SELECT queries are allowed — no INSERT/UPDATE/DELETE. "
            "Never guess column names — always check the schema first."
        ),
    )

    @mcp.tool()
    def list_tables() -> str:
        """List all available database tables with their row counts."""
        return _list_tables()

    @mcp.tool()
    def get_schema(tables: list[str] = []) -> str:
        """Get column definitions for the specified tables. Pass empty list for all tables."""
        return _get_schema(tables)

    @mcp.tool()
    def run_sql(sql: str, limit: int = 200) -> str:
        """Execute a SELECT SQL query and return results as a markdown table.
        Only SELECT is allowed. Limit defaults to 200 rows."""
        return _run_sql(sql, limit)

    @mcp.tool()
    def get_sample_data(table: str, n: int = 5) -> str:
        """Fetch a small preview of rows from a table (max 20 rows)."""
        return _get_sample_data(table, n)

    @mcp.tool()
    def find_drivers(metric: str, dimensions: list[str]) -> str:
        """Run GROUP BY queries to find which dimension values drive a metric.
        metric: column name e.g. 'freight_value' or 'order_total'.
        dimensions: list of column names to segment by e.g. ['customer_state'].
        Returns ranked markdown tables — one per dimension."""
        return _find_drivers(metric, dimensions)

    @mcp.tool()
    def detect_change(metric: str, timeframe: str, comparison: str = "prior period") -> str:
        """Hint for period-over-period comparisons. Use run_sql for actual data."""
        return _detect_change(metric, timeframe, comparison)

    @mcp.tool()
    def schema_summary() -> str:
        """Return a compact schema + data profile suitable for LLM context injection.
        Call this once at the start of an investigation."""
        return _schema_summary()

    return mcp


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    """Entry point for the `talonsight-mcp` command."""
    parser = argparse.ArgumentParser(
        description="TalonSight MCP server — exposes database tools to Hermes Agent"
    )
    parser.add_argument(
        "--transport", choices=["stdio", "http"], default="stdio",
        help="MCP transport (default: stdio)",
    )
    parser.add_argument(
        "--port", type=int, default=8765,
        help="Port for HTTP transport (default: 8765)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        stream=sys.stderr,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    )

    try:
        server = _build_server()
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.transport == "stdio":
        server.run(transport="stdio")
    else:
        server.run(transport="http", host="localhost", port=args.port)


if __name__ == "__main__":
    main()
