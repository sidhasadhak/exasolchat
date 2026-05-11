"""Database schema introspection.

Primary: pyexasol for ExasolDB (richer metadata, faster).
Fallback: SQLAlchemy for any other database.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

# Common date format patterns, ordered most-specific first
_DATE_PATTERNS = [
    (r"^\d{4}-\d{2}-\d{2}$",            "%Y-%m-%d",        "ISO 8601 (YYYY-MM-DD)"),
    (r"^\d{4}/\d{2}/\d{2}$",            "%Y/%m/%d",        "YYYY/MM/DD"),
    (r"^\d{2}-\d{2}-\d{4}$",            "%d-%m-%Y",        "DD-MM-YYYY"),
    (r"^\d{2}/\d{2}/\d{4}$",            "%m/%d/%Y",        "MM/DD/YYYY"),
    (r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}", "%Y-%m-%d %H:%M",  "ISO 8601 datetime"),
    (r"^\d{2}-[A-Za-z]{3}-\d{4}$",      "%d-%b-%Y",        "DD-Mon-YYYY"),
    (r"^[A-Za-z]+ \d{1,2}, \d{4}$",     "%B %d, %Y",       "Month DD, YYYY"),
]


def _norm_col_name(name: str) -> str:
    """Normalize a raw DB column name: strip, lowercase, whitespace → underscore.

    Examples:  "Order Date"  → "order_date"
               "Customer ID" → "customer_id"
               "  Sales  "   → "sales"
               "profit"      → "profit"   (unchanged)
    """
    return re.sub(r"\s+", "_", name.strip()).lower()


def _detect_date_format(samples: list) -> Optional[str]:
    """Return a human-readable description of the date format, or None."""
    str_samples = [str(s) for s in samples if s is not None][:5]
    for pattern, fmt, label in _DATE_PATTERNS:
        if all(re.match(pattern, s) for s in str_samples if s):
            return f"{label} (strptime format: '{fmt}')"
    return None


@dataclass
class ColumnInfo:
    name: str           # normalized: "order_date"
    type: str
    nullable: bool
    primary_key: bool
    foreign_key: Optional[str] = None
    comment: Optional[str] = None
    original_name: str = ""   # raw DB name: "Order Date"

    def __post_init__(self):
        if not self.original_name:
            self.original_name = self.name


@dataclass
class TableInfo:
    name: str
    schema: Optional[str]
    columns: list[ColumnInfo] = field(default_factory=list)
    row_count: Optional[int] = None
    sample_values: dict[str, list] = field(default_factory=dict)
    comment: Optional[str] = None


# ── Join-key inference ────────────────────────────────────────────────

# Suffixes stripped when normalising column names for join detection.
# Listed longest-first so greedy stripping is correct.
_JOIN_SUFFIXES = (
    "identifier", "number", "pseudonym", "pseudonyms",
    "code", "num", "key", "id", "no",
)


def _normalise_for_join(name: str) -> str:
    """Collapse a column name to its semantic root for join matching.

    "customer_id"         → "customer"
    "cust_id"             → "cust"
    "order_id"            → "order"
    "orderid"             → "order"
    "order_id_pseudonyms" → "order"
    """
    n = re.sub(r"[\s_]+", "", name.lower())
    # Strip compound suffixes iteratively (e.g. _id_pseudonyms → _id → root)
    for _ in range(4):
        stripped = False
        for suf in _JOIN_SUFFIXES:
            if n.endswith(suf) and len(n) > len(suf) + 2:
                n = n[: -len(suf)]
                stripped = True
                break
        if not stripped:
            break
    return n


def get_join_map(tables: list["TableInfo"]) -> dict:
    """Analyse column overlap across tables and return structured join data.

    Returns::
        {
          "joins":    [{"t1", "c1", "t2", "c2", "match"}, ...],
          "no_join":  [(t1_name, t2_name), ...]   # pairs with NO shared column
        }

    Phase 1 — exact: identical column name in ≥ 2 tables.
    Phase 2 — fuzzy: different column names with the same normalised root.
    """
    if len(tables) < 2:
        return {"joins": [], "no_join": []}

    exact_map: dict[str, list[str]] = {}
    fuzzy_map: dict[str, list[tuple[str, str]]] = {}

    for t in tables:
        for c in t.columns:
            exact_map.setdefault(c.name, []).append(t.name)
            root = _normalise_for_join(c.name)
            if len(root) > 2:
                fuzzy_map.setdefault(root, []).append((t.name, c.name))

    joins: list[dict] = []
    seen_col_pairs: set[frozenset] = set()
    joined_table_pairs: set[frozenset] = set()

    # Phase 1
    for col, tbls in exact_map.items():
        unique = list(dict.fromkeys(tbls))
        if len(unique) < 2:
            continue
        for i, t1 in enumerate(unique):
            for t2 in unique[i + 1:]:
                pair = frozenset({f"{t1}.{col}", f"{t2}.{col}"})
                if pair not in seen_col_pairs:
                    seen_col_pairs.add(pair)
                    joined_table_pairs.add(frozenset({t1, t2}))
                    joins.append({"t1": t1, "c1": col, "t2": t2, "c2": col, "match": "exact"})

    # Phase 2
    for root, occurrences in fuzzy_map.items():
        per_table: dict[str, str] = {}
        for tbl, col in occurrences:
            if tbl not in per_table:
                per_table[tbl] = col
        unique_tables = list(per_table)
        if len(unique_tables) < 2:
            continue
        for i, t1 in enumerate(unique_tables):
            for t2 in unique_tables[i + 1:]:
                c1, c2 = per_table[t1], per_table[t2]
                if c1 == c2:
                    continue
                pair = frozenset({f"{t1}.{c1}", f"{t2}.{c2}"})
                if pair not in seen_col_pairs:
                    seen_col_pairs.add(pair)
                    joined_table_pairs.add(frozenset({t1, t2}))
                    joins.append({"t1": t1, "c1": c1, "t2": t2, "c2": c2,
                                  "match": f"similar (root '{root}')"})

    # Pairs with NO detected direct join path
    table_names = [t.name for t in tables]
    no_join = [
        (t1, t2)
        for i, t1 in enumerate(table_names)
        for t2 in table_names[i + 1:]
        if frozenset({t1, t2}) not in joined_table_pairs
    ]

    return {"joins": joins, "no_join": no_join}


def _infer_join_hints(tables: list["TableInfo"]) -> str:
    """Format the join map as a prompt block for the LLM.

    Critically includes a 'NO DIRECT JOIN' section so the LLM does not
    hallucinate column names to connect tables that have no shared key.
    """
    jmap = get_join_map(tables)
    if not jmap["joins"] and not jmap["no_join"]:
        return ""

    lines: list[str] = []

    if jmap["joins"]:
        lines.append("DETECTED JOIN PATHS (use these ON / USING clauses):")
        for j in jmap["joins"]:
            if j["c1"] == j["c2"]:
                lines.append(f"  {j['t1']}.{j['c1']} = {j['t2']}.{j['c2']}  [{j['match']}]")
            else:
                lines.append(
                    f"  {j['t1']}.{j['c1']} = {j['t2']}.{j['c2']}"
                    f"  [{j['match']}]"
                )

    if jmap["no_join"]:
        lines.append(
            "\nNO DIRECT JOIN DETECTED — do NOT attempt to join these pairs "
            "(no shared column exists):"
        )
        for t1, t2 in jmap["no_join"]:
            lines.append(
                f"  {t1} ↔ {t2}  → an intermediate table is required"
            )

    return "\n".join(lines)


@dataclass
class SchemaContext:
    tables: list[TableInfo]
    dialect: str
    extra_context: str = ""

    def to_prompt(self) -> str:
        """Render schema as compact LLM-ready context."""
        lines = [f"DATABASE DIALECT: {self.dialect}", ""]
        for t in self.tables:
            header = f"TABLE: {t.schema + '.' if t.schema else ''}{t.name}"
            if t.row_count is not None:
                header += f"  (~{t.row_count:,} rows)"
            if t.comment:
                header += f"  -- {t.comment}"
            lines.append(header)

            for c in t.columns:
                parts = [f"  {c.name} {c.type}"]
                if c.primary_key:
                    parts.append("PK")
                if not c.nullable:
                    parts.append("NOT NULL")
                if c.foreign_key:
                    parts.append(f"FK->{c.foreign_key}")
                if c.comment:
                    parts.append(f"-- {c.comment}")
                lines.append(" | ".join(parts))

            if t.sample_values:
                lines.append("  Sample values:")
                for col, vals in t.sample_values.items():
                    display = ", ".join(repr(v) for v in vals[:5])
                    lines.append(f"    {col}: [{display}]")
            lines.append("")

        # DuckDB-specific: warn about VARCHAR columns that look like dates.
        # date_trunc / date_diff fail with "No function matches ... VARCHAR"
        # unless the column is explicitly cast to DATE/TIMESTAMP first.
        if "duckdb" in self.dialect.lower():
            _DATE_NAME_TOKENS = {
                "date", "time", "day", "month", "year",
                "created", "updated", "modified", "at",
                "timestamp", "dt", "period", "week",
            }
            _VARCHAR_TYPES = {"varchar", "text", "character varying", "string", "char"}
            cast_warnings: list[str] = []
            for t in self.tables:
                for c in t.columns:
                    col_type_lc = c.type.lower().split("(")[0].strip()
                    if col_type_lc not in _VARCHAR_TYPES:
                        continue
                    # Check if any date-like token appears in the column name
                    col_tokens = set(re.split(r"[\s_]", c.original_name.lower()))
                    col_tokens.update(re.split(r"[\s_]", c.name.lower()))
                    if col_tokens & _DATE_NAME_TOKENS:
                        display = c.original_name if c.original_name != c.name else c.name
                        cast_warnings.append(
                            f"  {t.name}.{display!r} is VARCHAR — wrap in CAST(\"{display}\" AS DATE) "
                            "or \"" + display + "\"::DATE before using date_trunc / date_diff / EXTRACT"
                        )
            if cast_warnings:
                lines.append("⚠ DUCKDB CAST REQUIRED — these date-like columns are VARCHAR and MUST be cast:")
                lines.extend(cast_warnings)
                lines.append("")

        # Auto-detected join candidates help the LLM pick correct ON clauses
        join_hints = _infer_join_hints(self.tables)
        if join_hints:
            lines.append(join_hints)
            lines.append("")

        if self.extra_context:
            lines.append("ADDITIONAL CONTEXT:")
            lines.append(self.extra_context)
        return "\n".join(lines)

    @property
    def table_names(self) -> list[str]:
        return [t.name for t in self.tables]

    def col_name_map(self) -> dict[str, str]:
        """Return {normalized_name: original_db_name} for every column that changed."""
        return {
            c.name: c.original_name
            for t in self.tables
            for c in t.columns
            if c.name != c.original_name
        }

    def denormalize_sql(self, sql: str) -> str:
        """Rewrite normalized column names in LLM/builder SQL to quoted originals.

        Two patterns handled:
          "order_date"  →  "Order Date"   (builder always quotes; swap the name inside)
          order_date    →  "Order Date"   (LLM wrote unquoted; add quotes + original name)

        Both replacements are quote-aware: text inside existing double-quoted
        strings (e.g. aliases like "Total order_id") is left untouched.
        """
        mapping = self.col_name_map()
        for norm, orig in mapping.items():
            # Replace "normalized" → "original" (quoted form, e.g. builder output)
            sql = _replace_quoted_identifier(sql, norm, orig)
            # Replace bare normalized → "original" (unquoted LLM output, spaces only)
            if " " in orig:
                sql = _replace_unquoted_identifier(sql, norm, orig)
        return sql


# ─── Quote-aware SQL identifier replacement ──────────────────────────

def _replace_quoted_identifier(sql: str, norm: str, orig: str) -> str:
    """Replace "norm" → "orig" only when norm appears as a standalone quoted identifier.

    Uses a quote-aware regex so that occurrences of "norm" that are part of a
    longer quoted string (e.g. "Total norm") are left untouched.
    The pattern matches either any full quoted token or exactly "norm", and
    only replaces the exact match.
    """
    pattern = re.compile(r'"[^"]*"')

    def replacer(m: re.Match) -> str:
        token = m.group(0)
        # Only swap if it is exactly the quoted normalized name
        if token == f'"{norm}"':
            return f'"{orig}"'
        return token  # e.g. "Total order_id" — leave unchanged

    return pattern.sub(replacer, sql)


def _replace_unquoted_identifier(sql: str, norm: str, orig: str) -> str:
    """Replace bare `norm` → `"orig"` only in unquoted regions of the SQL.

    Matches either a full quoted token (skipped) or the whole-word pattern
    (replaced). This prevents touching identifiers inside alias strings like
    "Total order_id".
    """
    pattern = re.compile(rf'"[^"]*"|\b{re.escape(norm)}\b')

    def replacer(m: re.Match) -> str:
        token = m.group(0)
        if token.startswith('"'):
            return token  # inside a quoted string — leave it alone
        return f'"{orig}"'

    return pattern.sub(replacer, sql)


# ─── Exasol introspection via pyexasol ──────────────────────────────

def introspect_exasol(
    conn,  # pyexasol connection
    schema: Optional[str] = None,
    sample_rows: int = 3,
    include_tables: Optional[list[str]] = None,
    exclude_tables: Optional[list[str]] = None,
) -> SchemaContext:
    """Introspect an ExasolDB schema using pyexasol."""
    target_schema = schema or conn.attr.get("current_schema", "")
    if not target_schema:
        # Get default schema
        result = conn.execute("SELECT CURRENT_SCHEMA")
        target_schema = result.fetchone()[0]

    # Get all tables in schema
    tables_query = """
        SELECT TABLE_NAME, TABLE_ROW_COUNT, TABLE_COMMENT
        FROM SYS.EXA_ALL_TABLES
        WHERE TABLE_SCHEMA = :schema
        ORDER BY TABLE_NAME
    """
    tables_result = conn.execute(tables_query, {"schema": target_schema.upper()})

    exclude = {t.upper() for t in (exclude_tables or [])}
    include = {t.upper() for t in (include_tables or [])} if include_tables else None

    tables: list[TableInfo] = []
    for row in tables_result:
        table_name = row[0]
        if table_name.upper() in exclude:
            continue
        if include and table_name.upper() not in include:
            continue

        row_count = row[1]
        table_comment = row[2]

        # Get columns
        cols_query = """
            SELECT COLUMN_NAME, COLUMN_TYPE, COLUMN_IS_NULLABLE,
                   COLUMN_COMMENT, COLUMN_ORDINAL_POSITION
            FROM SYS.EXA_ALL_COLUMNS
            WHERE COLUMN_SCHEMA = :schema AND COLUMN_TABLE = :table
            ORDER BY COLUMN_ORDINAL_POSITION
        """
        cols_result = conn.execute(
            cols_query, {"schema": target_schema.upper(), "table": table_name}
        )

        # Get primary key columns
        pk_query = """
            SELECT COLUMN_NAME
            FROM SYS.EXA_ALL_CONSTRAINT_COLUMNS
            WHERE CONSTRAINT_SCHEMA = :schema
              AND CONSTRAINT_TABLE = :table
              AND CONSTRAINT_TYPE = 'PRIMARY KEY'
        """
        try:
            pk_result = conn.execute(
                pk_query, {"schema": target_schema.upper(), "table": table_name}
            )
            pk_cols = {r[0] for r in pk_result}
        except Exception:
            pk_cols = set()

        # Get foreign keys
        fk_query = """
            SELECT cc.COLUMN_NAME, cc.REFERENCED_SCHEMA,
                   cc.REFERENCED_TABLE, cc.REFERENCED_COLUMN
            FROM SYS.EXA_ALL_CONSTRAINT_COLUMNS cc
            WHERE cc.CONSTRAINT_SCHEMA = :schema
              AND cc.CONSTRAINT_TABLE = :table
              AND cc.CONSTRAINT_TYPE = 'FOREIGN KEY'
        """
        fk_map: dict[str, str] = {}
        try:
            fk_result = conn.execute(
                fk_query, {"schema": target_schema.upper(), "table": table_name}
            )
            for fk_row in fk_result:
                fk_map[fk_row[0]] = f"{fk_row[1]}.{fk_row[2]}.{fk_row[3]}"
        except Exception:
            pass

        columns = []
        for col_row in cols_result:
            raw = col_row[0]
            columns.append(ColumnInfo(
                name=_norm_col_name(raw),
                original_name=raw,
                type=col_row[1],
                nullable=col_row[2],
                primary_key=raw in pk_cols,
                foreign_key=fk_map.get(raw),
                comment=col_row[3],
            ))

        # Sample values (keys normalized to match col.name)
        sample_values: dict[str, list] = {}
        if sample_rows > 0 and row_count and row_count > 0:
            try:
                qualified = f'"{target_schema}"."{table_name}"'
                sample_result = conn.execute(
                    f"SELECT * FROM {qualified} LIMIT {sample_rows}"
                )
                sample_data = sample_result.fetchall()
                col_names = [c[0] for c in sample_result.description()]
                for i, cn in enumerate(col_names):
                    vals = [row[i] for row in sample_data if row[i] is not None]
                    if vals:
                        sample_values[_norm_col_name(cn)] = vals[:5]
            except Exception:
                pass

        tables.append(TableInfo(
            name=table_name,
            schema=target_schema,
            columns=columns,
            row_count=row_count,
            sample_values=sample_values,
            comment=table_comment,
        ))

    return SchemaContext(tables=tables, dialect="exasol")


# ─── DuckDB native introspection ────────────────────────────────────

def introspect_duckdb(
    conn,  # duckdb.DuckDBPyConnection
    schema: Optional[str] = None,
    sample_rows: int = 3,
    include_tables: Optional[list[str]] = None,
    exclude_tables: Optional[list[str]] = None,
) -> SchemaContext:
    """Introspect a DuckDB database using information_schema.

    DuckDB supports querying Parquet/CSV files, attached databases, and
    in-memory tables. This reads the catalog via information_schema which
    covers all registered tables and views.
    """
    target_schema = schema or "main"

    # Get tables (tables + views)
    tables_query = f"""
        SELECT table_name, table_type
        FROM information_schema.tables
        WHERE table_schema = '{target_schema}'
        ORDER BY table_name
    """
    tables_result = conn.execute(tables_query).fetchall()

    exclude = {t.lower() for t in (exclude_tables or [])}
    include = {t.lower() for t in (include_tables or [])} if include_tables else None

    tables: list[TableInfo] = []
    for row in tables_result:
        table_name = row[0]
        table_type = row[1]  # BASE TABLE or VIEW

        if table_name.lower() in exclude:
            continue
        if include and table_name.lower() not in include:
            continue

        # Get columns
        cols_query = f"""
            SELECT column_name, data_type, is_nullable, ordinal_position,
                   column_default
            FROM information_schema.columns
            WHERE table_schema = '{target_schema}'
              AND table_name = '{table_name}'
            ORDER BY ordinal_position
        """
        cols_result = conn.execute(cols_query).fetchall()

        # Get primary key columns via constraints
        pk_cols: set[str] = set()
        try:
            pk_query = f"""
                SELECT column_name
                FROM information_schema.key_column_usage kcu
                JOIN information_schema.table_constraints tc
                  ON kcu.constraint_name = tc.constraint_name
                  AND kcu.table_schema = tc.table_schema
                WHERE tc.constraint_type = 'PRIMARY KEY'
                  AND tc.table_schema = '{target_schema}'
                  AND tc.table_name = '{table_name}'
            """
            pk_result = conn.execute(pk_query).fetchall()
            pk_cols = {r[0] for r in pk_result}
        except Exception:
            pass

        # Get foreign keys
        fk_map: dict[str, str] = {}
        try:
            fk_query = f"""
                SELECT kcu.column_name,
                       ccu.table_schema AS ref_schema,
                       ccu.table_name AS ref_table,
                       ccu.column_name AS ref_column
                FROM information_schema.key_column_usage kcu
                JOIN information_schema.table_constraints tc
                  ON kcu.constraint_name = tc.constraint_name
                JOIN information_schema.constraint_column_usage ccu
                  ON tc.constraint_name = ccu.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY'
                  AND tc.table_schema = '{target_schema}'
                  AND tc.table_name = '{table_name}'
            """
            fk_result = conn.execute(fk_query).fetchall()
            for fk_row in fk_result:
                fk_map[fk_row[0]] = f"{fk_row[1]}.{fk_row[2]}.{fk_row[3]}"
        except Exception:
            pass

        columns = []
        for col_row in cols_result:
            raw = col_row[0]
            columns.append(ColumnInfo(
                name=_norm_col_name(raw),
                original_name=raw,
                type=col_row[1],
                nullable=col_row[2] == "YES",
                primary_key=raw in pk_cols,
                foreign_key=fk_map.get(raw),
                comment=None,  # filled in after sample values are known
            ))

        # Row count
        row_count = None
        try:
            qualified = f'"{target_schema}"."{table_name}"'
            count_result = conn.execute(f"SELECT COUNT(*) FROM {qualified}").fetchone()
            row_count = count_result[0]
        except Exception:
            pass

        # Sample values + date format detection (keys normalized to match col.name)
        sample_values: dict[str, list] = {}
        if sample_rows > 0:
            try:
                qualified = f'"{target_schema}"."{table_name}"'
                sample_result = conn.execute(
                    f"SELECT * FROM {qualified} LIMIT {sample_rows}"
                )
                sample_df = sample_result.fetchdf()
                for col in sample_df.columns:
                    vals = sample_df[col].dropna().tolist()[:5]
                    if vals:
                        sample_values[_norm_col_name(col)] = vals
            except Exception:
                pass

        # Annotate date columns with detected format
        for col in columns:
            if col.comment:
                continue
            is_date_type = any(k in col.type.upper() for k in ("DATE", "TIME", "STAMP"))
            is_date_name = any(k in col.name for k in ("date", "time", "day", "month", "year"))
            if (is_date_type or is_date_name) and col.name in sample_values:
                fmt = _detect_date_format(sample_values[col.name])
                if fmt:
                    col.comment = f"date format: {fmt}"

        comment = f"({table_type.lower()})" if table_type == "VIEW" else None

        tables.append(TableInfo(
            name=table_name,
            schema=target_schema,
            columns=columns,
            row_count=row_count,
            sample_values=sample_values,
            comment=comment,
        ))

    return SchemaContext(tables=tables, dialect="duckdb")


# ─── SQLAlchemy fallback introspection ──────────────────────────────

def introspect_sqlalchemy(
    engine,
    schema: Optional[str] = None,
    sample_rows: int = 3,
    include_tables: Optional[list[str]] = None,
    exclude_tables: Optional[list[str]] = None,
) -> SchemaContext:
    """Introspect any SQLAlchemy-supported database."""
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    exclude = set(exclude_tables or [])
    tables: list[TableInfo] = []

    for table_name in inspector.get_table_names(schema=schema):
        if table_name in exclude:
            continue
        if include_tables and table_name not in include_tables:
            continue

        columns_raw = inspector.get_columns(table_name, schema=schema)
        pk_cols = set(
            inspector.get_pk_constraint(table_name, schema=schema)
            .get("constrained_columns", [])
        )
        fk_map: dict[str, str] = {}
        for fk in inspector.get_foreign_keys(table_name, schema=schema):
            for local, remote in zip(
                fk["constrained_columns"], fk["referred_columns"]
            ):
                fk_map[local] = f"{fk.get('referred_schema', '')}.{fk['referred_table']}.{remote}"

        columns = [
            ColumnInfo(
                name=_norm_col_name(col["name"]),
                original_name=col["name"],
                type=str(col["type"]),
                nullable=col.get("nullable", True),
                primary_key=col["name"] in pk_cols,
                foreign_key=fk_map.get(col["name"]),
            )
            for col in columns_raw
        ]

        # Row count
        row_count = None
        try:
            qualified = f"{schema}.{table_name}" if schema else table_name
            with engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {qualified}"))
                row_count = result.scalar()
        except Exception:
            pass

        # Sample values (keys normalized to match col.name)
        sample_values: dict[str, list] = {}
        if sample_rows > 0:
            try:
                qualified = f"{schema}.{table_name}" if schema else table_name
                with engine.connect() as conn:
                    result = conn.execute(
                        text(f"SELECT * FROM {qualified} LIMIT {sample_rows}")
                    )
                    rows = result.fetchall()
                    col_names = list(result.keys())
                    for i, cn in enumerate(col_names):
                        vals = [row[i] for row in rows if row[i] is not None]
                        if vals:
                            sample_values[_norm_col_name(cn)] = vals[:5]
            except Exception:
                pass

        tables.append(TableInfo(
            name=table_name, schema=schema, columns=columns,
            row_count=row_count, sample_values=sample_values,
        ))

    return SchemaContext(tables=tables, dialect=engine.dialect.name)
