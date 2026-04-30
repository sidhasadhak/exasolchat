"""Database schema introspection.

Primary: pyexasol for ExasolDB (richer metadata, faster).
Fallback: SQLAlchemy for any other database.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ColumnInfo:
    name: str
    type: str
    nullable: bool
    primary_key: bool
    foreign_key: Optional[str] = None
    comment: Optional[str] = None


@dataclass
class TableInfo:
    name: str
    schema: Optional[str]
    columns: list[ColumnInfo] = field(default_factory=list)
    row_count: Optional[int] = None
    sample_values: dict[str, list] = field(default_factory=dict)
    comment: Optional[str] = None


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

        if self.extra_context:
            lines.append("ADDITIONAL CONTEXT:")
            lines.append(self.extra_context)
        return "\n".join(lines)

    @property
    def table_names(self) -> list[str]:
        return [t.name for t in self.tables]


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
            columns.append(ColumnInfo(
                name=col_row[0],
                type=col_row[1],
                nullable=col_row[2],
                primary_key=col_row[0] in pk_cols,
                foreign_key=fk_map.get(col_row[0]),
                comment=col_row[3],
            ))

        # Sample values
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
                        sample_values[cn] = vals[:5]
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
            columns.append(ColumnInfo(
                name=col_row[0],
                type=col_row[1],
                nullable=col_row[2] == "YES",
                primary_key=col_row[0] in pk_cols,
                foreign_key=fk_map.get(col_row[0]),
            ))

        # Row count
        row_count = None
        try:
            qualified = f'"{target_schema}"."{table_name}"'
            count_result = conn.execute(f"SELECT COUNT(*) FROM {qualified}").fetchone()
            row_count = count_result[0]
        except Exception:
            pass

        # Sample values
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
                        sample_values[col] = vals
            except Exception:
                pass

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
                name=col["name"],
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

        # Sample values
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
                            sample_values[cn] = vals[:5]
            except Exception:
                pass

        tables.append(TableInfo(
            name=table_name, schema=schema, columns=columns,
            row_count=row_count, sample_values=sample_values,
        ))

    return SchemaContext(tables=tables, dialect=engine.dialect.name)
