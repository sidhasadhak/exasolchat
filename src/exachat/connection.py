"""Database connection management.

Handles pyexasol (Exasol-native), duckdb (native), and SQLAlchemy (everything else).
Provides a unified query execution interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

import pandas as pd


@dataclass
class ConnectionConfig:
    """Database connection configuration."""
    # Exasol-specific
    dsn: Optional[str] = None          # e.g., "exasoldb:8563"
    user: Optional[str] = None
    password: Optional[str] = None
    exasol_schema: Optional[str] = None

    # DuckDB-specific
    duckdb_path: Optional[str] = None  # path to .duckdb file, or ":memory:"

    # SQLAlchemy (fallback for non-Exasol, non-DuckDB)
    sqlalchemy_url: Optional[str] = None

    @property
    def is_exasol(self) -> bool:
        return self.dsn is not None

    @property
    def is_duckdb(self) -> bool:
        return self.duckdb_path is not None

    @classmethod
    def exasol(
        cls, dsn: str, user: str, password: str, schema: Optional[str] = None,
    ) -> "ConnectionConfig":
        return cls(dsn=dsn, user=user, password=password, exasol_schema=schema)

    @classmethod
    def duckdb(cls, path: str = ":memory:") -> "ConnectionConfig":
        return cls(duckdb_path=path)

    @classmethod
    def from_url(cls, url: str) -> "ConnectionConfig":
        """Parse a connection URL. Auto-detects Exasol, DuckDB, or other."""
        if url.startswith("exa+"):
            from urllib.parse import urlparse
            parsed = urlparse(url.replace("exa+pyexasol://", "http://"))
            dsn = f"{parsed.hostname}:{parsed.port or 8563}"
            schema = parsed.path.strip("/") or None
            return cls(
                dsn=dsn,
                user=parsed.username,
                password=parsed.password,
                exasol_schema=schema,
            )
        if url.startswith("duckdb://"):
            # duckdb:///path/to/file.duckdb or duckdb://:memory:
            path = url.replace("duckdb://", "", 1)
            path = path.lstrip("/") or ":memory:"
            if path != ":memory:" and not path.startswith("/"):
                path = "/" + path  # restore absolute path
            return cls(duckdb_path=path)
        if url.endswith(".duckdb") or url.endswith(".db") and "://" not in url:
            # Bare file path — assume DuckDB
            return cls(duckdb_path=url)
        return cls(sqlalchemy_url=url)


class DatabaseConnection:
    """Unified database interface for Exasol, DuckDB, and SQLAlchemy."""

    def __init__(self, config: ConnectionConfig):
        self.config = config
        self._pyexasol_conn = None
        self._duckdb_conn = None
        self._sqla_engine = None

    def connect(self):
        """Establish connection."""
        if self.config.is_exasol:
            import pyexasol
            self._pyexasol_conn = pyexasol.connect(
                dsn=self.config.dsn,
                user=self.config.user,
                password=self.config.password,
                schema=self.config.exasol_schema,
                compression=True,
            )
        elif self.config.is_duckdb:
            import duckdb
            self._duckdb_conn = duckdb.connect(
                database=self.config.duckdb_path,
                read_only=self.config.duckdb_path != ":memory:",
            )
        else:
            from sqlalchemy import create_engine
            self._sqla_engine = create_engine(self.config.sqlalchemy_url)

    @property
    def is_exasol(self) -> bool:
        return self._pyexasol_conn is not None

    @property
    def is_duckdb(self) -> bool:
        return self._duckdb_conn is not None

    @property
    def dialect(self) -> str:
        if self.is_exasol:
            return "exasol"
        if self.is_duckdb:
            return "duckdb"
        return self._sqla_engine.dialect.name

    @property
    def pyexasol_conn(self):
        """Raw pyexasol connection (for introspection)."""
        return self._pyexasol_conn

    @property
    def duckdb_conn(self):
        """Raw duckdb connection (for introspection)."""
        return self._duckdb_conn

    @property
    def sqla_engine(self):
        """Raw SQLAlchemy engine (for introspection)."""
        return self._sqla_engine

    def execute_query(self, sql: str, max_rows: int = 5000) -> pd.DataFrame:
        """Execute a SELECT query and return results as DataFrame."""
        if self.is_exasol:
            return self._execute_exasol(sql, max_rows)
        if self.is_duckdb:
            return self._execute_duckdb(sql, max_rows)
        return self._execute_sqlalchemy(sql, max_rows)

    def _execute_exasol(self, sql: str, max_rows: int) -> pd.DataFrame:
        """Execute via pyexasol."""
        stmt = self._pyexasol_conn.execute(sql)
        cols = [c[0] for c in stmt.description()]
        rows = stmt.fetchmany(max_rows)
        return pd.DataFrame(rows, columns=cols)

    def _execute_duckdb(self, sql: str, max_rows: int) -> pd.DataFrame:
        """Execute via native duckdb — returns DataFrame directly."""
        result = self._duckdb_conn.execute(sql)
        df = result.fetchdf()
        if len(df) > max_rows:
            df = df.head(max_rows)
        return df

    def _execute_sqlalchemy(self, sql: str, max_rows: int) -> pd.DataFrame:
        """Execute via SQLAlchemy."""
        from sqlalchemy import text
        with self._sqla_engine.connect() as conn:
            try:
                conn.execute(text("SET TRANSACTION READ ONLY"))
            except Exception:
                pass
            result = conn.execute(text(sql))
            rows = result.fetchmany(max_rows)
            columns = list(result.keys())
            return pd.DataFrame(rows, columns=columns)

    def close(self):
        """Close connection."""
        if self._pyexasol_conn:
            try:
                self._pyexasol_conn.close()
            except Exception:
                pass
        if self._duckdb_conn:
            try:
                self._duckdb_conn.close()
            except Exception:
                pass
        if self._sqla_engine:
            self._sqla_engine.dispose()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()
