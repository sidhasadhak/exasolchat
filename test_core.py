"""Tests for the ExasolChat core engine using mocked LLM + SQLite."""

import os
import pytest
from exasolchat.core import ExasolChat, QueryResult
from exasolchat.llm import LLMBackend, LLMResponse
from exasolchat.rag import NoopRAGMemory
from exasolchat.safety import RiskLevel


class MockLLM(LLMBackend):
    """Deterministic mock for testing."""

    def __init__(self, sql: str = "SELECT COUNT(*) AS cnt FROM customers"):
        self._sql = sql

    def generate_sql(self, schema_prompt, question, rag_examples=None):
        return LLMResponse(sql=self._sql, explanation="Mock", raw=f"```sql\n{self._sql}\n```")

    def generate_summary(self, question, sql, data_preview):
        return "Mock summary."

    def suggest_chart(self, question, columns, row_count):
        return {"chart_type": "table_only"}


@pytest.fixture
def db_path(tmp_path):
    """Create a temp SQLite DB."""
    import sqlite3
    path = str(tmp_path / "test.db")
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT NOT NULL, country TEXT);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(id), total REAL);
        INSERT INTO customers VALUES (1, 'Alice', 'DE'), (2, 'Bob', 'US');
        INSERT INTO orders VALUES (1, 1, 99.99), (2, 1, 45.0), (3, 2, 200.0);
    """)
    conn.close()
    return path


def _make_chat(db_path, llm=None, **kwargs):
    return ExasolChat(
        connection=f"sqlite:///{db_path}",
        llm=llm or MockLLM(),
        rag=NoopRAGMemory(),
        **kwargs,
    )


def test_ask_returns_data(db_path):
    chat = _make_chat(db_path)
    result = chat.ask("How many customers?")
    assert result.data is not None
    assert len(result.data) > 0
    assert result.error is None
    assert result.safety.is_allowed
    chat.close()


def test_blocks_dangerous_sql(db_path):
    chat = _make_chat(db_path, llm=MockLLM(sql="DROP TABLE customers"))
    result = chat.ask("Delete everything")
    assert result.safety.level == RiskLevel.BLOCKED
    assert result.error is not None
    chat.close()


def test_blocks_injection(db_path):
    chat = _make_chat(db_path, llm=MockLLM(sql="SELECT 1; DELETE FROM customers"))
    result = chat.ask("Hack it")
    assert result.safety.level == RiskLevel.BLOCKED
    chat.close()


def test_schema_introspected(db_path):
    chat = _make_chat(db_path)
    names = {t.name for t in chat.schema_context.tables}
    assert "customers" in names
    assert "orders" in names
    chat.close()


def test_history(db_path):
    chat = _make_chat(db_path)
    chat.ask("Q1")
    chat.ask("Q2")
    assert len(chat.history) == 2
    chat.close()


def test_extra_context(db_path):
    chat = _make_chat(db_path, extra_context="country uses ISO 3166")
    assert "ISO 3166" in chat.schema_prompt
    chat.close()


def test_add_context(db_path):
    chat = _make_chat(db_path)
    chat.add_context("total is in EUR")
    assert "EUR" in chat.schema_prompt
    chat.close()


def test_allowed_tables_enforcement(db_path):
    chat = _make_chat(
        db_path,
        llm=MockLLM(sql="SELECT * FROM orders"),
        allowed_tables=["customers"],
    )
    result = chat.ask("Show orders")
    assert result.safety.level == RiskLevel.BLOCKED
    chat.close()


def test_context_manager(db_path):
    with _make_chat(db_path) as chat:
        result = chat.ask("Count")
        assert result.data is not None


def test_manual_train(db_path):
    from exasolchat.rag import RAGMemory
    rag = NoopRAGMemory()
    chat = _make_chat(db_path, rag_enabled=False)
    # Should not raise even with noop
    chat.train("test question", "SELECT 1")
    chat.close()


# ── DuckDB-specific tests ───────────────────────────────────────────

class TestDuckDBConnection:
    """Test DuckDB native connection path."""

    @pytest.fixture
    def duck_path(self, tmp_path):
        import duckdb
        path = str(tmp_path / "test.duckdb")
        conn = duckdb.connect(path)
        conn.execute("""
            CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                name VARCHAR NOT NULL,
                price DOUBLE,
                category VARCHAR
            )
        """)
        conn.execute("""
            INSERT INTO products VALUES
            (1, 'Widget', 9.99, 'tools'),
            (2, 'Gadget', 24.99, 'electronics'),
            (3, 'Doohickey', 4.99, 'tools')
        """)
        conn.close()
        return path

    def _make_duck_chat(self, duck_path, llm=None, **kwargs):
        from exasolchat.connection import ConnectionConfig
        config = ConnectionConfig.duckdb(path=duck_path)
        return ExasolChat(
            connection=config,
            llm=llm or MockLLM(sql="SELECT COUNT(*) AS cnt FROM products"),
            rag=NoopRAGMemory(),
            **kwargs,
        )

    def test_duckdb_connect_and_query(self, duck_path):
        chat = self._make_duck_chat(duck_path)
        result = chat.ask("How many products?")
        assert result.data is not None
        assert result.error is None
        assert result.safety.is_allowed
        chat.close()

    def test_duckdb_schema_introspected(self, duck_path):
        chat = self._make_duck_chat(duck_path)
        names = {t.name for t in chat.schema_context.tables}
        assert "products" in names
        chat.close()

    def test_duckdb_dialect_detected(self, duck_path):
        chat = self._make_duck_chat(duck_path)
        assert chat.schema_context.dialect == "duckdb"
        chat.close()

    def test_duckdb_columns_introspected(self, duck_path):
        chat = self._make_duck_chat(duck_path)
        products = next(t for t in chat.schema_context.tables if t.name == "products")
        col_names = {c.name for c in products.columns}
        assert col_names == {"id", "name", "price", "category"}
        chat.close()

    def test_duckdb_sample_values(self, duck_path):
        chat = self._make_duck_chat(duck_path)
        products = next(t for t in chat.schema_context.tables if t.name == "products")
        assert "name" in products.sample_values
        assert "Widget" in products.sample_values["name"]
        chat.close()

    def test_duckdb_blocks_read_csv(self, duck_path):
        evil_llm = MockLLM(sql="SELECT * FROM read_csv('/etc/passwd')")
        chat = self._make_duck_chat(duck_path, llm=evil_llm)
        result = chat.ask("Read a file")
        assert result.safety.level == RiskLevel.BLOCKED
        chat.close()

    def test_duckdb_blocks_attach(self, duck_path):
        evil_llm = MockLLM(sql="ATTACH '/tmp/other.db' AS other")
        chat = self._make_duck_chat(duck_path, llm=evil_llm)
        result = chat.ask("Attach a db")
        assert result.safety.level == RiskLevel.BLOCKED
        chat.close()

    def test_duckdb_url_parsing(self):
        from exasolchat.connection import ConnectionConfig
        config = ConnectionConfig.from_url("duckdb:///tmp/data.duckdb")
        assert config.is_duckdb
        assert config.duckdb_path == "/tmp/data.duckdb"

    def test_duckdb_memory_url(self):
        from exasolchat.connection import ConnectionConfig
        config = ConnectionConfig.from_url("duckdb://:memory:")
        assert config.is_duckdb
        assert config.duckdb_path == ":memory:"
