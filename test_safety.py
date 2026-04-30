"""Tests for SQL safety validation — the most critical module."""

import pytest
from exasolchat.safety import RiskLevel, validate_sql, sanitize_sql


class TestSafeQueries:
    """These MUST be allowed through."""

    def test_simple_select(self):
        assert validate_sql("SELECT * FROM users").is_allowed

    def test_select_where(self):
        assert validate_sql("SELECT name FROM users WHERE id = 5").is_allowed

    def test_join(self):
        assert validate_sql(
            "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id"
        ).is_allowed

    def test_aggregation(self):
        assert validate_sql(
            "SELECT dept, COUNT(*) cnt FROM employees GROUP BY dept HAVING COUNT(*) > 5"
        ).is_allowed

    def test_cte(self):
        assert validate_sql(
            "WITH top AS (SELECT * FROM users LIMIT 10) SELECT * FROM top"
        ).is_allowed

    def test_subquery(self):
        assert validate_sql(
            "SELECT * FROM users WHERE id IN (SELECT user_id FROM orders)"
        ).is_allowed

    def test_window_function(self):
        assert validate_sql(
            "SELECT name, ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) rn FROM emp"
        ).is_allowed

    def test_trailing_semicolon(self):
        assert validate_sql("SELECT 1;").is_allowed

    def test_case_expression(self):
        assert validate_sql(
            "SELECT CASE WHEN x > 0 THEN 'pos' ELSE 'neg' END FROM t"
        ).is_allowed

    def test_nested_subqueries(self):
        assert validate_sql(
            "SELECT * FROM (SELECT * FROM (SELECT 1 AS x) a) b"
        ).is_allowed


class TestBlockedQueries:
    """These MUST be blocked. No exceptions."""

    def test_drop(self):
        assert validate_sql("DROP TABLE users").level == RiskLevel.BLOCKED

    def test_delete(self):
        assert validate_sql("DELETE FROM users WHERE 1=1").level == RiskLevel.BLOCKED

    def test_insert(self):
        assert validate_sql("INSERT INTO users (name) VALUES ('x')").level == RiskLevel.BLOCKED

    def test_update(self):
        assert validate_sql("UPDATE users SET admin = true").level == RiskLevel.BLOCKED

    def test_create(self):
        assert validate_sql("CREATE TABLE evil (id int)").level == RiskLevel.BLOCKED

    def test_alter(self):
        assert validate_sql("ALTER TABLE users ADD col text").level == RiskLevel.BLOCKED

    def test_truncate(self):
        assert validate_sql("TRUNCATE TABLE users").level == RiskLevel.BLOCKED

    def test_statement_stacking(self):
        assert validate_sql("SELECT 1; DROP TABLE users").level == RiskLevel.BLOCKED

    def test_statement_stacking_insert(self):
        assert validate_sql("SELECT 1; INSERT INTO t VALUES (1)").level == RiskLevel.BLOCKED

    def test_exec(self):
        assert validate_sql("EXEC sp_danger").level == RiskLevel.BLOCKED

    def test_execute(self):
        assert validate_sql("EXECUTE sp_danger").level == RiskLevel.BLOCKED

    def test_call(self):
        assert validate_sql("CALL dangerous_proc()").level == RiskLevel.BLOCKED

    def test_pg_sleep(self):
        assert validate_sql("SELECT pg_sleep(10)").level == RiskLevel.BLOCKED

    def test_waitfor(self):
        assert validate_sql("SELECT 1 WAITFOR DELAY '0:0:10'").level == RiskLevel.BLOCKED

    def test_benchmark(self):
        assert validate_sql("SELECT BENCHMARK(9999999, SHA1('x'))").level == RiskLevel.BLOCKED

    def test_into_outfile(self):
        assert validate_sql("SELECT * FROM t INTO OUTFILE '/tmp/x'").level == RiskLevel.BLOCKED

    def test_load_data(self):
        assert validate_sql("LOAD DATA INFILE '/etc/passwd' INTO TABLE t").level == RiskLevel.BLOCKED

    def test_grant(self):
        assert validate_sql("GRANT ALL ON t TO evil").level == RiskLevel.BLOCKED

    def test_export(self):
        assert validate_sql("EXPORT TABLE t INTO CSV AT '/tmp'").level == RiskLevel.BLOCKED

    def test_import(self):
        assert validate_sql("IMPORT INTO t FROM CSV AT '/tmp'").level == RiskLevel.BLOCKED

    def test_exasol_script_create(self):
        assert validate_sql(
            "CREATE PYTHON SCRIPT my_script() AS print('pwned')"
        ).level == RiskLevel.BLOCKED

    def test_set_command(self):
        assert validate_sql("SET SOME_VAR = 'value'").level == RiskLevel.BLOCKED

    def test_empty(self):
        assert validate_sql("").level == RiskLevel.BLOCKED

    def test_whitespace(self):
        assert validate_sql("   ").level == RiskLevel.BLOCKED

    def test_starts_with_update(self):
        assert validate_sql("UPDATE users SET x = 1").level == RiskLevel.BLOCKED

    def test_merge(self):
        assert validate_sql("MERGE INTO t USING s ON t.id = s.id").level == RiskLevel.BLOCKED


class TestSuspiciousQueries:
    """These should be flagged but still allowed (user sees warning)."""

    def test_union_select(self):
        v = validate_sql("SELECT 1 UNION SELECT password FROM users")
        assert v.level == RiskLevel.SUSPICIOUS

    def test_or_1_equals_1(self):
        v = validate_sql("SELECT * FROM users WHERE name = '' OR 1=1")
        assert v.level == RiskLevel.SUSPICIOUS

    def test_string_tautology(self):
        v = validate_sql("SELECT * FROM users WHERE x = '' OR 'a'='a'")
        assert v.level == RiskLevel.SUSPICIOUS

    def test_exa_system_table(self):
        v = validate_sql("SELECT * FROM EXA_ALL_USERS")
        assert v.level == RiskLevel.SUSPICIOUS


class TestSchemaAccessControl:
    """Schema/table allowlist enforcement."""

    def test_allowed_schema_passes(self):
        v = validate_sql(
            "SELECT * FROM SALES.CUSTOMERS",
            allowed_schemas={"SALES"},
        )
        assert v.is_allowed

    def test_blocked_schema(self):
        v = validate_sql(
            "SELECT * FROM INTERNAL.SECRETS",
            allowed_schemas={"SALES", "ANALYTICS"},
        )
        assert v.level == RiskLevel.BLOCKED
        assert "INTERNAL" in v.reason

    def test_allowed_table_passes(self):
        v = validate_sql(
            "SELECT * FROM CUSTOMERS",
            allowed_tables={"CUSTOMERS", "ORDERS"},
        )
        assert v.is_allowed

    def test_blocked_table(self):
        v = validate_sql(
            "SELECT * FROM PASSWORDS",
            allowed_tables={"CUSTOMERS", "ORDERS"},
        )
        assert v.level == RiskLevel.BLOCKED

    def test_case_insensitive(self):
        v = validate_sql(
            "SELECT * FROM sales.customers",
            allowed_schemas={"SALES"},
        )
        assert v.is_allowed


class TestSanitize:
    def test_strips_semicolons(self):
        assert sanitize_sql("SELECT 1;") == "SELECT 1"

    def test_collapses_whitespace(self):
        assert sanitize_sql("SELECT  1\n  FROM\n  t") == "SELECT 1 FROM t"


class TestDuckDBSpecificBlocked:
    """DuckDB-specific dangerous operations must be blocked."""

    def test_copy(self):
        assert validate_sql("COPY t TO '/tmp/out.csv'").level == RiskLevel.BLOCKED

    def test_attach(self):
        assert validate_sql("ATTACH '/tmp/other.db' AS other").level == RiskLevel.BLOCKED

    def test_detach(self):
        assert validate_sql("DETACH other").level == RiskLevel.BLOCKED

    def test_install_extension(self):
        assert validate_sql("INSTALL httpfs").level == RiskLevel.BLOCKED

    def test_load_extension(self):
        assert validate_sql("LOAD httpfs").level == RiskLevel.BLOCKED

    def test_pragma(self):
        assert validate_sql("PRAGMA database_list").level == RiskLevel.BLOCKED

    def test_checkpoint(self):
        assert validate_sql("CHECKPOINT").level == RiskLevel.BLOCKED

    def test_vacuum(self):
        assert validate_sql("VACUUM").level == RiskLevel.BLOCKED

    def test_read_csv_in_select(self):
        v = validate_sql("SELECT * FROM read_csv('/etc/passwd')")
        assert v.level == RiskLevel.BLOCKED

    def test_read_parquet_in_select(self):
        v = validate_sql("SELECT * FROM read_parquet('s3://bucket/data.parquet')")
        assert v.level == RiskLevel.BLOCKED

    def test_read_json_in_select(self):
        v = validate_sql("SELECT * FROM read_json('/tmp/data.json')")
        assert v.level == RiskLevel.BLOCKED

    def test_read_csv_auto(self):
        v = validate_sql("SELECT * FROM read_csv_auto('/tmp/data.csv')")
        assert v.level == RiskLevel.BLOCKED

    def test_statement_stacking_with_copy(self):
        v = validate_sql("SELECT 1; COPY t TO '/tmp/x.csv'")
        assert v.level == RiskLevel.BLOCKED

    def test_statement_stacking_with_attach(self):
        v = validate_sql("SELECT 1; ATTACH '/tmp/x.db'")
        assert v.level == RiskLevel.BLOCKED


class TestDuckDBSpecificSuspicious:
    """DuckDB internal functions should be flagged."""

    def test_duckdb_internal_function(self):
        v = validate_sql("SELECT duckdb_settings()")
        assert v.level == RiskLevel.SUSPICIOUS

    def test_pg_catalog(self):
        v = validate_sql("SELECT * FROM pg_catalog.pg_tables")
        assert v.level == RiskLevel.SUSPICIOUS
