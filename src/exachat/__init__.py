"""ExasolChat — safe text-to-SQL for ExasolDB and DuckDB with local LLMs."""

from exachat.core import ExasolChat, QueryResult
from exachat.connection import ConnectionConfig
from exachat.safety import RiskLevel, SafetyVerdict, validate_sql

__all__ = [
    "ExasolChat",
    "QueryResult",
    "ConnectionConfig",
    "RiskLevel",
    "SafetyVerdict",
    "validate_sql",
]
