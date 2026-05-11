"""TalonSight — safe text-to-SQL for ExasolDB and DuckDB with local LLMs."""

from talonsight.core import TalonSight, QueryResult
from talonsight.connection import ConnectionConfig
from talonsight.builder import QueryBuilder
from talonsight.metrics import MetricsCatalog
from talonsight.safety import RiskLevel, SafetyVerdict, validate_sql
from talonsight.schema import get_join_map

__all__ = [
    "TalonSight",
    "QueryResult",
    "ConnectionConfig",
    "QueryBuilder",
    "MetricsCatalog",
    "RiskLevel",
    "SafetyVerdict",
    "validate_sql",
]

# Backward-compat alias — existing code using ExasolChat still works
ExasolChat = TalonSight
