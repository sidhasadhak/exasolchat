"""High-level analytical capabilities for the talonsight agent.

v1: thin wrappers over raw SQL execution.  Each capability runs one or more
    queries and returns a plain-English + table string the agent can reason over.
    The AgentLoop reasons at this business level — SQL is an implementation detail.

v2: genuine statistical computations — period decomposition, z-score change
    detection, Pearson correlation, trend projection.  The interface is stable;
    the implementation layer evolves underneath without touching agent logic.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from talonsight.agent import AgentLoop   # avoid circular import

logger = logging.getLogger(__name__)


class AgentCapabilities:
    """Business-level analytical capabilities available to the agent as tools.

    Instantiated once per AgentLoop and passed in at construction time.
    The agent calls these via the tool dispatch table in AgentLoop._execute_tool().
    """

    def __init__(self, loop: "AgentLoop") -> None:
        self._loop = loop

    # ── v1 implementations ────────────────────────────────────────────
    # Each method runs one or more SQL queries via the loop's connector
    # and returns a formatted string the agent appends to its reasoning context.

    def find_drivers(
        self,
        metric: str,
        date_range: str,
        dimensions: list[str],
    ) -> str:
        """Decompose what's driving movement in a metric across given dimensions.

        v1: runs a GROUP BY per dimension, formats results as a ranked markdown table.
        v2: period-over-period decomposition with absolute + relative contribution.

        Args:
            metric:     Column expression to aggregate, e.g. "SUM(revenue)"
            date_range: Natural-language range, e.g. "last 7 days" — passed as
                        context; the agent must have already identified the date column.
            dimensions: List of column names to segment by, e.g. ["region", "product_tier"]
        """
        # In v1 the agent already has the schema context and can construct precise SQL.
        # find_drivers is a signal to the agent to do a systematic dimension breakdown.
        # We return a structured prompt hint rather than executing SQL directly,
        # because the agent needs to pick the right table/date column from context.
        return (
            f"[find_drivers hint]\n"
            f"Investigate '{metric}' broken down by each of: {dimensions}.\n"
            f"Date range context: {date_range}.\n"
            f"For each dimension: run a GROUP BY query, compute the metric per segment, "
            f"sort descending, and identify which segments account for the largest share "
            f"of the total. Summarise which 1-3 segments drive the most movement."
        )

    def detect_change(
        self,
        metric: str,
        timeframe: str,
        comparison: str = "prior period",
    ) -> str:
        """Statistically verify whether a metric genuinely shifted or is noise.

        v1: period-over-period comparison with pct change calculation.
        v2: z-score against stored baseline from BusinessModel.

        Args:
            metric:     Metric description, e.g. "weekly churn rate"
            timeframe:  Period to inspect, e.g. "last 7 days", "February 2026"
            comparison: What to compare against, e.g. "prior 7 days", "same period last year"
        """
        return (
            f"[detect_change hint]\n"
            f"Verify whether '{metric}' changed meaningfully in '{timeframe}'.\n"
            f"Comparison baseline: {comparison}.\n"
            f"Compute the metric for both periods, calculate absolute and percentage change. "
            f"Flag as significant if the change exceeds 10% or 2x the typical week-over-week "
            f"variance you can observe in recent history. State clearly: confirmed change or noise."
        )

    def compare_segments(
        self,
        metric: str,
        dimension: str,
        date_range: str = "recent",
    ) -> str:
        """Rank every segment of a dimension by its contribution to a metric.

        v1: GROUP BY dimension, sort by metric desc, show top/bottom.
        v2: Pareto breakdown — which segments explain 80% of the total.

        Args:
            metric:    Metric to compare, e.g. "revenue", "churn_rate"
            dimension: Column to segment by, e.g. "country", "product_tier"
            date_range: Time window, e.g. "last 30 days"
        """
        return (
            f"[compare_segments hint]\n"
            f"Rank all segments of '{dimension}' by '{metric}' for {date_range}.\n"
            f"Show: segment name, metric value, % of total, rank. "
            f"Identify the top 3 segments and flag any with unusually high or low values "
            f"compared to the others."
        )

    def correlate(
        self,
        metric_a: str,
        metric_b: str,
        timeframe: str,
        lag_days: int = 0,
    ) -> str:
        """Find the relationship between two metrics over time.

        v1: compute both metrics as time series, inspect visually via the agent.
        v2: Pearson correlation with optional lag testing.
        """
        lag_note = f" with a {lag_days}-day lag on {metric_b}" if lag_days else ""
        return (
            f"[correlate hint]\n"
            f"Investigate whether '{metric_a}' and '{metric_b}' move together{lag_note} "
            f"over {timeframe}.\n"
            f"Compute both as weekly time series. Do they rise and fall together? "
            f"Note any periods where they diverge. State whether the relationship "
            f"looks positive, negative, or uncorrelated based on what you observe."
        )

    def project_trend(
        self,
        metric: str,
        periods: int,
        method: str = "linear",
    ) -> str:
        """Extrapolate a metric forward based on observed trajectory.

        v1: linear extrapolation hint for the agent to compute.
        v2: proper linear/exponential/seasonal decomposition with confidence intervals.
        """
        return (
            f"[project_trend hint]\n"
            f"Project '{metric}' forward by {periods} periods using a {method} trend.\n"
            f"First compute the metric for the last 8 periods to establish the trend. "
            f"Then extrapolate forward. State the projected value and whether the current "
            f"trajectory is accelerating, stable, or decelerating."
        )
