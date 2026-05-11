"""Metrics catalog — user-defined business metrics with SQL expressions.

Metrics are stored as JSON files in ~/.talonsight/metrics/ (or a custom path).
They inject into the LLM prompt so "what is revenue?" uses the finance-approved
formula, and they appear as calculated fields in the visual query builder.

Example metric:
{
    "name": "revenue",
    "description": "Total net revenue excluding refunds",
    "sql": "SUM(\"order_amount\") - SUM(\"refunds\")",
    "dimensions": ["date", "country"],
    "filters": ["exclude_test_users = true"],
    "caveats": "Finance-approved metric",
    "tables": ["orders", "refunds"]
}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


METRIC_TEMPLATE = {
    "name": "",
    "description": "",
    "sql": "",
    "dimensions": [],
    "filters": [],
    "caveats": "",
    "tables": [],
}


class MetricsCatalog:
    """Persistent catalog of named business metrics."""

    def __init__(self, metrics_path: Optional[str] = None):
        self._dir = Path(metrics_path or Path.home() / ".talonsight" / "metrics")
        self._dir.mkdir(parents=True, exist_ok=True)
        self._metrics: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        self._metrics.clear()
        for f in sorted(self._dir.glob("*.json")):
            try:
                data = json.loads(f.read_text())
                entries = data if isinstance(data, list) else [data]
                for m in entries:
                    if m.get("name"):
                        self._metrics[m["name"]] = m
            except Exception:
                pass

    def add(self, metric: dict) -> None:
        """Add or update a metric. Persists to disk immediately."""
        name = metric.get("name", "").strip()
        if not name:
            raise ValueError("Metric must have a non-empty 'name'.")
        if not metric.get("sql", "").strip():
            raise ValueError("Metric must have a non-empty 'sql' expression.")
        self._metrics[name] = metric
        (self._dir / f"{name}.json").write_text(json.dumps(metric, indent=2))

    def remove(self, name: str) -> None:
        """Delete a metric from memory and disk."""
        self._metrics.pop(name, None)
        f = self._dir / f"{name}.json"
        if f.exists():
            f.unlink()

    def get(self, name: str) -> Optional[dict]:
        return self._metrics.get(name)

    def all(self) -> list[dict]:
        return list(self._metrics.values())

    def format_for_prompt(self) -> str:
        """Render metrics as a prompt block for the LLM."""
        if not self._metrics:
            return ""
        lines = [
            "AVAILABLE METRICS (use these exact SQL expressions — do not invent your own formulas):"
        ]
        for m in self._metrics.values():
            line = f"- {m['name']}: {m['sql']}"
            if m.get("description"):
                line += f"  — {m['description']}"
            if m.get("caveats"):
                line += f"  [Caveat: {m['caveats']}]"
            if m.get("dimensions"):
                line += f"  [Valid dimensions: {', '.join(m['dimensions'])}]"
            lines.append(line)
        return "\n".join(lines)

    @property
    def count(self) -> int:
        return len(self._metrics)

    def reload(self) -> None:
        """Re-read all metrics from disk (e.g. after external edits)."""
        self._load()
