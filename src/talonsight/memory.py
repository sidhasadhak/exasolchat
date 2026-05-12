"""Persistent business memory for talonsight agent investigations.

v1: write confirmed findings after each final_answer; inject the last N
    findings as context into the agent system prompt so investigations
    build on each other across sessions.

v2: full metric baselines, open investigations, business event calendar,
    rejected-analysis store, and similarity-based context retrieval.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Root directory for all talonsight user data
_TALONSIGHT_DIR = Path(os.environ.get("TALONSIGHT_HOME", Path.home() / ".talonsight"))
_MEMORY_DIR = _TALONSIGHT_DIR / "memory"


@dataclass
class Finding:
    """A confirmed analytical finding from a completed agent investigation."""
    question: str
    narrative: str
    sql: str
    tables_used: list[str]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    confirmed: bool = True


class BusinessModel:
    """Persistent knowledge about your business, built from every investigation.

    One file per database connection stored at:
        ~/.talonsight/memory/{connection_id}.json

    The agent reads recent findings at the start of every investigation so it
    starts from a rich context rather than from nothing. The longer you use it,
    the smarter it gets about YOUR specific business.

    v1 scope
    --------
    - record_finding()   — called by final_answer tool
    - get_context()      — injects last N findings into the system prompt

    v2 additions (interfaces stubbed, not yet implemented)
    ------
    - metric_baselines   — what "normal" looks like for each KPI
    - open_investigations — questions still being tracked
    - business_events    — pricing changes, outages, campaigns
    - rejected_analyses  — bad SQL + feedback for avoidance
    """

    def __init__(self, connection_id: str) -> None:
        _MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        # Sanitise connection_id to a safe filename
        safe_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in connection_id)
        self._path = _MEMORY_DIR / f"{safe_id}.json"
        self._data = self._load()

    # ── Persistence ───────────────────────────────────────────────────

    def _load(self) -> dict:
        if self._path.exists():
            try:
                with self._path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:
                logger.warning("Could not load business model from %s: %s", self._path, exc)
        return {
            "findings": [],
            # v2 keys — present but empty so the file format is stable
            "metric_baselines": {},
            "open_investigations": [],
            "business_events": [],
            "rejected_analyses": [],
        }

    def _save(self) -> None:
        try:
            with self._path.open("w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.warning("Could not save business model to %s: %s", self._path, exc)

    # ── v1 API ────────────────────────────────────────────────────────

    def record_finding(self, finding: Finding) -> None:
        """Append a confirmed finding. Called by the final_answer tool."""
        self._data["findings"].append(asdict(finding))
        self._save()
        logger.debug("Business model: recorded finding for %r", finding.question[:60])

    def get_context(self, max_findings: int = 5) -> str:
        """Return the last N confirmed findings as a text block for the system prompt.

        v1: plain-text, most-recent-first.
        v2: similarity-based retrieval against the current question.
        """
        findings = self._data.get("findings", [])
        if not findings:
            return ""

        recent = findings[-max_findings:][::-1]  # most recent first
        lines = []
        for f in recent:
            lines.append(
                f"- Q: {f['question']}\n"
                f"  Finding: {f['narrative']}"
            )
        return "\n".join(lines)

    def finding_count(self) -> int:
        return len(self._data.get("findings", []))

    # ── v2 stubs (interfaces defined, not yet implemented) ────────────

    def set_baseline(self, metric: str, mean: float, std: float, unit: str = "") -> None:
        """v2: store a metric baseline for change-detection."""
        self._data["metric_baselines"][metric] = {
            "mean": mean, "std": std, "unit": unit,
            "as_of": datetime.now(timezone.utc).isoformat(),
        }
        self._save()

    def add_business_event(self, date: str, description: str) -> None:
        """v2: record a significant business event (pricing change, outage, campaign)."""
        self._data["business_events"].append({"date": date, "description": description})
        self._save()

    def get_baselines(self) -> dict:
        """v2: return all metric baselines (empty dict in v1)."""
        return self._data.get("metric_baselines", {})
