"""Persistent business memory for talonsight.

Every investigation leaves behind structured knowledge that makes the next
investigation smarter.  Three layers compound over time:

  Findings     — confirmed Q&A pairs with SQL, tables, and key numbers
  KPIs         — metric definitions discovered through exploration
  Domain Facts — confirmed facts about the data itself (date ranges, segment sizes, …)

Context injection
-----------------
get_full_context(question) scores all stored knowledge against the incoming
question using token overlap and returns the most relevant slice — so the agent
doesn't just get "last 5 findings" but the most contextually useful ones.

Storage
-------
One JSON file per database connection at ~/.talonsight/memory/{id}.json.
The schema is additive — new keys are ignored by older code.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_TALONSIGHT_DIR = Path(os.environ.get("TALONSIGHT_HOME", Path.home() / ".talonsight"))
_MEMORY_DIR = _TALONSIGHT_DIR / "memory"

_MAX_FINDINGS = 200   # hard cap before pruning oldest
_MAX_FACTS    = 100
_MAX_KPIS     = 50


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Finding:
    """A confirmed analytical answer from a completed agent investigation."""
    question: str
    narrative: str
    sql: str
    tables_used: list[str]
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    confirmed: bool = True
    # Enrichment fields — populated automatically where possible
    key_numbers: dict[str, float] = field(default_factory=dict)  # {"revenue": 1e6}
    tags: list[str] = field(default_factory=list)                # ["freight","SP"]


@dataclass
class KPI:
    """A business metric definition discovered through exploration."""
    name: str                        # "Monthly Revenue"
    sql_expression: str              # "SUM(payment_value)"
    table: str                       # "order_payments"
    unit: str = ""                   # "$", "%", "orders", …
    description: str = ""
    last_value: Optional[float] = None
    last_checked: Optional[str] = None
    discovered_from: str = ""        # the question that surfaced this KPI
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class DomainFact:
    """A confirmed fact about the data — ranges, segment sizes, quirks."""
    fact: str                        # "Orders span 2016-09 → 2018-10"
    category: str = "general"        # "date_range" | "segment" | "quality" | "general"
    confidence: str = "confirmed"    # "confirmed" | "inferred"
    source_question: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ── Business Model ────────────────────────────────────────────────────────────

class BusinessModel:
    """Persistent, compounding knowledge about a connected database.

    Stored at: ~/.talonsight/memory/{connection_id}.json
    One instance per TalonSight session; survives restarts.

    The longer it runs, the richer the context it injects into every
    agent investigation — findings become baselines, KPIs become reusable,
    domain facts become assumptions the agent can rely on.
    """

    def __init__(self, connection_id: str) -> None:
        _MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        safe_id = "".join(
            c if c.isalnum() or c in "-_." else "_"
            for c in connection_id
        )[:120]
        self._path = _MEMORY_DIR / f"{safe_id}.json"
        self._data = self._load()
        self._purge_invalid_findings()  # clean up any corrupt entries from old runs

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> dict:
        defaults: dict = {
            "findings": [],
            "kpis": {},               # name (lower) → KPI dict
            "domain_facts": [],
            "metric_baselines": {},   # kept for v1 compat
            "business_events": [],
        }
        if self._path.exists():
            try:
                stored = json.loads(self._path.read_text(encoding="utf-8"))
                # Merge stored data with defaults so any keys added in new
                # versions of the schema are always present (forward-compat).
                return {**defaults, **stored}
            except Exception as exc:
                logger.warning("Could not load business model %s: %s", self._path, exc)
        return defaults

    def _save(self) -> None:
        try:
            self._path.write_text(
                json.dumps(self._data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Could not save business model %s: %s", self._path, exc)

    # ── Write API ─────────────────────────────────────────────────────────────

    def _purge_invalid_findings(self) -> None:
        """Remove findings whose narrative looks like a JSON blob or agent error.
        Called once on load to clean up corrupt entries from old runs.
        """
        before = len(self._data.get("findings", []))
        self._data["findings"] = [
            f for f in self._data.get("findings", [])
            if _is_valid_narrative(f.get("narrative", ""))
        ]
        if len(self._data["findings"]) < before:
            self._save()
            logger.info(
                "BusinessModel: purged %d invalid finding(s)",
                before - len(self._data["findings"]),
            )

    def record_finding(self, finding: Finding) -> None:
        """Append a confirmed finding after a successful final_answer."""
        if not _is_valid_narrative(finding.narrative):
            logger.debug("BusinessModel: skipping invalid finding narrative")
            return
        self._data["findings"].append(asdict(finding))
        # Prune oldest if over cap
        if len(self._data["findings"]) > _MAX_FINDINGS:
            self._data["findings"] = self._data["findings"][-_MAX_FINDINGS:]
        self._save()
        logger.debug("BusinessModel: finding recorded — %r", finding.question[:60])

    def record_kpi(self, kpi: KPI) -> None:
        """Store or update a KPI definition."""
        key = kpi.name.lower().strip()
        self._data["kpis"][key] = asdict(kpi)
        if len(self._data["kpis"]) > _MAX_KPIS:
            # Drop least-recently updated
            by_age = sorted(
                self._data["kpis"].items(),
                key=lambda x: x[1].get("timestamp", ""),
            )
            self._data["kpis"] = dict(by_age[-_MAX_KPIS:])
        self._save()
        logger.debug("BusinessModel: KPI recorded — %r", kpi.name)

    def record_domain_fact(self, fact: DomainFact) -> None:
        """Store a confirmed fact about the data."""
        # Deduplicate by fact text (case-insensitive)
        existing = {f["fact"].lower() for f in self._data["domain_facts"]}
        if fact.fact.lower() not in existing:
            self._data["domain_facts"].append(asdict(fact))
            if len(self._data["domain_facts"]) > _MAX_FACTS:
                self._data["domain_facts"] = self._data["domain_facts"][-_MAX_FACTS:]
            self._save()

    # ── Read API — context injection ──────────────────────────────────────────

    def get_full_context(self, question: str = "", max_findings: int = 6) -> str:
        """Return the richest relevant context for injecting into the system prompt.

        Sections returned (only non-empty ones included):
          1. Domain facts — confirmed facts about the data
          2. KPI definitions — reusable metric SQL
          3. Relevant past findings — scored by relevance to the current question
        """
        parts: list[str] = []

        facts_block = self._format_domain_facts()
        if facts_block:
            parts.append(facts_block)

        kpis_block = self._format_kpis()
        if kpis_block:
            parts.append(kpis_block)

        findings_block = self._format_relevant_findings(question, max_findings)
        if findings_block:
            parts.append(findings_block)

        return "\n\n".join(parts)

    def get_context(self, max_findings: int = 5) -> str:
        """Legacy interface — returns recent findings as plain text.
        Prefer get_full_context() for richer context.
        """
        return self._format_relevant_findings("", max_findings)

    # ── Private formatting ────────────────────────────────────────────────────

    def _format_domain_facts(self) -> str:
        facts = self._data.get("domain_facts", [])
        if not facts:
            return ""
        lines = ["CONFIRMED DATA FACTS:"]
        for f in facts[-20:]:  # last 20, they accumulate slowly
            conf = " (inferred)" if f.get("confidence") == "inferred" else ""
            lines.append(f"  • {f['fact']}{conf}")
        return "\n".join(lines)

    def _format_kpis(self) -> str:
        kpis = self._data.get("kpis", {})
        if not kpis:
            return ""
        lines = ["KNOWN KPI DEFINITIONS (reuse these SQL patterns):"]
        for kpi in list(kpis.values())[-15:]:
            unit = f" [{kpi['unit']}]" if kpi.get("unit") else ""
            val = (
                f" — last value: {kpi['last_value']:,.2f}{unit}"
                if kpi.get("last_value") is not None else ""
            )
            lines.append(
                f"  • {kpi['name']}: {kpi['sql_expression']}"
                f" (from {kpi['table']}){val}"
            )
        return "\n".join(lines)

    def _format_relevant_findings(self, question: str, max_n: int) -> str:
        findings = self._data.get("findings", [])
        if not findings:
            return ""

        if question:
            scored = [
                (f, _relevance_score(question, f["question"] + " " + f["narrative"]))
                for f in findings
            ]
            # Mix: top relevant + most recent, deduplicated
            by_relevance = sorted(scored, key=lambda x: x[1], reverse=True)
            by_recency   = list(reversed(findings))
            seen: set[int] = set()
            merged: list[dict] = []
            for f, _ in by_relevance[:max_n]:
                idx = id(f)
                if idx not in seen:
                    seen.add(idx)
                    merged.append(f)
            for f in by_recency:
                if len(merged) >= max_n:
                    break
                idx = id(f)
                if idx not in seen:
                    seen.add(idx)
                    merged.append(f)
            selected = merged[:max_n]
        else:
            selected = list(reversed(findings))[:max_n]

        if not selected:
            return ""

        lines = ["PAST INVESTIGATION FINDINGS (use as trusted baselines):"]
        for f in selected:
            ts = f.get("timestamp", "")[:10]  # date only
            lines.append(f"  [{ts}] Q: {f['question']}")
            lines.append(f"         → {f['narrative']}")
        return "\n".join(lines)

    # ── Stats for UI ──────────────────────────────────────────────────────────

    def finding_count(self) -> int:
        return len(self._data.get("findings", []))

    def kpi_count(self) -> int:
        return len(self._data.get("kpis", {}))

    def fact_count(self) -> int:
        return len(self._data.get("domain_facts", []))

    def get_kpis(self) -> list[dict]:
        return list(self._data.get("kpis", {}).values())

    def get_recent_findings(self, n: int = 10) -> list[dict]:
        findings = self._data.get("findings", [])
        return list(reversed(findings))[:n]

    def get_domain_facts(self) -> list[dict]:
        return self._data.get("domain_facts", [])

    def summary(self) -> str:
        """One-line knowledge summary for the UI."""
        fc = self.finding_count()
        kc = self.kpi_count()
        dc = self.fact_count()
        parts = []
        if fc:
            parts.append(f"{fc} finding{'s' if fc != 1 else ''}")
        if kc:
            parts.append(f"{kc} KPI{'s' if kc != 1 else ''}")
        if dc:
            parts.append(f"{dc} fact{'s' if dc != 1 else ''}")
        return ", ".join(parts) if parts else "No knowledge yet"

    # ── v1 compat stubs ───────────────────────────────────────────────────────

    def set_baseline(self, metric: str, mean: float, std: float, unit: str = "") -> None:
        self._data["metric_baselines"][metric] = {
            "mean": mean, "std": std, "unit": unit,
            "as_of": datetime.now(timezone.utc).isoformat(),
        }
        self._save()

    def add_business_event(self, date: str, description: str) -> None:
        self._data["business_events"].append({"date": date, "description": description})
        self._save()

    def get_baselines(self) -> dict:
        return self._data.get("metric_baselines", {})


# ── KPI auto-extraction from SQL ──────────────────────────────────────────────

_AGG_RE = re.compile(
    r"\b(SUM|COUNT|AVG|MIN|MAX|MEDIAN|PERCENTILE_CONT)\s*\(([^)]{1,80})\)",
    re.IGNORECASE,
)
_AS_RE = re.compile(r"\bAS\s+(['\"`]?)(\w+)\1", re.IGNORECASE)

def extract_kpis_from_sql(sql: str, source_question: str = "") -> list[KPI]:
    """Best-effort extraction of KPI definitions from a SQL query.

    Finds aggregate expressions (SUM, COUNT, AVG, …) and their aliases,
    then pairs them with the FROM table. Returns KPI objects ready for
    record_kpi().
    """
    if not sql:
        return []

    kpis: list[KPI] = []
    # Find the primary FROM table
    from_match = re.search(r'\bFROM\s+"?(\w+)"?\."?(\w+)"?|\bFROM\s+"?(\w+)"?',
                           sql, re.IGNORECASE)
    table = ""
    if from_match:
        table = from_match.group(2) or from_match.group(3) or ""

    # Walk aggregate expressions
    aliases = {m.start(): m.group(2) for m in _AS_RE.finditer(sql)}

    for m in _AGG_RE.finditer(sql):
        func  = m.group(1).upper()
        inner = m.group(2).strip()
        expr  = f"{func}({inner})"

        # Find nearest alias after this expression
        alias = next(
            (v for pos, v in sorted(aliases.items()) if pos > m.end()), ""
        )

        name = (
            alias.replace("_", " ").title()
            if alias
            else f"{func.title()} of {inner.strip('\"').replace('_', ' ').title()}"
        )

        kpis.append(KPI(
            name=name,
            sql_expression=expr,
            table=table,
            discovered_from=source_question,
        ))

    return kpis[:5]  # cap at 5 per query — avoid noise from complex CTEs


# ── Domain fact extraction ────────────────────────────────────────────────────

def extract_domain_facts(narrative: str, sql: str,
                         source_question: str = "") -> list[DomainFact]:
    """Extract confirmable facts from a narrative + SQL result.

    Looks for patterns like date ranges, record counts, percentage splits.
    Returns DomainFact objects ready for record_domain_fact().
    """
    facts: list[DomainFact] = []
    text = narrative

    # Date range mentions: "2016 to 2018", "from Jan 2017"
    date_range = re.search(
        r"(20\d\d[-/]\d\d(?:[-/]\d\d)?\s*(?:to|through|→|-)\s*20\d\d[-/]\d\d(?:[-/]\d\d)?)",
        text, re.IGNORECASE,
    )
    if date_range:
        facts.append(DomainFact(
            fact=f"Date range in data: {date_range.group(1)}",
            category="date_range",
            source_question=source_question,
        ))

    # Large number mentions: "99,441 customers", "112,650 orders"
    for m in re.finditer(r"([\d,]+)\s+(customer|order|product|seller|transaction)s?", text, re.I):
        num = m.group(1).replace(",", "")
        entity = m.group(2).lower() + "s"
        if not num or not num.isdigit():
            continue
        if int(num) > 100:
            facts.append(DomainFact(
                fact=f"Approximately {m.group(1)} {entity} in the dataset",
                category="segment",
                source_question=source_question,
            ))

    # Percentage findings: "accounts for 35% of revenue"
    for m in re.finditer(
        r"(\w[\w\s]{2,30})\s+accounts?\s+for\s+([\d.]+%)\s+of\s+([\w\s]{2,30})",
        text, re.I,
    ):
        facts.append(DomainFact(
            fact=f"{m.group(1).strip()} accounts for {m.group(2)} of {m.group(3).strip()}",
            category="segment",
            confidence="confirmed",
            source_question=source_question,
        ))

    return facts[:3]  # cap — narratives can be noisy


# ── Relevance scoring ─────────────────────────────────────────────────────────

_STOP = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to",
    "of", "and", "or", "for", "with", "from", "by", "be", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should", "may",
    "can", "what", "how", "why", "when", "where", "which", "show", "me",
    "tell", "get", "give", "find", "list", "all", "any", "some",
})

def _tokenise(text: str) -> set[str]:
    return {
        w for w in re.split(r"\W+", text.lower())
        if len(w) > 2 and w not in _STOP
    }

def _is_valid_narrative(text: str) -> bool:
    """Return False if the narrative looks like a JSON blob, agent error, or
    other non-human-readable content that should never be stored as a finding."""
    if not text or len(text.strip()) < 20:
        return False
    t = text.strip()
    # Starts with JSON object/array
    if t.startswith(("{", "[", "```")):
        return False
    # Contains agent internal markers
    bad_markers = (
        '"plan":', '"steps":', '"tool_call"', '"function"',
        "TOOL ERROR", "SQL ERROR", "BLOCKED:", "Investigation reached the step limit",
        "internal error",
    )
    tl = t.lower()
    if any(m.lower() in tl for m in bad_markers):
        return False
    # Looks like raw JSON key-value pairs
    if re.search(r'"\w+":\s*[\[{"\d]', t):
        return False
    return True


def _relevance_score(query: str, candidate: str) -> float:
    """Simple token-overlap relevance — no external deps."""
    q = _tokenise(query)
    c = _tokenise(candidate)
    if not q or not c:
        return 0.0
    overlap = len(q & c)
    return overlap / (len(q) + 1)
