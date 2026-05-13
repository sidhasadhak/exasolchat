"""Schema intelligence graph — relationship detection, table classification,
business domain identification.

Runs once on connect from the already-introspected SchemaContext.
Zero extra DB queries in v1; v2 adds optional cardinality sampling.

The graph feeds two consumers:
  1. Agent system prompt — compact natural-language context injected into
     every question so the LLM understands the data model from step 0.
  2. Streamlit UI — structured dict for the Schema Intelligence panel.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from talonsight.schema import SchemaContext


# ── Domain keyword signatures ─────────────────────────────────────────────────

_DOMAINS: dict[str, set[str]] = {
    "e-commerce":  {"order", "product", "customer", "cart", "payment", "shipping",
                    "inventory", "seller", "review", "basket", "checkout", "refund",
                    "item", "freight", "delivery"},
    "saas":        {"user", "subscription", "event", "session", "plan", "feature",
                    "billing", "tenant", "workspace", "audit", "usage", "trial"},
    "finance":     {"account", "transaction", "journal", "ledger", "invoice",
                    "payment", "budget", "expense", "asset", "liability", "entry"},
    "hr":          {"employee", "department", "position", "payroll", "attendance",
                    "leave", "recruitment", "candidate", "performance", "salary"},
    "healthcare":  {"patient", "appointment", "diagnosis", "medication", "provider",
                    "claim", "prescription", "visit", "procedure", "clinical"},
    "logistics":   {"shipment", "route", "vehicle", "warehouse", "carrier",
                    "manifest", "tracking", "delivery", "freight", "dispatch"},
    "analytics":   {"event", "pageview", "session", "funnel", "cohort",
                    "metric", "dimension", "aggregate", "impression", "click"},
}

# ── Column name patterns ──────────────────────────────────────────────────────

_RE_MEASURE = re.compile(
    r"(price|amount|cost|revenue|total|value|salary|wage|fee|"
    r"quantity|qty|count|volume|weight|size|duration|score|rate|"
    r"balance|profit|loss|margin|discount|tax|freight|spend|budget)",
    re.I,
)
_RE_DATE = re.compile(
    r"(date|_at|timestamp|created|updated|deleted|purchased|shipped|"
    r"delivered|expires|started|ended|approved|closed|opened)",
    re.I,
)
_RE_FK = re.compile(r"_id$|_fk$|_key$|_ref$|_code$", re.I)
_RE_PK = re.compile(r"^id$|_id$", re.I)
_RE_TEXT = re.compile(r"(char|text|string|varchar|enum|name|title|desc)", re.I)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Relationship:
    """A detected join path between two tables."""
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    rel_type: str = "N:1"       # N:1 | 1:1 | M:N
    confidence: float = 1.0     # 1.0 = declared FK; <1.0 = inferred
    source: str = "declared"    # declared | name_match


@dataclass
class TableNode:
    """Semantic metadata for a single table."""
    name: str
    fqn: str                        # exact SQL reference: "schema"."table"
    table_type: str = "unknown"     # fact | dimension | bridge | lookup | unknown
    row_count: Optional[int] = None
    pk_candidates: list[str] = field(default_factory=list)
    fk_columns: list[str] = field(default_factory=list)
    measures: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    date_columns: list[str] = field(default_factory=list)

    @property
    def type_emoji(self) -> str:
        return {
            "fact": "📊", "dimension": "📋",
            "bridge": "🔗", "lookup": "📖", "unknown": "❓",
        }.get(self.table_type, "❓")

    @property
    def type_label(self) -> str:
        return self.table_type.title()


# ── Main class ────────────────────────────────────────────────────────────────

class SchemaGraph:
    """Semantic graph of a connected database schema.

    Build once on connect — call .build() — then read .to_agent_context()
    for prompt injection and .to_dict() for the UI.
    """

    def __init__(self, schema_context: "SchemaContext") -> None:
        self._ctx = schema_context
        self.nodes: dict[str, TableNode] = {}     # lower(name) → node
        self.relationships: list[Relationship] = []
        self.domain: str = "unknown"
        self.domain_confidence: float = 0.0
        self.summary: str = ""
        self._built = False

    # ── Entry point ───────────────────────────────────────────────────────────

    def build(self) -> "SchemaGraph":
        """Analyse the schema. Safe to call multiple times (no-op after first)."""
        if self._built:
            return self
        try:
            self._build_nodes()
            self._detect_relationships()
            self._classify_tables()
            self._detect_domain()
            self._build_summary()
        except Exception:
            pass  # graph build failure must never block a connection
        self._built = True
        return self

    # ── Build nodes ───────────────────────────────────────────────────────────

    def _build_nodes(self) -> None:
        for tbl in self._ctx.tables:
            fqn = (
                f'"{tbl.schema}"."{tbl.name}"' if tbl.schema
                else f'"{tbl.name}"'
            )
            node = TableNode(name=tbl.name, fqn=fqn, row_count=tbl.row_count)
            for col in tbl.columns:
                cn = col.name.lower()
                ct = (col.type or "").lower()
                if _RE_FK.search(cn):
                    node.fk_columns.append(col.name)
                if _RE_PK.search(cn):
                    node.pk_candidates.append(col.name)
                if _RE_MEASURE.search(cn):
                    node.measures.append(col.name)
                if _RE_DATE.search(cn):
                    node.date_columns.append(col.name)
                if _RE_TEXT.search(ct) or _RE_TEXT.search(cn):
                    node.categorical_cols.append(col.name)
            self.nodes[tbl.name.lower()] = node

    # ── Relationship detection ────────────────────────────────────────────────

    def _detect_relationships(self) -> None:
        """Infer FK relationships from column naming conventions (N:1)."""
        seen: set[tuple] = set()

        for tbl in self._ctx.tables:
            for col in tbl.columns:
                cn = col.name.lower()
                if not _RE_FK.search(cn):
                    continue

                # Strip FK suffix → candidate target table base name
                base = re.sub(r"_id$|_fk$|_key$|_ref$|_code$", "", cn, flags=re.I)
                if not base or base == tbl.name.lower():
                    continue

                # Try singular, plural, and stripped-s forms
                for candidate in _name_variants(base):
                    target_tbl = next(
                        (t for t in self._ctx.tables if t.name.lower() == candidate),
                        None,
                    )
                    if target_tbl is None:
                        continue

                    # Find best target column: exact match, then any PK-like
                    target_col = next(
                        (c.name for c in target_tbl.columns
                         if c.name.lower() in ("id", cn)),
                        None,
                    ) or next(
                        (c.name for c in target_tbl.columns
                         if _RE_PK.search(c.name.lower())),
                        None,
                    ) or (target_tbl.columns[0].name if target_tbl.columns else None)

                    if target_col is None:
                        continue

                    key = (
                        tbl.name.lower(), col.name.lower(),
                        target_tbl.name.lower(), target_col.lower(),
                    )
                    if key in seen:
                        break
                    seen.add(key)

                    self.relationships.append(Relationship(
                        from_table=tbl.name,
                        from_column=col.name,
                        to_table=target_tbl.name,
                        to_column=target_col,
                        rel_type="N:1",
                        confidence=0.9,
                        source="name_match",
                    ))
                    break  # one target per FK column

    # ── Table classification ──────────────────────────────────────────────────

    def _classify_tables(self) -> None:
        if not self.nodes:
            return

        # Tables that are the *target* of relationships are dimension candidates
        referenced: set[str] = {r.to_table.lower() for r in self.relationships}

        row_counts = sorted(
            n.row_count for n in self.nodes.values()
            if n.row_count and n.row_count > 0
        )
        median_rows = row_counts[len(row_counts) // 2] if row_counts else 0

        for name, node in self.nodes.items():
            tbl_info = next(
                (t for t in self._ctx.tables if t.name.lower() == name), None
            )
            n_cols = len(tbl_info.columns) if tbl_info else 1
            n_fk = len(node.fk_columns)
            fk_ratio = n_fk / n_cols if n_cols else 0
            rows = node.row_count or 0

            if fk_ratio >= 0.5 and n_cols <= 6 and n_fk >= 2:
                # Mostly FK columns, few others → bridge / junction table
                node.table_type = "bridge"
            elif rows > 0 and rows < 200 and n_cols <= 5:
                # Tiny, few columns → lookup / code table
                node.table_type = "lookup"
            elif (
                n_fk >= 2
                and len(node.measures) >= 1
                and (rows == 0 or rows >= median_rows * 0.3)
            ) or (
                len(node.date_columns) >= 1
                and n_fk >= 1
                and rows >= median_rows * 0.5
            ):
                # Multiple FK refs + measurable columns + high cardinality → fact
                node.table_type = "fact"
            elif name in referenced or n_fk == 0:
                # Pointed to by others, or has no outgoing FKs → dimension
                node.table_type = "dimension"
            else:
                node.table_type = "unknown"

    # ── Domain detection ──────────────────────────────────────────────────────

    def _detect_domain(self) -> None:
        tokens: set[str] = set()
        for tbl in self._ctx.tables:
            tokens.update(re.split(r"[_\s]+", tbl.name.lower()))
            for col in tbl.columns:
                tokens.update(re.split(r"[_\s]+", col.name.lower()))

        scores: dict[str, int] = {}
        for domain, keywords in _DOMAINS.items():
            score = len(keywords & tokens)
            if score:
                scores[domain] = score

        if scores:
            best = max(scores, key=scores.__getitem__)
            self.domain = best
            self.domain_confidence = round(
                scores[best] / sum(scores.values()), 2
            )
        else:
            self.domain = "general"
            self.domain_confidence = 0.0

    # ── Summary ───────────────────────────────────────────────────────────────

    def _build_summary(self) -> None:
        counts = {t: 0 for t in ("fact", "dimension", "bridge", "lookup", "unknown")}
        for node in self.nodes.values():
            counts[node.table_type] += 1

        parts = []
        if counts["fact"]:
            parts.append(f"{counts['fact']} fact")
        if counts["dimension"]:
            parts.append(f"{counts['dimension']} dimension")
        if counts["bridge"]:
            parts.append(f"{counts['bridge']} bridge")
        if counts["lookup"]:
            parts.append(f"{counts['lookup']} lookup")

        domain_str = (
            self.domain.replace("-", " ").title()
            if self.domain not in ("unknown", "general") else ""
        )
        table_breakdown = ", ".join(parts) if parts else f"{len(self.nodes)} tables"
        rels_str = (
            f"{len(self.relationships)} join path"
            + ("s" if len(self.relationships) != 1 else "")
        )
        prefix = f"{domain_str} schema — " if domain_str else "Schema — "
        self.summary = f"{prefix}{table_breakdown}, {rels_str} detected."

    # ── Output interfaces ─────────────────────────────────────────────────────

    def to_agent_context(self) -> str:
        """Compact block injected into every agent system prompt.

        Gives the LLM instant awareness of:
          - What the business domain is
          - Which tables are facts vs dimensions
          - Which columns are measures vs dates vs FKs
          - All detected join paths
        """
        if not self._built:
            self.build()

        lines = [f"SCHEMA INTELLIGENCE — {self.summary}"]

        # Table roles
        lines.append("\nTable roles:")
        for node in self.nodes.values():
            extras = []
            if node.measures:
                extras.append(f"measures: {', '.join(node.measures[:4])}")
            if node.date_columns:
                extras.append(f"dates: {', '.join(node.date_columns[:2])}")
            extra_str = f"  [{'; '.join(extras)}]" if extras else ""
            lines.append(
                f"  {node.type_emoji} {node.fqn}  [{node.table_type}]{extra_str}"
            )

        # Join paths
        if self.relationships:
            lines.append("\nJoin paths (use these in JOINs):")
            for rel in self.relationships:
                conf = (
                    f"  ~{int(rel.confidence*100)}% confidence"
                    if rel.confidence < 1.0 else ""
                )
                lines.append(
                    f"  {rel.from_table}.{rel.from_column}"
                    f" → {rel.to_table}.{rel.to_column}"
                    f" ({rel.rel_type}){conf}"
                )
        else:
            lines.append("\nNo join paths detected — tables may be independent.")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialisable representation for Streamlit UI rendering."""
        if not self._built:
            self.build()
        return {
            "domain": self.domain,
            "domain_confidence": self.domain_confidence,
            "summary": self.summary,
            "nodes": [
                {
                    "name": n.name,
                    "fqn": n.fqn,
                    "type": n.table_type,
                    "emoji": n.type_emoji,
                    "row_count": n.row_count,
                    "measures": n.measures,
                    "date_columns": n.date_columns,
                    "fk_columns": n.fk_columns,
                    "categorical_cols": n.categorical_cols,
                }
                for n in self.nodes.values()
            ],
            "relationships": [
                {
                    "from_table": r.from_table,
                    "from_column": r.from_column,
                    "to_table": r.to_table,
                    "to_column": r.to_column,
                    "type": r.rel_type,
                    "confidence": r.confidence,
                    "source": r.source,
                }
                for r in self.relationships
            ],
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _name_variants(base: str) -> list[str]:
    """Return plausible table name forms for a stripped FK base."""
    variants = [base]
    if not base.endswith("s"):
        variants.append(base + "s")
    if base.endswith("s"):
        variants.append(base[:-1])
    if base.endswith("ies"):
        variants.append(base[:-3] + "y")
    return variants
