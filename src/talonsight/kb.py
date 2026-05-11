"""SQL pattern knowledge base — RAG retrieval over structured JSON chunks.

Chunks are embedded by: title + intent_tags + when_to_use + anti_patterns.
Retrieval injects pattern templates, hints, and anti-patterns into the LLM prompt.

Built-in patterns live in knowledge_base/*.json (bundled with the package).
Additional patterns can be loaded from any directory via load_dir().

Also provides SchemaIndex: ChromaDB-backed per-table schema retrieval that
narrows the schema prompt to only relevant tables for large databases.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from talonsight.schema import TableInfo

logger = logging.getLogger(__name__)


class _BagOfWordsEF:
    """Offline ChromaDB embedding — no model download required.

    Hashed token counts projected into a fixed-dim vector, L2-normalised.
    Works entirely offline; used as the default and as the fallback when the
    semantic embedding server is unreachable.
    """
    DIM = 512

    def name(self) -> str:
        return "bag-of-words-v2"

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in input]

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self.__call__(input)

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.DIM
        for token in text.lower().split():
            idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.DIM
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class _SemanticEF:
    """Semantic embedding function backed by an HTTP embedding API.

    Supports two API types:
    - ``"ollama"``  — Ollama's  POST /api/embeddings  (one text at a time)
    - ``"openai"``  — Any OpenAI-compatible  POST /v1/embeddings  (batched)

    Safety guarantees
    -----------------
    * Probes the server at construction time.  If unreachable, ``available``
      is set to False and every call transparently falls back to
      ``_BagOfWordsEF`` — no errors surface to the caller.
    * ``name()`` encodes the API type **and** a hash of the model identifier.
      ChromaDB stores this in collection metadata; when it changes (different
      model or different backend) ChromaDB raises a conflict that our
      ``_get_or_create`` handler catches and resolves by rebuilding the
      collection.  This prevents mixing vectors from different models.
    * Batch calls for OpenAI-compatible are done in a single HTTP request;
      Ollama is called sequentially because it doesn't support batching.
    """

    def __init__(self, base_url: str, model: str, api_type: str = "ollama") -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_type = api_type  # "ollama" or "openai"
        self._fallback = _BagOfWordsEF()
        self.available = False

        # Probe: embed a single token to confirm connectivity and get dimension
        try:
            test = self._call_api(["ping"])
            self._dim = len(test[0])
            self.available = True
        except Exception as exc:
            warnings.warn(
                f"Semantic embedding server unreachable ({exc}). "
                "Falling back to bag-of-words embeddings.",
                RuntimeWarning,
                stacklevel=2,
            )

    # ── ChromaDB EF protocol ──────────────────────────────────────────

    def name(self) -> str:
        """Stable identifier that encodes model identity for ChromaDB."""
        model_hash = hashlib.md5(self._model.encode()).hexdigest()[:8]
        return f"semantic-{self._api_type}-{model_hash}"

    def __call__(self, input: list[str]) -> list[list[float]]:
        if not self.available:
            return self._fallback(input)
        try:
            return self._call_api(input)
        except Exception as exc:
            logger.warning("Semantic embedding call failed (%s); using bag-of-words.", exc)
            return self._fallback(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self.__call__(input)

    # ── HTTP calls ────────────────────────────────────────────────────

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        import httpx
        if self._api_type == "ollama":
            return self._call_ollama(texts)
        return self._call_openai(texts)

    def _call_ollama(self, texts: list[str]) -> list[list[float]]:
        """Ollama embeds one text at a time via POST /api/embeddings."""
        import httpx
        vecs: list[list[float]] = []
        with httpx.Client(timeout=30.0) as client:
            for text in texts:
                resp = client.post(
                    f"{self._base_url}/api/embeddings",
                    json={"model": self._model, "prompt": text},
                )
                resp.raise_for_status()
                vecs.append(resp.json()["embedding"])
        return vecs

    def _call_openai(self, texts: list[str]) -> list[list[float]]:
        """OpenAI-compatible batched POST /embeddings."""
        import httpx
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{self._base_url}/embeddings",
                json={"model": self._model, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            # Sort by index to preserve input order
            return [d["embedding"] for d in sorted(data, key=lambda x: x["index"])]


class _FastEmbedEF:
    """In-process semantic embeddings via fastembed (no server required).

    Uses the ``nomic-ai/nomic-embed-text-v1.5`` model by default — a 768-dim
    model (~130 MB) that is downloaded once on first use and cached locally.
    Runs on CPU via ONNX; on Apple Silicon this is accelerated automatically.

    Requires the ``embeddings`` optional extra::

        pip install talonsight[embeddings]

    Falls back to ``_BagOfWordsEF`` with a warning if fastembed is not
    installed, so the rest of the app continues to work.
    """

    DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self._model_name = model or self.DEFAULT_MODEL
        self._model = None          # lazy — loaded on first embed call
        self._fallback = _BagOfWordsEF()
        self.available = False

        try:
            from fastembed import TextEmbedding  # noqa: F401 — just probe import
            self._TextEmbedding = TextEmbedding
            self.available = True
        except ImportError:
            warnings.warn(
                "fastembed is not installed — semantic embeddings unavailable. "
                "Run: pip install talonsight[embeddings]  "
                "Falling back to bag-of-words embeddings.",
                RuntimeWarning,
                stacklevel=2,
            )

    # ── ChromaDB EF protocol ──────────────────────────────────────────

    def name(self) -> str:
        model_hash = hashlib.md5(self._model_name.encode()).hexdigest()[:8]
        return f"fastembed-{model_hash}"

    def __call__(self, input: list[str]) -> list[list[float]]:
        if not self.available:
            return self._fallback(input)
        try:
            m = self._get_model()
            return [e.tolist() for e in m.embed(input)]
        except Exception as exc:
            logger.warning("FastEmbed call failed (%s); using bag-of-words.", exc)
            return self._fallback(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self.__call__(input)

    # ── Internals ─────────────────────────────────────────────────────

    def _get_model(self):
        """Lazy model load — downloads on first call, cached afterwards."""
        if self._model is None:
            self._model = self._TextEmbedding(self._model_name)
        return self._model


def build_embedding_fn(
    backend: str = "bow",
    url: str = "",
    model: str = "",
) -> "_BagOfWordsEF | _FastEmbedEF | _SemanticEF":
    """Return the right ChromaDB embedding function for the given backend.

    Parameters
    ----------
    backend:
        ``"bow"``       — offline bag-of-words (default, no setup needed)
        ``"fastembed"`` — in-process via fastembed (``pip install talonsight[embeddings]``)
        ``"ollama"``    — Ollama embedding API (``ollama pull nomic-embed-text``)
        ``"openai"``    — OpenAI-compatible embedding API (LM Studio, etc.)
    url:
        Base URL for server-backed backends.
        Ollama default : ``http://localhost:11434``
        OpenAI default : ``http://localhost:1234/v1``
    model:
        Embedding model name.
        fastembed default : ``nomic-ai/nomic-embed-text-v1.5``
        Ollama / OpenAI  : ``nomic-embed-text``
    """
    if not backend or backend == "bow":
        return _BagOfWordsEF()

    if backend == "fastembed":
        fe_model = model.strip() or _FastEmbedEF.DEFAULT_MODEL
        return _FastEmbedEF(model=fe_model)

    # Server-backed: ollama or openai
    defaults = {
        "ollama": "http://localhost:11434",
        "openai": "http://localhost:1234/v1",
    }
    resolved_url = url.strip() or defaults.get(backend, "http://localhost:11434")
    resolved_model = model.strip() or "nomic-embed-text"
    return _SemanticEF(base_url=resolved_url, model=resolved_model, api_type=backend)


def _embed_text(chunk: dict) -> str:
    """Build the text to embed for a KB chunk.

    Handles three schema variants:
    - Legacy flat format  — strings / lists-of-strings
    - Enriched domain format — anti_patterns as list[dict], retrieval_metadata,
      metric_nature, sql_assets, inflation/deflation causes, causal chains
    - SQL pattern format  — concept_explanation, mistake_patterns, dialect_traps,
      decision_logic, when_not_to_use, sql_features
    """
    parts = [chunk.get("title", "")]

    tags = chunk.get("intent_tags", [])
    if isinstance(tags, list):
        parts.append(" ".join(str(t) for t in tags))
    elif isinstance(tags, str):
        parts.append(tags)

    # sql_features (SQL JSON schema) — adds "window", "cte", "partition" etc.
    sql_features = chunk.get("sql_features", [])
    if isinstance(sql_features, list) and sql_features:
        parts.append(" ".join(str(f) for f in sql_features))

    when = chunk.get("when_to_use", "")
    if isinstance(when, list):
        parts.append(" ".join(str(w) for w in when))
    elif isinstance(when, str):
        parts.append(when)

    # when_not_to_use (SQL JSON) — negative signal for disambiguation
    when_not = chunk.get("when_not_to_use", [])
    if isinstance(when_not, list) and when_not:
        parts.append(" ".join(str(w) for w in when_not))
    elif isinstance(when_not, str) and when_not:
        parts.append(when_not)

    # anti_patterns: list[str] (legacy) or list[dict] (enriched domain)
    anti = chunk.get("anti_patterns", [])
    if isinstance(anti, list):
        anti_parts = []
        for a in anti:
            if isinstance(a, str):
                anti_parts.append(a)
            elif isinstance(a, dict):
                anti_parts.append(" ".join(filter(None, [
                    a.get("pattern", ""),
                    a.get("anti_pattern", ""),
                    a.get("why_wrong", ""),
                    a.get("why_it_breaks", ""),
                ])))
        if anti_parts:
            parts.append(" ".join(anti_parts))
    elif isinstance(anti, str):
        parts.append(anti)

    # concept_explanation (SQL JSON) — what_it_does anchors semantic meaning
    concept = chunk.get("concept_explanation", {})
    if isinstance(concept, dict):
        what_it_does = concept.get("what_it_does", "")
        if what_it_does:
            parts.append(what_it_does)

    # mistake_patterns (SQL JSON) — mistake labels expand retrieval surface
    mistakes = chunk.get("mistake_patterns", [])
    if isinstance(mistakes, list):
        mistake_parts = [m.get("mistake", "") for m in mistakes if isinstance(m, dict) and m.get("mistake")]
        if mistake_parts:
            parts.append(" ".join(mistake_parts))

    # decision_logic (SQL JSON) — choose_this_when + decision_phrase_triggers
    dl = chunk.get("decision_logic", {})
    if isinstance(dl, dict):
        choose_when = dl.get("choose_this_when", [])
        if isinstance(choose_when, list) and choose_when:
            parts.append(" ".join(str(c) for c in choose_when))
        triggers = dl.get("decision_phrase_triggers", [])
        if isinstance(triggers, list) and triggers:
            parts.append(" ".join(str(t) for t in triggers))

    # retrieval_metadata (domain): synonyms + query_examples widen recall
    rm = chunk.get("retrieval_metadata", {})
    if isinstance(rm, dict):
        synonyms = rm.get("synonyms", [])
        if isinstance(synonyms, list):
            parts.append(" ".join(str(s) for s in synonyms))
        query_examples = rm.get("query_examples", [])
        if isinstance(query_examples, list):
            parts.append(" ".join(str(q) for q in query_examples))

    return " ".join(p for p in parts if p)


class KnowledgeBase:
    """ChromaDB-backed SQL pattern knowledge base."""

    _BUILTIN_DIR = Path(__file__).parent / "knowledge_base"

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        n_results: int = 5,
        ef: Optional["_BagOfWordsEF | _SemanticEF"] = None,
    ):
        self._n_results = n_results
        self._persist_dir = persist_dir or str(Path.home() / ".talonsight" / "kb")
        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)

        import chromadb
        self._client = chromadb.PersistentClient(path=self._persist_dir)
        self._ef = ef if ef is not None else _BagOfWordsEF()
        self._collection = self._get_or_create("talonsight_kb")

        # Always load the built-in patterns
        self._load_builtin()

    def _get_or_create(self, name: str):
        import chromadb
        try:
            return self._client.get_or_create_collection(
                name=name,
                embedding_function=self._ef,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            if "conflict" in str(e).lower() or "embedding function" in str(e).lower():
                self._client.delete_collection(name=name)
                return self._client.create_collection(
                    name=name,
                    embedding_function=self._ef,
                    metadata={"hnsw:space": "cosine"},
                )
            raise

    def _load_builtin(self) -> None:
        if not self._BUILTIN_DIR.exists():
            return
        chunks = []
        for f in sorted(self._BUILTIN_DIR.glob("*.json")):
            try:
                data = json.loads(f.read_text())
                if isinstance(data, list):
                    chunks.extend(data)
                elif isinstance(data, dict):
                    chunks.append(data)
            except Exception:
                pass
        if chunks:
            self._upsert(chunks)

    def load_dir(self, path: str) -> int:
        """Load additional JSON chunks from a directory. Returns count ingested."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"KB directory not found: {path}")
        chunks = []
        for f in sorted(p.glob("**/*.json")):
            try:
                data = json.loads(f.read_text())
                if isinstance(data, list):
                    chunks.extend(data)
                elif isinstance(data, dict):
                    chunks.append(data)
            except Exception:
                pass
        return self._upsert(chunks)

    def load_file(self, path: str) -> int:
        """Load chunks from a single JSON file."""
        data = json.loads(Path(path).read_text())
        chunks = data if isinstance(data, list) else [data]
        return self._upsert(chunks)

    def _upsert(self, chunks: list[dict]) -> int:
        # Deduplicate by ID within the batch: when two chunks share an ID,
        # keep the one with more keys (the enriched SQL-pattern schema always
        # beats the legacy flat schema, so the richer pattern wins).
        id_to_chunk: dict[str, dict] = {}
        for chunk in chunks:
            cid = str(chunk.get("id") or hashlib.sha256(
                json.dumps(chunk, sort_keys=True).encode()
            ).hexdigest()[:16])
            if cid not in id_to_chunk or len(chunk) > len(id_to_chunk[cid]):
                id_to_chunk[cid] = chunk

        ids, documents, metadatas = [], [], []
        for cid, chunk in id_to_chunk.items():
            try:
                # Compute all three values BEFORE any append so a failure in
                # _embed_text (e.g. unexpected types) cannot leave ids/documents/
                # metadatas with unequal lengths.
                doc_text = _embed_text(chunk)
                chunk_json = json.dumps(chunk)
                ids.append(cid)
                documents.append(doc_text)
                metadatas.append({"chunk_json": chunk_json})
            except Exception as exc:
                logger.warning("Skipping KB chunk %r: %s", chunk.get("id", "?"), exc)
        if ids:
            self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        return len(ids)

    def search(self, question: str, n_results: Optional[int] = None) -> list[dict]:
        """Return the top-N most relevant pattern chunks for a question."""
        n = n_results or self._n_results
        if self._collection.count() == 0:
            return []
        actual_n = min(n, self._collection.count())
        results = self._collection.query(query_texts=[question], n_results=actual_n)
        patterns = []
        for meta in results["metadatas"][0]:
            try:
                patterns.append(json.loads(meta["chunk_json"]))
            except Exception:
                pass
        return patterns

    def format_for_prompt(self, patterns: list[dict], dialect: str = "") -> str:
        """Render retrieved patterns as a prompt snippet.

        Handles three schema variants:
        - Legacy flat format  — string template, string anti_patterns
        - Enriched domain format — metric_nature dict, sql_assets, inflation/
          deflation causes with detection SQL, causal_relationships
        - SQL pattern format  — concept_explanation, template dict (minimal/
          realistic/composable), mistake_patterns, dialect_traps, decision_logic,
          edge_cases, performance_notes
        """
        parts = []
        for p in patterns:
            lines = [f"-- Pattern: {p.get('title', '')}"]

            # ── Domain: metric_nature ─────────────────────────────────────
            nature = p.get("metric_nature", "")
            if isinstance(nature, dict):
                what = nature.get("what_it_measures", "")
                what_not = nature.get("what_it_does_not_measure", "")
                misconception = nature.get("common_misconception", "")
                nature_parts = []
                if what:
                    nature_parts.append(f"Measures: {what}")
                if what_not:
                    nature_parts.append(f"Does NOT measure: {what_not}")
                if misconception:
                    nature_parts.append(f"Misconception: {misconception}")
                if nature_parts:
                    lines.append("-- Metric nature:\n" + "\n".join(f"   {n}" for n in nature_parts))
            elif nature:
                lines.append(f"-- Metric nature: {nature}")

            when = p.get("when_to_use", "")
            if when:
                when_str = "; ".join(when) if isinstance(when, list) else when
                lines.append(f"-- Use when: {when_str}")

            # ── SQL pattern: concept_explanation ─────────────────────────
            concept = p.get("concept_explanation", {})
            if isinstance(concept, dict) and concept.get("what_it_does"):
                lines.append(f"-- What it does: {concept['what_it_does']}")
                key_behavior = concept.get("key_behavior", "")
                if key_behavior:
                    lines.append(f"-- Key behavior: {key_behavior}")

            # ── Domain: sql_assets (graduated SQL) ────────────────────────
            sql_assets = p.get("sql_assets", {})
            if isinstance(sql_assets, dict) and sql_assets:
                for level in ("basic", "intermediate", "advanced"):
                    sql = sql_assets.get(level, "")
                    if sql:
                        lines.append(f"-- SQL ({level}):\n{sql}")
            elif isinstance(sql_assets, str) and sql_assets:
                lines.append(f"-- SQL:\n{sql_assets}")

            # ── template field: string (legacy) or dict (SQL pattern) ─────
            template = p.get("template", "")
            if isinstance(template, dict) and template:
                # SQL pattern: prefer minimal → realistic; skip composable (too long)
                minimal = template.get("minimal", "")
                realistic = template.get("realistic", "")
                if minimal:
                    lines.append(f"-- Template (minimal):\n{minimal}")
                if realistic:
                    lines.append(f"-- Template (realistic):\n{realistic}")
            elif isinstance(template, str) and template and not sql_assets:
                lines.append(f"-- Template:\n{template}")

            # ── Domain/Legacy: anti_patterns ─────────────────────────────
            anti = p.get("anti_patterns", [])
            if anti:
                if isinstance(anti, list):
                    anti_strs = []
                    for a in anti:
                        if isinstance(a, str):
                            anti_strs.append(a)
                        elif isinstance(a, dict):
                            label = a.get("pattern") or a.get("anti_pattern", "")
                            why = a.get("why_wrong") or a.get("why_it_breaks", "")
                            entry = f"{label}: {why}" if why else label
                            if entry:
                                anti_strs.append(entry)
                    if anti_strs:
                        lines.append("-- Avoid:\n" + "\n".join(f"   • {s}" for s in anti_strs))
                elif isinstance(anti, str):
                    lines.append(f"-- Avoid: {anti}")

            # ── SQL pattern: mistake_patterns ─────────────────────────────
            mistakes = p.get("mistake_patterns", [])
            if isinstance(mistakes, list) and mistakes:
                mistake_lines = []
                for m in mistakes:
                    if not isinstance(m, dict):
                        continue
                    mistake = m.get("mistake", "")
                    goes_wrong = m.get("what_goes_wrong", "")
                    good_sql = m.get("good_sql", "")
                    if mistake:
                        entry = f"   • {mistake}"
                        if goes_wrong:
                            entry += f": {goes_wrong}"
                        mistake_lines.append(entry)
                        if good_sql:
                            mistake_lines.append(f"     Fix: {good_sql}")
                if mistake_lines:
                    lines.append("-- Common mistakes:\n" + "\n".join(mistake_lines))

            # ── SQL pattern: dialect_traps ────────────────────────────────
            traps = p.get("dialect_traps", [])
            if isinstance(traps, list) and traps:
                trap_lines = []
                dl_key = f"{dialect.lower()}_behavior" if dialect else ""
                for trap in traps:
                    if not isinstance(trap, dict):
                        continue
                    construct = trap.get("construct", "")
                    safe_alt = trap.get("safe_alternative", "")
                    if dialect and dl_key in trap:
                        behavior = trap[dl_key]
                        entry = f"   • [{construct}] {behavior}"
                    else:
                        entry = f"   • [{construct}]" if construct else ""
                    if entry:
                        trap_lines.append(entry)
                    if safe_alt:
                        trap_lines.append(f"     Safe: {safe_alt}")
                if trap_lines:
                    lines.append("-- Dialect traps:\n" + "\n".join(trap_lines))

            # ── SQL pattern: decision_logic ───────────────────────────────
            dl = p.get("decision_logic", {})
            if isinstance(dl, dict):
                choose_when = dl.get("choose_this_when", [])
                if isinstance(choose_when, list) and choose_when:
                    lines.append("-- Choose this when:\n" +
                                 "\n".join(f"   • {c}" for c in choose_when))

            # ── SQL pattern: edge_cases ───────────────────────────────────
            edge_cases = p.get("edge_cases", [])
            if isinstance(edge_cases, list) and edge_cases:
                ec_lines = []
                for ec in edge_cases:
                    if isinstance(ec, dict):
                        scenario = ec.get("scenario", "")
                        handling = ec.get("handling", "")
                        if scenario:
                            entry = f"   • {scenario}"
                            if handling:
                                entry += f" → {handling}"
                            ec_lines.append(entry)
                if ec_lines:
                    lines.append("-- Edge cases:\n" + "\n".join(ec_lines))

            # ── SQL pattern: performance_notes ────────────────────────────
            perf = p.get("performance_notes", {})
            if isinstance(perf, dict):
                hint = perf.get("optimization_hint", "")
                when_slow = perf.get("when_slow", "")
                if hint:
                    lines.append(f"-- Performance: {hint}")
                elif when_slow:
                    lines.append(f"-- Performance: slow when {when_slow}")

            # ── Domain: inflation/deflation causes with detection SQL ──────
            for direction in ("inflation_causes", "deflation_causes"):
                causes = p.get(direction, [])
                if not isinstance(causes, list) or not causes:
                    continue
                label = "Inflation causes" if direction == "inflation_causes" else "Deflation causes"
                cause_lines = []
                for c in causes:
                    if isinstance(c, str):
                        cause_lines.append(f"   • {c}")
                    elif isinstance(c, dict):
                        cause_text = c.get("cause", c.get("signal", ""))
                        det_sql = c.get("detection_sql", "")
                        if cause_text:
                            cause_lines.append(f"   • {cause_text}")
                        if det_sql:
                            cause_lines.append(f"     Detection: {det_sql}")
                if cause_lines:
                    lines.append(f"-- {label}:\n" + "\n".join(cause_lines))

            # ── Domain: causal_relationships ──────────────────────────────
            causal = p.get("causal_relationships", [])
            if isinstance(causal, list) and causal:
                causal_lines = []
                for cr in causal:
                    if isinstance(cr, dict):
                        if_cond = cr.get("if", "")
                        then_res = cr.get("then", "")
                        action = cr.get("action", "")
                        entry = f"   • If {if_cond} → {then_res}"
                        if action:
                            entry += f" (action: {action})"
                        causal_lines.append(entry)
                    elif isinstance(cr, str):
                        causal_lines.append(f"   • {cr}")
                if causal_lines:
                    lines.append("-- Causal relationships:\n" + "\n".join(causal_lines))

            hints = p.get("llm_hints", "")
            if hints:
                lines.append(f"-- Hints: {hints}")

            # Legacy dialect_notes (domain JSONs)
            dialect_notes = p.get("dialect_notes", {})
            if dialect and isinstance(dialect_notes, dict) and dialect in dialect_notes:
                lines.append(f"-- {dialect} note: {dialect_notes[dialect]}")

            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    @property
    def count(self) -> int:
        return self._collection.count()


# ── Schema Index ──────────────────────────────────────────────────────────────

class SchemaIndex:
    """ChromaDB-backed per-table schema retrieval for large databases.

    For schemas with more than THRESHOLD tables, each table is stored as a
    ChromaDB document (name + column names + types).  On every query, only the
    most relevant tables are retrieved instead of dumping the entire schema into
    the prompt.  Tables that are join-connected to retrieved ones are always
    included too, so JOIN paths are never accidentally broken.

    For schemas at or below THRESHOLD tables this class is a transparent
    pass-through — it returns all tables so callers don't need to branch.
    """

    THRESHOLD = 15       # activate only when schema exceeds this many tables
    DEFAULT_N  = 10      # number of tables to retrieve per query

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        ef: Optional["_BagOfWordsEF | _SemanticEF"] = None,
    ):
        self._persist_dir = persist_dir or str(Path.home() / ".talonsight" / "schema_idx")
        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)
        import chromadb
        self._client = chromadb.PersistentClient(path=self._persist_dir)
        self._ef = ef if ef is not None else _BagOfWordsEF()
        self._collection = None
        self._tables: list[TableInfo] = []
        # Pre-computed adjacency: table_name → set of join-connected table names
        self._join_neighbors: dict[str, set[str]] = {}

    # ── Indexing ──────────────────────────────────────────────────────

    def index(self, tables: list[TableInfo], fingerprint: str) -> None:
        """Index all tables for the current connection.

        Re-indexes on every connect so schema changes are picked up
        automatically.  The collection is keyed by a short fingerprint so
        stale collections from previous connections are replaced.
        """
        import chromadb

        self._tables = tables
        # Explicit FK neighbors used for safe join expansion in retrieve()
        self._join_neighbors = self._build_explicit_neighbors(tables)

        if len(tables) <= self.THRESHOLD:
            return  # small schema — no index needed

        coll_name = f"sch{fingerprint[:12]}"  # 15 chars, safe for chroma
        try:
            self._client.delete_collection(coll_name)
        except Exception:
            pass
        self._collection = self._client.create_collection(
            name=coll_name,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

        ids, documents, metadatas = [], [], []
        for i, t in enumerate(tables):
            # Build searchable text: table name (repeated for weight) + all column names/types
            col_text = " ".join(f"{c.name} {c.type}" for c in t.columns)
            doc = f"{t.name} {t.name} {col_text}"
            ids.append(str(i))
            documents.append(doc)
            metadatas.append({"idx": i, "name": t.name})

        self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    # ── Retrieval ─────────────────────────────────────────────────────

    def retrieve(self, question: str, n: Optional[int] = None) -> list[TableInfo]:
        """Return the tables relevant to *question*, expanded with explicit FK neighbors.

        Expansion uses only declared FK relationships — not fuzzy name matching —
        to avoid pulling in unrelated tables through coincidental column name
        overlap.  Total returned tables are capped at 2× the retrieval count.

        Falls back to all tables if the schema is small or the index is absent.
        """
        if not self._tables:
            return []

        total = len(self._tables)
        if total <= self.THRESHOLD or self._collection is None:
            return self._tables  # small schema — use all

        n_fetch = min(n or self.DEFAULT_N, total)
        results = self._collection.query(query_texts=[question], n_results=n_fetch)

        retrieved_names: set[str] = {m["name"] for m in results["metadatas"][0]}

        # Expand with explicit FK neighbors only (one hop, capped)
        max_tables = min(n_fetch * 2, total)
        expanded: set[str] = set(retrieved_names)
        for name in list(retrieved_names):          # iterate initial set only
            for neighbor in self._join_neighbors.get(name, set()):
                if len(expanded) >= max_tables:
                    break
                expanded.add(neighbor)

        # Return in original schema order
        return [t for t in self._tables if t.name in expanded]

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_explicit_neighbors(tables: list[TableInfo]) -> dict[str, set[str]]:
        """Build table-name → {directly FK-joined table names} from declared FKs only.

        Fuzzy name matching is intentionally excluded here — it's used by
        schema.get_join_map() for join-hint generation but is too aggressive
        for retrieval expansion (unrelated tables share column name roots and
        would pull each other in transitively).
        """
        neighbors: dict[str, set[str]] = {t.name: set() for t in tables}
        known = set(neighbors)

        for t in tables:
            for c in t.columns:
                if c.foreign_key:
                    parts = c.foreign_key.split(".")
                    ref = parts[-2] if len(parts) >= 2 else None
                    if ref and ref in known:
                        neighbors[t.name].add(ref)
                        neighbors[ref].add(t.name)

        return neighbors

    @property
    def active(self) -> bool:
        """True if the index is built and retrieval is in effect."""
        return self._collection is not None and len(self._tables) > self.THRESHOLD
