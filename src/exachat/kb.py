"""SQL pattern knowledge base — RAG retrieval over structured JSON chunks.

Chunks are embedded by: title + intent_tags + when_to_use + anti_patterns.
Retrieval injects pattern templates, hints, and anti-patterns into the LLM prompt.

Built-in patterns live in knowledge_base/*.json (bundled with the package).
Additional patterns can be loaded from any directory via load_dir().
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Optional


class _BagOfWordsEF:
    """Offline ChromaDB embedding — no model download required.

    Hashed token counts projected into a fixed-dim vector, L2-normalised.
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


def _embed_text(chunk: dict) -> str:
    """Build the text to embed for a KB chunk."""
    parts = [chunk.get("title", "")]

    tags = chunk.get("intent_tags", [])
    if isinstance(tags, list):
        parts.append(" ".join(tags))
    elif isinstance(tags, str):
        parts.append(tags)

    when = chunk.get("when_to_use", "")
    if isinstance(when, list):
        parts.append(" ".join(when))
    elif isinstance(when, str):
        parts.append(when)

    anti = chunk.get("anti_patterns", [])
    if isinstance(anti, list):
        parts.append(" ".join(anti))
    elif isinstance(anti, str):
        parts.append(anti)

    return " ".join(p for p in parts if p)


class KnowledgeBase:
    """ChromaDB-backed SQL pattern knowledge base."""

    _BUILTIN_DIR = Path(__file__).parent / "knowledge_base"

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        n_results: int = 3,
    ):
        self._n_results = n_results
        self._persist_dir = persist_dir or str(Path.home() / ".exachat" / "kb")
        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)

        import chromadb
        self._client = chromadb.PersistentClient(path=self._persist_dir)
        self._ef = _BagOfWordsEF()
        self._collection = self._get_or_create("exachat_kb")

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
        ids, documents, metadatas = [], [], []
        for chunk in chunks:
            try:
                doc_id = str(chunk.get("id") or hashlib.sha256(
                    json.dumps(chunk, sort_keys=True).encode()
                ).hexdigest()[:16])
                ids.append(doc_id)
                documents.append(_embed_text(chunk))
                metadatas.append({"chunk_json": json.dumps(chunk)})
            except Exception:
                pass
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
        """Render retrieved patterns as a prompt snippet."""
        parts = []
        for p in patterns:
            lines = [f"-- Pattern: {p.get('title', '')}"]

            when = p.get("when_to_use", "")
            if when:
                when_str = "; ".join(when) if isinstance(when, list) else when
                lines.append(f"-- Use when: {when_str}")

            template = p.get("template", "")
            if template:
                lines.append(f"-- Template:\n{template}")

            anti = p.get("anti_patterns", [])
            if anti:
                anti_str = "; ".join(anti) if isinstance(anti, list) else anti
                lines.append(f"-- Avoid: {anti_str}")

            hints = p.get("llm_hints", "")
            if hints:
                lines.append(f"-- Hints: {hints}")

            dialect_notes = p.get("dialect_notes", {})
            if dialect and isinstance(dialect_notes, dict) and dialect in dialect_notes:
                lines.append(f"-- {dialect} note: {dialect_notes[dialect]}")

            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    @property
    def count(self) -> int:
        return self._collection.count()
