"""RAG (retrieval-augmented generation) memory for ExasolChat.

Stores successful question→SQL pairs in ChromaDB and retrieves semantically
similar ones to inject as few-shot examples into LLM prompts.

Persists to ~/.exasolchat/rag/ by default.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Optional


class _BagOfWordsEF:
    """Minimal ChromaDB embedding function — no model download required.

    Uses hashed token counts projected into a fixed-dim vector, L2-normalised.
    Good enough for SQL Q&A retrieval; zero external dependencies.
    """
    DIM = 512

    def name(self) -> str:
        return "bag-of-words"

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in input]

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.DIM
        for token in text.lower().split():
            idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.DIM
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class RAGMemory:
    """ChromaDB-backed semantic Q&A memory."""

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: str = "exasolchat_rag",
        n_results: int = 3,
    ):
        self._n_results = n_results
        self._persist_dir = persist_dir or str(
            Path.home() / ".exasolchat" / "rag"
        )
        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)

        import chromadb
        self._client = chromadb.PersistentClient(path=self._persist_dir)
        self._ef = _BagOfWordsEF()
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, question: str, sql: str) -> None:
        """Store a question→SQL pair. Deduplicates by question hash."""
        doc_id = hashlib.sha256(question.lower().strip().encode()).hexdigest()[:16]
        self._collection.upsert(
            ids=[doc_id],
            documents=[question],
            metadatas=[{"sql": sql}],
        )

    def search(self, question: str, n_results: Optional[int] = None) -> list[dict]:
        """Return the n most similar past Q&A pairs.

        Each item: {"question": str, "sql": str, "distance": float}
        """
        n = n_results or self._n_results
        if self._collection.count() == 0:
            return []

        actual_n = min(n, self._collection.count())
        results = self._collection.query(
            query_texts=[question],
            n_results=actual_n,
        )

        pairs = []
        for i, doc in enumerate(results["documents"][0]):
            pairs.append({
                "question": doc,
                "sql": results["metadatas"][0][i]["sql"],
                "distance": results["distances"][0][i],
            })
        return pairs

    def format_for_prompt(self, examples: list[dict]) -> str:
        """Render retrieved Q&A pairs as a prompt snippet."""
        parts = []
        for ex in examples:
            parts.append(f"Q: {ex['question']}\nSQL:\n```sql\n{ex['sql']}\n```")
        return "\n\n".join(parts)

    def clear(self) -> None:
        """Delete all stored Q&A pairs."""
        ids = self._collection.get()["ids"]
        if ids:
            self._collection.delete(ids=ids)

    @property
    def count(self) -> int:
        return self._collection.count()

    def list_all(self) -> list[dict]:
        """Return all stored pairs as list of {"question", "sql"} dicts."""
        data = self._collection.get()
        pairs = []
        for i, doc in enumerate(data["documents"]):
            pairs.append({
                "question": doc,
                "sql": data["metadatas"][i]["sql"],
            })
        return pairs


class NoopRAGMemory:
    """Drop-in RAGMemory that does nothing — for stateless/testing use."""

    def add(self, question: str, sql: str) -> None:
        pass

    def search(self, question: str, n_results: Optional[int] = None) -> list[dict]:
        return []

    def format_for_prompt(self, examples: list[dict]) -> str:
        return ""

    def clear(self) -> None:
        pass

    @property
    def count(self) -> int:
        return 0

    def list_all(self) -> list[dict]:
        return []
