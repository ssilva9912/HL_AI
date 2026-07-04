import math

from backend.interfaces.embedder import EmbeddedChunk
from backend.interfaces.vector_store import SearchResult


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._items: list[EmbeddedChunk] = []

    def add(self, embedded_chunk: EmbeddedChunk) -> None:
        self._items.append(embedded_chunk)

    def add_many(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        self._items.extend(embedded_chunks)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        results = [
            SearchResult(
                embedded_chunk=item,
                score=self._cosine_similarity(query_vector, item.vector),
            )
            for item in self._items
        ]

        return sorted(results, key=lambda result: result.score, reverse=True)[:top_k]

    def count(self) -> int:
        return len(self._items)

    @staticmethod
    def _cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
        if len(vector_a) != len(vector_b):
            raise ValueError("vectors must have the same dimension")

        dot_product = sum(a * b for a, b in zip(vector_a, vector_b, strict=True))
        norm_a = math.sqrt(sum(a * a for a in vector_a))
        norm_b = math.sqrt(sum(b * b for b in vector_b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)
