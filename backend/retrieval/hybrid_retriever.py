from collections import defaultdict
from collections.abc import Sequence

from backend.interfaces.chunker import DocumentChunk
from backend.interfaces.retriever import RetrievalResult, Retriever

ChunkKey = tuple[str, int, int, int]


class HybridRetriever:
    def __init__(
        self,
        retrievers: Sequence[Retriever],
        rrf_k: int = 60,
        candidate_multiplier: int = 4,
    ) -> None:
        if not retrievers:
            raise ValueError("at least one retriever is required")

        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")

        if candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be positive")

        self._retrievers = list(retrievers)
        self._rrf_k = rrf_k
        self._candidate_multiplier = candidate_multiplier

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("query must not be empty")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        candidate_k = top_k * self._candidate_multiplier

        fused_scores: dict[ChunkKey, float] = defaultdict(float)
        chunks: dict[ChunkKey, DocumentChunk] = {}

        for retriever in self._retrievers:
            results = retriever.search(
                query=query,
                top_k=candidate_k,
            )

            for rank, result in enumerate(results, start=1):
                key = self._chunk_key(result.chunk)

                chunks[key] = result.chunk
                fused_scores[key] += 1.0 / (self._rrf_k + rank)

        ranked_keys = sorted(
            fused_scores,
            key=lambda key: fused_scores[key],
            reverse=True,
        )

        return [
            RetrievalResult(
                chunk=chunks[key],
                score=fused_scores[key],
                retriever="hybrid_rrf",
            )
            for key in ranked_keys[:top_k]
        ]

    @staticmethod
    def _chunk_key(chunk: DocumentChunk) -> ChunkKey:
        return (
            str(chunk.source_document.source_path),
            chunk.chunk_index,
            chunk.start_char,
            chunk.end_char,
        )
