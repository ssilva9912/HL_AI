from backend.interfaces.embedder import EmbeddingProvider
from backend.interfaces.retriever import RetrievalResult
from backend.interfaces.vector_store import VectorStore


class DenseRetriever:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("query must not be empty")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_vector = self._embedder.embed_text(query)
        search_results = self._vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
        )

        return [
            RetrievalResult(
                chunk=result.embedded_chunk.chunk,
                score=result.score,
                retriever="dense",
            )
            for result in search_results
        ]
