from backend.interfaces.reranker import Reranker
from backend.interfaces.retriever import RetrievalResult, Retriever


class RerankingRetriever:
    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        candidate_multiplier: int = 4,
    ) -> None:
        if candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be positive")

        self._retriever = retriever
        self._reranker = reranker
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

        candidates = self._retriever.search(
            query=query,
            top_k=candidate_k,
        )

        return self._reranker.rerank(
            query=query,
            results=candidates,
            top_k=top_k,
        )
