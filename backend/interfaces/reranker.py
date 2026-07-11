from typing import Protocol

from backend.interfaces.retriever import RetrievalResult


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]: ...
