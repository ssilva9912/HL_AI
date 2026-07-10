from collections.abc import Callable, Sequence

from rank_bm25 import BM25Okapi

from backend.interfaces.chunker import DocumentChunk
from backend.interfaces.retriever import RetrievalResult
from backend.retrieval.tokenizer import tokenize


class BM25Retriever:
    def __init__(
        self,
        chunks: Sequence[DocumentChunk],
        tokenizer: Callable[[str], list[str]] = tokenize,
    ) -> None:
        self._chunks = list(chunks)
        self._tokenizer = tokenizer

        tokenized_corpus = [self._tokenizer(chunk.content) for chunk in self._chunks]

        self._index = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("query must not be empty")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if self._index is None:
            return []

        query_tokens = self._tokenizer(query)
        scores = self._index.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(self._chunks)),
            key=lambda index: float(scores[index]),
            reverse=True,
        )

        results: list[RetrievalResult] = []

        for index in ranked_indices[:top_k]:
            score = float(scores[index])

            results.append(
                RetrievalResult(
                    chunk=self._chunks[index],
                    score=score,
                    retriever="bm25",
                )
            )

        return results
