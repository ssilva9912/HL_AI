from unittest.mock import Mock

import pytest

from backend.interfaces.retriever import RetrievalResult
from backend.retrieval.reranking_retriever import RerankingRetriever


def test_search_retrieves_expanded_candidate_pool() -> None:
    retriever = Mock()
    reranker = Mock()

    retriever.search.return_value = []
    reranker.rerank.return_value = []

    pipeline = RerankingRetriever(
        retriever=retriever,
        reranker=reranker,
        candidate_multiplier=4,
    )

    pipeline.search(
        query="hybrid retrieval",
        top_k=5,
    )

    retriever.search.assert_called_once_with(
        query="hybrid retrieval",
        top_k=20,
    )

    reranker.rerank.assert_called_once_with(
        query="hybrid retrieval",
        results=[],
        top_k=5,
    )


def test_search_returns_reranked_results() -> None:
    retriever = Mock()
    reranker = Mock()

    candidate = Mock(spec=RetrievalResult)
    reranked = Mock(spec=RetrievalResult)

    retriever.search.return_value = [candidate]
    reranker.rerank.return_value = [reranked]

    pipeline = RerankingRetriever(
        retriever=retriever,
        reranker=reranker,
    )

    results = pipeline.search(
        query="local retrieval architecture",
        top_k=1,
    )

    assert results == [reranked]


def test_empty_query_raises_error() -> None:
    pipeline = RerankingRetriever(
        retriever=Mock(),
        reranker=Mock(),
    )

    with pytest.raises(
        ValueError,
        match="query must not be empty",
    ):
        pipeline.search("   ")


def test_invalid_top_k_raises_error() -> None:
    pipeline = RerankingRetriever(
        retriever=Mock(),
        reranker=Mock(),
    )

    with pytest.raises(
        ValueError,
        match="top_k must be positive",
    ):
        pipeline.search(
            query="valid query",
            top_k=0,
        )


def test_invalid_candidate_multiplier_raises_error() -> None:
    with pytest.raises(
        ValueError,
        match="candidate_multiplier must be positive",
    ):
        RerankingRetriever(
            retriever=Mock(),
            reranker=Mock(),
            candidate_multiplier=0,
        )
