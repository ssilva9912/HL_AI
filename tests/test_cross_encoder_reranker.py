from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from backend.ingestion.scanner import FileMetadata
from backend.interfaces.chunker import DocumentChunk
from backend.interfaces.parser import ParsedDocument
from backend.interfaces.retriever import RetrievalResult
from backend.retrieval.cross_encoder_reranker import CrossEncoderReranker


def create_chunk(
    content: str,
    chunk_index: int,
) -> DocumentChunk:
    source_path = Path("data/test.txt")

    metadata = FileMetadata(
        name="test.txt",
        path=source_path,
        extension=".txt",
        size_bytes=len(content),
    )

    document = ParsedDocument(
        source_path=source_path,
        file_type="text",
        content=content,
        metadata=metadata,
    )

    return DocumentChunk(
        source_document=document,
        content=content,
        chunk_index=chunk_index,
        start_char=0,
        end_char=len(content),
    )


@patch("backend.retrieval.cross_encoder_reranker.CrossEncoder")
def test_reranker_reorders_results(
    cross_encoder_class: Mock,
) -> None:
    model = cross_encoder_class.return_value
    model.predict.return_value = [0.2, 0.9, 0.4]

    results = [
        RetrievalResult(
            chunk=create_chunk("Architecture overview", 0),
            score=0.032,
            retriever="hybrid_rrf",
        ),
        RetrievalResult(
            chunk=create_chunk(
                "BM25 and dense retrieval are combined with RRF.",
                1,
            ),
            score=0.031,
            retriever="hybrid_rrf",
        ),
        RetrievalResult(
            chunk=create_chunk("Garden irrigation", 2),
            score=0.030,
            retriever="hybrid_rrf",
        ),
    ]

    reranker = CrossEncoderReranker("fake-model")

    reranked = reranker.rerank(
        query="How are BM25 and dense retrieval combined?",
        results=results,
        top_k=2,
    )

    assert len(reranked) == 2
    assert reranked[0].chunk.chunk_index == 1
    assert reranked[0].score == pytest.approx(0.9)
    assert reranked[0].original_score == pytest.approx(0.031)
    assert reranked[0].retriever == "cross_encoder"


@patch("backend.retrieval.cross_encoder_reranker.CrossEncoder")
def test_empty_results_return_empty_list(
    cross_encoder_class: Mock,
) -> None:
    reranker = CrossEncoderReranker("fake-model")

    assert reranker.rerank("query", [], top_k=5) == []

    cross_encoder_class.return_value.predict.assert_not_called()


@patch("backend.retrieval.cross_encoder_reranker.CrossEncoder")
def test_empty_query_raises_error(
    cross_encoder_class: Mock,
) -> None:
    reranker = CrossEncoderReranker("fake-model")

    with pytest.raises(
        ValueError,
        match="query must not be empty",
    ):
        reranker.rerank("   ", [], top_k=5)


@patch("backend.retrieval.cross_encoder_reranker.CrossEncoder")
def test_invalid_top_k_raises_error(
    cross_encoder_class: Mock,
) -> None:
    reranker = CrossEncoderReranker("fake-model")

    with pytest.raises(
        ValueError,
        match="top_k must be positive",
    ):
        reranker.rerank("query", [], top_k=0)


@patch("backend.retrieval.cross_encoder_reranker.CrossEncoder")
def test_model_is_loaded_lazily(
    cross_encoder_class: Mock,
) -> None:
    reranker = CrossEncoderReranker("fake-model")

    cross_encoder_class.assert_not_called()

    results = [
        RetrievalResult(
            chunk=create_chunk("Relevant content", 0),
            score=0.032,
            retriever="hybrid_rrf",
        )
    ]

    cross_encoder_class.return_value.predict.return_value = [0.8]

    reranker.rerank(
        query="relevant query",
        results=results,
        top_k=1,
    )

    cross_encoder_class.assert_called_once_with("fake-model")


@patch("backend.retrieval.cross_encoder_reranker.CrossEncoder")
def test_model_is_reused_across_calls(
    cross_encoder_class: Mock,
) -> None:
    cross_encoder_class.return_value.predict.return_value = [0.8]

    reranker = CrossEncoderReranker("fake-model")

    results = [
        RetrievalResult(
            chunk=create_chunk("Relevant content", 0),
            score=0.032,
            retriever="hybrid_rrf",
        )
    ]

    reranker.rerank("first query", results, top_k=1)
    reranker.rerank("second query", results, top_k=1)

    cross_encoder_class.assert_called_once_with("fake-model")
    assert cross_encoder_class.return_value.predict.call_count == 2
