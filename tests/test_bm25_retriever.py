from pathlib import Path

import pytest

from backend.ingestion.scanner import FileMetadata
from backend.interfaces.chunker import DocumentChunk
from backend.interfaces.parser import ParsedDocument
from backend.retrieval.bm25_retriever import BM25Retriever


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


def test_search_returns_exact_term_match_first() -> None:
    chunks = [
        create_chunk(
            "Tomatoes require sunlight and regular watering.",
            chunk_index=0,
        ),
        create_chunk(
            "The OllamaEmbedder sends requests to the api_embed endpoint.",
            chunk_index=1,
        ),
        create_chunk(
            "Homelab AI performs local document retrieval.",
            chunk_index=2,
        ),
    ]

    retriever = BM25Retriever(chunks)

    results = retriever.search(
        query="OllamaEmbedder api_embed",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk.chunk_index == 1
    assert results[0].retriever == "bm25"
    assert results[0].score > results[1].score


def test_search_preserves_underscored_identifiers() -> None:
    chunks = [
        create_chunk(
            "The source_document contains parsed file metadata.",
            chunk_index=0,
        ),
        create_chunk(
            "The garden controller reads moisture sensors.",
            chunk_index=1,
        ),
    ]

    retriever = BM25Retriever(chunks)

    results = retriever.search(
        query="source_document",
        top_k=1,
    )

    assert results[0].chunk.chunk_index == 0


def test_empty_index_returns_no_results() -> None:
    retriever = BM25Retriever([])

    assert retriever.search("homelab", top_k=5) == []


def test_empty_query_raises_error() -> None:
    retriever = BM25Retriever([create_chunk("Example document", chunk_index=0)])

    with pytest.raises(
        ValueError,
        match="query must not be empty",
    ):
        retriever.search("   ")


def test_non_positive_top_k_raises_error() -> None:
    retriever = BM25Retriever([create_chunk("Example document", chunk_index=0)])

    with pytest.raises(
        ValueError,
        match="top_k must be positive",
    ):
        retriever.search("example", top_k=0)
