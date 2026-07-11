from pathlib import Path

import pytest

from backend.ingestion.scanner import FileMetadata
from backend.interfaces.chunker import DocumentChunk
from backend.interfaces.parser import ParsedDocument
from backend.interfaces.retriever import RetrievalResult
from backend.retrieval.hybrid_retriever import HybridRetriever


class StubRetriever:
    def __init__(
        self,
        name: str,
        chunks: list[DocumentChunk],
    ) -> None:
        self._name = name
        self._chunks = chunks

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                chunk=chunk,
                score=float(len(self._chunks) - index),
                retriever=self._name,
            )
            for index, chunk in enumerate(self._chunks[:top_k])
        ]


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
        start_char=chunk_index * 100,
        end_char=(chunk_index * 100) + len(content),
    )


def test_rrf_rewards_chunks_found_by_multiple_retrievers() -> None:
    shared = create_chunk(
        "Homelab AI performs local retrieval.",
        chunk_index=0,
    )

    lexical_only = create_chunk(
        "Exact identifier api_embed.",
        chunk_index=1,
    )

    dense_only = create_chunk(
        "Private semantic document search.",
        chunk_index=2,
    )

    lexical = StubRetriever(
        name="bm25",
        chunks=[lexical_only, shared],
    )

    dense = StubRetriever(
        name="dense",
        chunks=[dense_only, shared],
    )

    retriever = HybridRetriever(
        retrievers=[lexical, dense],
    )

    results = retriever.search(
        query="local document retrieval",
        top_k=3,
    )

    assert results[0].chunk == shared
    assert results[0].retriever == "hybrid_rrf"
    assert len(results) == 3


def test_hybrid_retriever_deduplicates_chunks() -> None:
    shared = create_chunk(
        "Shared result.",
        chunk_index=0,
    )

    retriever = HybridRetriever(
        retrievers=[
            StubRetriever("bm25", [shared]),
            StubRetriever("dense", [shared]),
        ]
    )

    results = retriever.search(
        query="shared",
        top_k=5,
    )

    assert len(results) == 1


def test_hybrid_retriever_requires_retrievers() -> None:
    with pytest.raises(
        ValueError,
        match="at least one retriever is required",
    ):
        HybridRetriever([])


def test_hybrid_retriever_rejects_empty_query() -> None:
    chunk = create_chunk(
        "Example",
        chunk_index=0,
    )

    retriever = HybridRetriever(retrievers=[StubRetriever("bm25", [chunk])])

    with pytest.raises(
        ValueError,
        match="query must not be empty",
    ):
        retriever.search("   ")


def test_hybrid_retriever_rejects_invalid_top_k() -> None:
    chunk = create_chunk(
        "Example",
        chunk_index=0,
    )

    retriever = HybridRetriever(retrievers=[StubRetriever("bm25", [chunk])])

    with pytest.raises(
        ValueError,
        match="top_k must be positive",
    ):
        retriever.search("example", top_k=0)
