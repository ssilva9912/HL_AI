from pathlib import Path

import pytest

from backend.ingestion.scanner import FileMetadata
from backend.interfaces.chunker import DocumentChunk
from backend.interfaces.embedder import EmbeddedChunk
from backend.interfaces.parser import ParsedDocument
from backend.storage.in_memory_vector_store import InMemoryVectorStore


def make_embedded_chunk(content: str, vector: list[float]) -> EmbeddedChunk:
    metadata = FileMetadata(
        path=Path("test.txt"),
        name="test.txt",
        size_bytes=len(content),
        extension=".txt",
    )

    document = ParsedDocument(
        source_path=Path("test.txt"),
        file_type="text",
        content=content,
        metadata=metadata,
    )

    chunk = DocumentChunk(
        source_document=document,
        content=content,
        chunk_index=0,
        start_char=0,
        end_char=len(content),
    )

    return EmbeddedChunk(chunk=chunk, vector=vector)


def test_in_memory_vector_store_adds_chunk() -> None:
    store = InMemoryVectorStore()
    embedded = make_embedded_chunk("hello", [1.0, 0.0])

    store.add(embedded)

    assert store.count() == 1


def test_in_memory_vector_store_searches_by_similarity() -> None:
    store = InMemoryVectorStore()

    first = make_embedded_chunk("first", [1.0, 0.0])
    second = make_embedded_chunk("second", [0.0, 1.0])

    store.add_many([first, second])

    results = store.search(query_vector=[1.0, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0].embedded_chunk == first
    assert results[0].score == pytest.approx(1.0)


def test_in_memory_vector_store_respects_top_k() -> None:
    store = InMemoryVectorStore()

    store.add_many(
        [
            make_embedded_chunk("a", [1.0, 0.0]),
            make_embedded_chunk("b", [0.8, 0.2]),
            make_embedded_chunk("c", [0.0, 1.0]),
        ]
    )

    results = store.search(query_vector=[1.0, 0.0], top_k=2)

    assert len(results) == 2


def test_in_memory_vector_store_rejects_invalid_top_k() -> None:
    store = InMemoryVectorStore()

    with pytest.raises(ValueError):
        store.search(query_vector=[1.0, 0.0], top_k=0)


def test_in_memory_vector_store_rejects_dimension_mismatch() -> None:
    store = InMemoryVectorStore()
    store.add(make_embedded_chunk("hello", [1.0, 0.0]))

    with pytest.raises(ValueError):
        store.search(query_vector=[1.0])
