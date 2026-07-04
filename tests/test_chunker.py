from pathlib import Path

import pytest

from backend.chunking.fixed_size_chunker import FixedSizeChunker
from backend.ingestion.scanner import FileMetadata
from backend.interfaces.parser import ParsedDocument


def make_document(content: str) -> ParsedDocument:
    metadata = FileMetadata(
        path=Path("test.txt"),
        name="test.txt",
        size_bytes=len(content),
        extension=".txt",
    )

    return ParsedDocument(
        source_path=Path("test.txt"),
        file_type="text",
        content=content,
        metadata=metadata,
    )


def test_fixed_size_chunker_splits_document() -> None:
    document = make_document("abcdefghijklmnopqrstuvwxyz")
    chunker = FixedSizeChunker(chunk_size=10, overlap=2)

    chunks = chunker.chunk(document)

    assert len(chunks) == 3
    assert chunks[0].content == "abcdefghij"
    assert chunks[1].content == "ijklmnopqr"
    assert chunks[2].content == "qrstuvwxyz"


def test_fixed_size_chunker_preserves_metadata() -> None:
    document = make_document("hello world")
    chunker = FixedSizeChunker(chunk_size=5, overlap=1)

    chunks = chunker.chunk(document)

    assert chunks[0].source_document == document
    assert chunks[0].chunk_index == 0
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == 5


def test_fixed_size_chunker_returns_empty_for_empty_document() -> None:
    document = make_document("")
    chunker = FixedSizeChunker()

    chunks = chunker.chunk(document)

    assert chunks == []


def test_fixed_size_chunker_rejects_invalid_config() -> None:
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=0)

    with pytest.raises(ValueError):
        FixedSizeChunker(overlap=-1)

    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=100, overlap=100)