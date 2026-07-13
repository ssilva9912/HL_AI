from pathlib import Path

import pytest

from backend.chunking.semantic_chunker import SemanticChunker
from backend.ingestion.scanner import FileMetadata
from backend.interfaces.parser import ParsedDocument


def make_document(content: str) -> ParsedDocument:
    path = Path("test.txt")

    return ParsedDocument(
        source_path=path,
        file_type=".txt",
        content=content,
        metadata=FileMetadata(
            name=path.name,
            path=path,
            extension=".txt",
            size_bytes=len(content.encode("utf-8")),
        ),
    )


def test_empty_document_returns_no_chunks() -> None:
    chunker = SemanticChunker()

    chunks = chunker.chunk(make_document(""))

    assert chunks == []


def test_whitespace_only_document_returns_no_chunks() -> None:
    chunker = SemanticChunker()

    chunks = chunker.chunk(make_document("   \n\n   "))

    assert chunks == []


def test_single_sentence_creates_one_chunk() -> None:
    content = "Semantic chunking keeps complete sentences together."
    chunker = SemanticChunker(chunk_size=100, overlap=10)

    chunks = chunker.chunk(make_document(content))

    assert len(chunks) == 1
    assert chunks[0].content == content
    assert chunks[0].chunk_index == 0
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == len(content)


def test_sentences_are_grouped_without_exceeding_chunk_size() -> None:
    content = (
        "The first sentence is short. "
        "The second sentence is also short. "
        "The third sentence belongs in another chunk."
    )

    chunker = SemanticChunker(chunk_size=70, overlap=0)

    chunks = chunker.chunk(make_document(content))

    assert len(chunks) == 2
    assert chunks[0].content == ("The first sentence is short. The second sentence is also short.")
    assert chunks[1].content == ("The third sentence belongs in another chunk.")

    assert all(len(chunk.content) <= 70 for chunk in chunks)


def test_paragraph_boundaries_are_preserved_in_chunk_content() -> None:
    content = "This is the first paragraph.\n\nThis is the second paragraph."

    chunker = SemanticChunker(chunk_size=100, overlap=0)

    chunks = chunker.chunk(make_document(content))

    assert len(chunks) == 1
    assert chunks[0].content == content


def test_overlap_preserves_complete_sentence() -> None:
    first = "First sentence contains useful context."
    second = "Second sentence should overlap."
    third = "Third sentence begins the next chunk."

    content = f"{first} {second} {third}"

    chunker = SemanticChunker(
        chunk_size=80,
        overlap=len(second),
    )

    chunks = chunker.chunk(make_document(content))

    assert len(chunks) == 2
    assert second in chunks[0].content
    assert second in chunks[1].content
    assert third in chunks[1].content


def test_long_sentence_is_split_without_losing_text() -> None:
    content = (
        "This sentence contains many words and must be split because its "
        "total character length is greater than the configured chunk size."
    )

    chunker = SemanticChunker(chunk_size=45, overlap=0)

    chunks = chunker.chunk(make_document(content))

    assert len(chunks) > 1
    assert all(len(chunk.content) <= 45 for chunk in chunks)

    reconstructed = " ".join(chunk.content for chunk in chunks)

    assert reconstructed == content


def test_chunk_metadata_matches_original_document() -> None:
    content = "First sentence. Second sentence. Third sentence."

    document = make_document(content)

    chunker = SemanticChunker(
        chunk_size=35,
        overlap=0,
    )

    chunks = chunker.chunk(document)

    for index, chunk in enumerate(chunks):
        assert chunk.source_document is document
        assert chunk.chunk_index == index
        assert chunk.content == content[chunk.start_char : chunk.end_char]


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [
        (0, 0),
        (-1, 0),
        (100, -1),
        (100, 100),
        (100, 101),
    ],
)
def test_invalid_configuration_raises_value_error(
    chunk_size: int,
    overlap: int,
) -> None:
    with pytest.raises(ValueError):
        SemanticChunker(
            chunk_size=chunk_size,
            overlap=overlap,
        )
