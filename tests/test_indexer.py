from pathlib import Path

import pytest

from backend.indexing.indexer import Indexer
from backend.interfaces.chunker import DocumentChunk
from backend.interfaces.embedder import EmbeddedChunk
from backend.interfaces.parser import ParsedDocument


class FakeParser:
    def can_parse(self, file) -> bool:
        return file.extension == ".txt"

    def parse(self, file) -> ParsedDocument:
        return ParsedDocument(
            source_path=file.path,
            file_type=file.extension,
            content=file.path.read_text(encoding="utf-8"),
            metadata=file,
        )


class FakeChunker:
    def chunk(
        self,
        document: ParsedDocument,
    ) -> list[DocumentChunk]:
        if not document.content:
            return []

        return [
            DocumentChunk(
                source_document=document,
                content=document.content,
                chunk_index=0,
                start_char=0,
                end_char=len(document.content),
            )
        ]


class FakeEmbedder:
    def __init__(self) -> None:
        self.received_chunks: list[DocumentChunk] = []

    def embed(
        self,
        chunk: DocumentChunk,
    ) -> EmbeddedChunk:
        return EmbeddedChunk(
            chunk=chunk,
            vector=[1.0, 0.0],
        )

    def embed_many(
        self,
        chunks: list[DocumentChunk],
    ) -> list[EmbeddedChunk]:
        self.received_chunks = chunks

        return [self.embed(chunk) for chunk in chunks]


class FakeVectorStore:
    def __init__(self) -> None:
        self.items: list[EmbeddedChunk] = []

    def add_many(
        self,
        embedded_chunks: list[EmbeddedChunk],
    ) -> None:
        self.items.extend(embedded_chunks)

    def count(self) -> int:
        return len(self.items)


def build_indexer() -> tuple[
    Indexer,
    FakeEmbedder,
    FakeVectorStore,
]:
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()

    indexer = Indexer(
        parser=FakeParser(),
        chunker=FakeChunker(),
        embedder=embedder,
        vector_store=vector_store,
    )

    return indexer, embedder, vector_store


def test_index_paths_returns_indexed_corpus(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.txt"
    second_path = tmp_path / "second.txt"

    first_path.write_text(
        "First document.",
        encoding="utf-8",
    )
    second_path.write_text(
        "Second document.",
        encoding="utf-8",
    )

    indexer, embedder, vector_store = build_indexer()

    corpus = indexer.index_paths(
        [
            first_path,
            second_path,
        ]
    )

    assert corpus.document_count == 2
    assert corpus.chunk_count == 2
    assert len(corpus.embedded_chunks) == 2

    assert corpus.embedder is embedder
    assert corpus.vector_store is vector_store

    assert vector_store.count() == 2
    assert len(embedder.received_chunks) == 2


def test_index_paths_preserves_document_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "document.txt"
    content = "Homelab AI indexes local documents."

    path.write_text(
        content,
        encoding="utf-8",
    )

    indexer, _, _ = build_indexer()

    corpus = indexer.index_paths([path])

    document = corpus.documents[0]

    assert document.source_path == path
    assert document.file_type == ".txt"
    assert document.content == content

    assert document.metadata.name == "document.txt"
    assert document.metadata.extension == ".txt"
    assert document.metadata.size_bytes == len(content.encode("utf-8"))


def test_index_paths_skips_unsupported_files(
    tmp_path: Path,
) -> None:
    text_path = tmp_path / "supported.txt"
    unsupported_path = tmp_path / "unsupported.csv"

    text_path.write_text(
        "Supported content.",
        encoding="utf-8",
    )
    unsupported_path.write_text(
        "Unsupported content.",
        encoding="utf-8",
    )

    indexer, _, vector_store = build_indexer()

    corpus = indexer.index_paths(
        [
            text_path,
            unsupported_path,
        ]
    )

    assert corpus.document_count == 1
    assert corpus.documents[0].source_path == text_path
    assert vector_store.count() == 1


def test_index_directory_indexes_nested_files(
    tmp_path: Path,
) -> None:
    nested_directory = tmp_path / "nested"
    nested_directory.mkdir()

    root_file = tmp_path / "root.txt"
    nested_file = nested_directory / "nested.txt"

    root_file.write_text(
        "Root document.",
        encoding="utf-8",
    )
    nested_file.write_text(
        "Nested document.",
        encoding="utf-8",
    )

    indexer, _, _ = build_indexer()

    corpus = indexer.index_directory(
        tmp_path,
        recursive=True,
    )

    assert corpus.document_count == 2

    indexed_names = {document.metadata.name for document in corpus.documents}

    assert indexed_names == {
        "root.txt",
        "nested.txt",
    }


def test_non_recursive_directory_indexing(
    tmp_path: Path,
) -> None:
    nested_directory = tmp_path / "nested"
    nested_directory.mkdir()

    root_file = tmp_path / "root.txt"
    nested_file = nested_directory / "nested.txt"

    root_file.write_text(
        "Root document.",
        encoding="utf-8",
    )
    nested_file.write_text(
        "Nested document.",
        encoding="utf-8",
    )

    indexer, _, _ = build_indexer()

    corpus = indexer.index_directory(
        tmp_path,
        recursive=False,
    )

    assert corpus.document_count == 1
    assert corpus.documents[0].metadata.name == "root.txt"


def test_empty_document_does_not_call_embedder(
    tmp_path: Path,
) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")

    indexer, embedder, vector_store = build_indexer()

    corpus = indexer.index_paths([path])

    assert corpus.document_count == 1
    assert corpus.chunk_count == 0
    assert corpus.embedded_chunks == []

    assert embedder.received_chunks == []
    assert vector_store.count() == 0


def test_empty_path_collection_raises_value_error() -> None:
    indexer, _, _ = build_indexer()

    with pytest.raises(
        ValueError,
        match="at least one path is required",
    ):
        indexer.index_paths([])


def test_missing_file_raises_file_not_found(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing.txt"

    indexer, _, _ = build_indexer()

    with pytest.raises(
        FileNotFoundError,
        match="file does not exist",
    ):
        indexer.index_paths([missing_path])


def test_directory_path_in_index_paths_raises_value_error(
    tmp_path: Path,
) -> None:
    indexer, _, _ = build_indexer()

    with pytest.raises(
        ValueError,
        match="path is not a file",
    ):
        indexer.index_paths([tmp_path])


def test_missing_directory_raises_file_not_found(
    tmp_path: Path,
) -> None:
    missing_directory = tmp_path / "missing"

    indexer, _, _ = build_indexer()

    with pytest.raises(
        FileNotFoundError,
        match="directory does not exist",
    ):
        indexer.index_directory(missing_directory)


def test_file_passed_as_directory_raises_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "file.txt"
    path.write_text(
        "Content.",
        encoding="utf-8",
    )

    indexer, _, _ = build_indexer()

    with pytest.raises(
        NotADirectoryError,
        match="path is not a directory",
    ):
        indexer.index_directory(path)


def test_empty_directory_raises_value_error(
    tmp_path: Path,
) -> None:
    indexer, _, _ = build_indexer()

    with pytest.raises(
        ValueError,
        match="directory contains no files",
    ):
        indexer.index_directory(tmp_path)


def test_default_indexer_supports_text_and_markdown(
    tmp_path: Path,
) -> None:
    text_path = tmp_path / "notes.txt"
    markdown_path = tmp_path / "guide.md"

    text_path.write_text(
        "Plain text document.",
        encoding="utf-8",
    )
    markdown_path.write_text(
        "# Markdown document",
        encoding="utf-8",
    )

    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()

    indexer = Indexer(
        chunker=FakeChunker(),
        embedder=embedder,
        vector_store=vector_store,
    )

    corpus = indexer.index_paths(
        [
            text_path,
            markdown_path,
        ]
    )

    assert corpus.document_count == 2
    assert corpus.chunk_count == 2

    indexed_types = {document.file_type for document in corpus.documents}

    assert indexed_types == {
        "text",
        "markdown",
    }

    indexed_names = {document.metadata.name for document in corpus.documents}

    assert indexed_names == {
        "notes.txt",
        "guide.md",
    }

    assert vector_store.count() == 2
