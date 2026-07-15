from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from backend.chunking.semantic_chunker import SemanticChunker
from backend.embeddings.ollama_embedder import OllamaEmbedder
from backend.ingestion.scanner import FileMetadata
from backend.interfaces.chunker import Chunker, DocumentChunk
from backend.interfaces.embedder import EmbeddedChunk, EmbeddingProvider
from backend.interfaces.parser import ParsedDocument, Parser
from backend.interfaces.vector_store import VectorStore
from backend.parser.markdown_parser import MarkdownParser
from backend.parser.registry import ParserRegistry
from backend.parser.text_parser import TextParser
from backend.storage.in_memory_vector_store import InMemoryVectorStore


class IndexVectorStore(VectorStore, Protocol):
    def add_many(
        self,
        embedded_chunks: list[EmbeddedChunk],
    ) -> None: ...

    def count(self) -> int: ...


@dataclass(frozen=True)
class IndexedCorpus:
    documents: list[ParsedDocument]
    chunks: list[DocumentChunk]
    embedded_chunks: list[EmbeddedChunk]
    vector_store: IndexVectorStore
    embedder: EmbeddingProvider

    @property
    def document_count(self) -> int:
        return len(self.documents)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


class Indexer:
    def __init__(
        self,
        parser: Parser | None = None,
        chunker: Chunker | None = None,
        embedder: EmbeddingProvider | None = None,
        vector_store: IndexVectorStore | None = None,
    ) -> None:
        self._parser = parser or ParserRegistry(
            parsers=[
                TextParser(),
                MarkdownParser(),
            ]
        )
        self._chunker = chunker or SemanticChunker()
        self._embedder = embedder or OllamaEmbedder()
        self._vector_store = vector_store or InMemoryVectorStore()

    def index_paths(
        self,
        paths: Sequence[Path],
    ) -> IndexedCorpus:
        if not paths:
            raise ValueError("at least one path is required")

        documents: list[ParsedDocument] = []
        chunks: list[DocumentChunk] = []

        for path in paths:
            metadata = self._build_metadata(path)

            if not self._parser.can_parse(metadata):
                continue

            document = self._parser.parse(metadata)
            document_chunks = self._chunker.chunk(document)

            documents.append(document)
            chunks.extend(document_chunks)

        embedded_chunks = self._embed_chunks(chunks)
        self._vector_store.add_many(embedded_chunks)

        return IndexedCorpus(
            documents=documents,
            chunks=chunks,
            embedded_chunks=embedded_chunks,
            vector_store=self._vector_store,
            embedder=self._embedder,
        )

    def index_directory(
        self,
        directory: Path,
        recursive: bool = True,
    ) -> IndexedCorpus:
        if not directory.exists():
            raise FileNotFoundError(f"directory does not exist: {directory}")

        if not directory.is_dir():
            raise NotADirectoryError(f"path is not a directory: {directory}")

        iterator = directory.rglob("*") if recursive else directory.glob("*")

        paths = sorted(path for path in iterator if path.is_file())

        if not paths:
            raise ValueError(f"directory contains no files: {directory}")

        return self.index_paths(paths)

    def _embed_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> list[EmbeddedChunk]:
        if not chunks:
            return []

        return self._embedder.embed_many(chunks)

    @staticmethod
    def _build_metadata(path: Path) -> FileMetadata:
        if not path.exists():
            raise FileNotFoundError(f"file does not exist: {path}")

        if not path.is_file():
            raise ValueError(f"path is not a file: {path}")

        return FileMetadata(
            name=path.name,
            path=path,
            extension=path.suffix.lower(),
            size_bytes=path.stat().st_size,
        )
