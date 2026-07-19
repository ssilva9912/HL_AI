from collections.abc import Sequence
from pathlib import Path

from backend.embeddings.ollama_embedder import OllamaEmbedder
from backend.indexing.embedding import EmbeddingPipeline
from backend.indexing.interfaces import IndexVectorStore
from backend.indexing.models import IndexedCorpus
from backend.indexing.processor import DocumentProcessor
from backend.interfaces.chunker import Chunker
from backend.interfaces.embedder import EmbeddingProvider
from backend.interfaces.parser import Parser
from backend.storage.in_memory_vector_store import InMemoryVectorStore


class Indexer:
    def __init__(
        self,
        parser: Parser | None = None,
        chunker: Chunker | None = None,
        embedder: EmbeddingProvider | None = None,
        vector_store: IndexVectorStore | None = None,
    ) -> None:
        self._processor = DocumentProcessor(
            parser=parser,
            chunker=chunker,
        )

        self._embedder = embedder or OllamaEmbedder()
        self._vector_store = vector_store or InMemoryVectorStore()

        self._embedding_pipeline = EmbeddingPipeline(
            self._embedder,
            self._vector_store,
        )

    def index_file(
        self,
        path: Path,
    ) -> IndexedCorpus:
        processed = self._processor.process_file(path)

        return self._embedding_pipeline.embed_processed_documents([processed])

    def index_paths(
        self,
        paths: Sequence[Path],
    ) -> IndexedCorpus:
        if not paths:
            raise ValueError("at least one path is required")

        processed = self._processor.process_paths(list(paths))

        return self._embedding_pipeline.embed_processed_documents(processed)

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
