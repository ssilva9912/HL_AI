from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.config import Settings, get_settings
from backend.demo import (
    build_rag_pipeline,
    create_demo_documents,
)
from backend.embeddings.ollama_embedder import (
    OllamaEmbedder,
)
from backend.indexing.indexer import Indexer
from backend.indexing.models import IndexedCorpus
from backend.interfaces.parser import ParsedDocument
from backend.storage.qdrant_vector_store import (
    QdrantVectorStore,
)
from evaluation.dataset import load_evaluation_dataset
from evaluation.evaluator import (
    EvaluationReport,
    evaluate_retriever,
)

SUPPORTED_DOCUMENT_EXTENSIONS = {
    ".txt",
    ".md",
    ".pdf",
}

DEFAULT_EVALUATION_DATASET = Path("evaluation/sample_questions.json")

IndexerFactory = Callable[[], Indexer]


class InvalidDocumentNameError(ValueError):
    pass


class UnsupportedDocumentTypeError(ValueError):
    pass


class EmptyDocumentError(ValueError):
    pass


class DocumentTooLargeError(ValueError):
    pass


@dataclass(frozen=True)
class RAGSource:
    text: str
    score: float
    document: str | None = None
    chunk_id: str | None = None


@dataclass(frozen=True)
class RAGAnswer:
    answer: str
    sources: list[RAGSource]


@dataclass(frozen=True)
class IngestionResult:
    document: str
    size_bytes: int
    document_count: int
    chunk_count: int
    status: str = "indexed"


class HomelabRAGService:
    """
    API-facing adapter for the Homelab AI RAG pipeline.

    Qdrant stores document chunk embeddings on disk. Existing vectors are
    restored after an application restart. New uploads are embedded and
    indexed individually instead of rebuilding the entire document directory.
    """

    def __init__(
        self,
        document_directory: Path | None = None,
        settings: Settings | None = None,
        indexer_factory: IndexerFactory | None = None,
        evaluation_dataset_path: Path | None = None,
        vector_store_path: Path | None = None,
    ) -> None:
        self._settings = settings or get_settings()

        self._document_directory = document_directory or self._settings.document_directory

        self._indexer_factory = indexer_factory

        self._evaluation_dataset_path = evaluation_dataset_path or DEFAULT_EVALUATION_DATASET

        self._vector_store_path = vector_store_path or self._settings.vector_store_path

        self._vector_store: QdrantVectorStore | None = None
        self._corpus: IndexedCorpus | None = None

    @property
    def max_upload_bytes(self) -> int:
        return self._settings.max_upload_bytes

    def _prepare_document_directory(self) -> None:
        self._document_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not any(self._document_directory.iterdir()):
            create_demo_documents(self._document_directory)

    def _create_embedder(self) -> OllamaEmbedder:
        return OllamaEmbedder(
            model=self._settings.embedding_model,
            base_url=self._settings.ollama_url,
            timeout=self._settings.embedding_timeout,
        )

    def _get_vector_store(
        self,
    ) -> QdrantVectorStore:
        if self._vector_store is None:
            self._vector_store = QdrantVectorStore(
                storage_path=self._vector_store_path,
                collection_name=(self._settings.vector_collection_name),
            )

        return self._vector_store

    def _create_indexer(self) -> Indexer:
        if self._indexer_factory is not None:
            return self._indexer_factory()

        return Indexer(
            embedder=self._create_embedder(),
            vector_store=self._get_vector_store(),
        )

    def _load_persisted_corpus(
        self,
    ) -> IndexedCorpus | None:
        if self._indexer_factory is not None:
            return None

        vector_store = self._get_vector_store()
        embedded_chunks = vector_store.items()

        if not embedded_chunks:
            return None

        chunks = [embedded_chunk.chunk for embedded_chunk in embedded_chunks]

        documents_by_key: dict[
            tuple[str, str],
            ParsedDocument,
        ] = {}

        for chunk in chunks:
            document = chunk.source_document

            key = (
                str(document.source_path),
                document.metadata.name,
            )

            documents_by_key.setdefault(
                key,
                document,
            )

        return IndexedCorpus(
            documents=list(documents_by_key.values()),
            chunks=chunks,
            embedded_chunks=embedded_chunks,
            vector_store=vector_store,
            embedder=self._create_embedder(),
        )

    def _rebuild_corpus(self) -> IndexedCorpus:
        self._prepare_document_directory()

        if self._indexer_factory is None:
            self._get_vector_store().clear()

        indexer = self._create_indexer()

        self._corpus = indexer.index_directory(self._document_directory)

        return self._corpus

    def _get_corpus(self) -> IndexedCorpus:
        if self._corpus is not None:
            return self._corpus

        persisted_corpus = self._load_persisted_corpus()

        if persisted_corpus is not None:
            self._corpus = persisted_corpus
            return persisted_corpus

        return self._rebuild_corpus()

    def ask(
        self,
        question: str,
        top_k: int | None = None,
    ) -> RAGAnswer:
        normalized_question = question.strip()

        if not normalized_question:
            raise ValueError("question must not be empty")

        resolved_top_k = top_k if top_k is not None else self._settings.default_top_k

        if resolved_top_k <= 0:
            raise ValueError("top_k must be positive")

        corpus = self._get_corpus()

        pipeline = build_rag_pipeline(
            corpus=corpus,
            top_k=resolved_top_k,
        )

        result = pipeline.ask(normalized_question)

        return RAGAnswer(
            answer=result.answer,
            sources=[self._normalize_source(source) for source in result.sources],
        )

    def evaluate(
        self,
        top_k: int = 5,
    ) -> EvaluationReport:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        questions = load_evaluation_dataset(self._evaluation_dataset_path)

        return evaluate_retriever(
            questions=questions,
            retrieve_documents=(self._retrieve_document_names),
            top_k=top_k,
        )

    def _retrieve_document_names(
        self,
        question: str,
        top_k: int,
    ) -> Sequence[str]:
        answer = self.ask(
            question=question,
            top_k=top_k,
        )

        return [source.document for source in answer.sources if source.document is not None]

    def ingest_document(
        self,
        filename: str,
        content: bytes,
    ) -> IngestionResult:
        normalized_filename = filename.strip()

        if not normalized_filename or "/" in normalized_filename or "\\" in normalized_filename:
            raise InvalidDocumentNameError(
                "A valid filename without directory components is required."
            )

        extension = Path(normalized_filename).suffix.lower()

        if extension not in SUPPORTED_DOCUMENT_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_DOCUMENT_EXTENSIONS))

            raise UnsupportedDocumentTypeError(
                f"Unsupported document type: "
                f"{extension or '<none>'}. "
                f"Supported extensions: {supported}."
            )

        if not content:
            raise EmptyDocumentError("Uploaded document cannot be empty.")

        if len(content) > self.max_upload_bytes:
            raise DocumentTooLargeError("Uploaded document exceeds the configured size limit.")

        self._document_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination = self._document_directory / normalized_filename

        previous_content = destination.read_bytes() if destination.exists() else None

        previous_corpus = self._corpus

        if self._indexer_factory is not None:
            destination.write_bytes(content)
            self._corpus = None

            try:
                rebuilt_corpus = self._rebuild_corpus()
            except Exception:
                self._restore_document(
                    destination=destination,
                    previous_content=previous_content,
                )

                self._corpus = previous_corpus
                raise

            return IngestionResult(
                document=normalized_filename,
                size_bytes=len(content),
                document_count=(rebuilt_corpus.document_count),
                chunk_count=(rebuilt_corpus.chunk_count),
            )

        existing_corpus = (
            previous_corpus if previous_corpus is not None else self._load_persisted_corpus()
        )

        previous_document_chunks = (
            [
                embedded_chunk
                for embedded_chunk in existing_corpus.embedded_chunks
                if getattr(
                    embedded_chunk.chunk.source_document.metadata,
                    "name",
                    None,
                )
                == normalized_filename
            ]
            if existing_corpus is not None
            else []
        )

        destination.write_bytes(content)

        vector_store_touched = False

        try:
            staged_corpus = Indexer(
                embedder=self._create_embedder(),
            ).index_file(destination)

            vector_store = self._get_vector_store()

            vector_store_touched = True

            vector_store.replace_document(
                document_name=normalized_filename,
                embedded_chunks=(staged_corpus.embedded_chunks),
            )

            updated_corpus = self._load_persisted_corpus()

            if updated_corpus is None:
                raise RuntimeError(
                    "The document was embedded, but the persisted vector index could not be loaded."
                )

            self._corpus = updated_corpus

            return IngestionResult(
                document=normalized_filename,
                size_bytes=len(content),
                document_count=(updated_corpus.document_count),
                chunk_count=(updated_corpus.chunk_count),
            )

        except Exception:
            self._restore_document(
                destination=destination,
                previous_content=previous_content,
            )

            try:
                if vector_store_touched:
                    self._get_vector_store().replace_document(
                        document_name=(normalized_filename),
                        embedded_chunks=(previous_document_chunks),
                    )
            finally:
                self._corpus = previous_corpus

            raise

    @staticmethod
    def _restore_document(
        destination: Path,
        previous_content: bytes | None,
    ) -> None:
        if previous_content is None:
            destination.unlink(missing_ok=True)
        else:
            destination.write_bytes(previous_content)

    @staticmethod
    def _normalize_source(
        source: Any,
    ) -> RAGSource:
        chunk = source.chunk

        document_name: str | None = None

        source_document = getattr(
            chunk,
            "source_document",
            None,
        )

        if source_document is not None:
            metadata = getattr(
                source_document,
                "metadata",
                None,
            )

            if metadata is not None:
                name = getattr(
                    metadata,
                    "name",
                    None,
                )

                if name is not None:
                    document_name = str(name)

        chunk_index = getattr(
            chunk,
            "chunk_index",
            None,
        )

        return RAGSource(
            text=str(chunk.content),
            score=float(source.score),
            document=document_name,
            chunk_id=(str(chunk_index) if chunk_index is not None else None),
        )
