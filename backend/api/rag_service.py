from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.config import Settings, get_settings
from backend.demo import build_rag_pipeline, create_demo_documents
from backend.indexing.indexer import Indexer
from backend.indexing.models import IndexedCorpus
from evaluation.dataset import load_evaluation_dataset
from evaluation.evaluator import EvaluationReport, evaluate_retriever

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

    The corpus is indexed lazily and reused across search requests. Uploading
    a document rebuilds the current in-memory corpus so the new file becomes
    searchable immediately.
    """

    def __init__(
        self,
        document_directory: Path | None = None,
        settings: Settings | None = None,
        indexer_factory: IndexerFactory | None = None,
        evaluation_dataset_path: Path | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._document_directory = document_directory or self._settings.document_directory
        self._indexer_factory = indexer_factory or Indexer
        self._evaluation_dataset_path = evaluation_dataset_path or DEFAULT_EVALUATION_DATASET
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

    def _rebuild_corpus(self) -> IndexedCorpus:
        self._prepare_document_directory()

        indexer = self._indexer_factory()
        self._corpus = indexer.index_directory(self._document_directory)

        return self._corpus

    def _get_corpus(self) -> IndexedCorpus:
        if self._corpus is None:
            return self._rebuild_corpus()

        return self._corpus

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
            retrieve_documents=self._retrieve_document_names,
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
                f"Unsupported document type: {extension or '<none>'}. "
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

        destination.write_bytes(content)
        self._corpus = None

        try:
            corpus = self._rebuild_corpus()
        except Exception:
            if previous_content is None:
                destination.unlink(missing_ok=True)
            else:
                destination.write_bytes(previous_content)

            self._corpus = None
            raise

        return IngestionResult(
            document=normalized_filename,
            size_bytes=len(content),
            document_count=corpus.document_count,
            chunk_count=corpus.chunk_count,
        )

    @staticmethod
    def _normalize_source(source: Any) -> RAGSource:
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
