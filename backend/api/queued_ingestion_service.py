import logging
import mimetypes
from collections.abc import Callable
from pathlib import Path
from threading import Lock
from uuid import UUID

from sqlalchemy.orm import Session

from backend.api.rag_service import (
    SUPPORTED_DOCUMENT_EXTENSIONS,
    DocumentTooLargeError,
    EmptyDocumentError,
    HomelabRAGService,
    InvalidDocumentNameError,
    UnsupportedDocumentTypeError,
)
from backend.config import Settings
from backend.database import (
    IngestionQueue,
    QueuedIngestion,
    QueuedIngestionWorker,
)
from backend.database.ingestion_recovery import (
    IngestionRecovery,
)
from backend.database.ingestion_retry import (
    IngestionRetry,
    RetriedIngestion,
)

logger = logging.getLogger(__name__)

SessionFactory = Callable[[], Session]


class QueuedDocumentIngestionService:
    """
    Coordinate durable upload staging and serialized background indexing.

    PostgreSQL stores the queue while one in-process worker owns writes to
    the embedded Qdrant database.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        rag_service: HomelabRAGService,
        session_factory: SessionFactory,
    ) -> None:
        self._settings = settings
        self._rag_service = rag_service

        self._queue = IngestionQueue(
            session_factory=session_factory,
            staging_directory=settings.staging_directory,
        )
        self._worker = QueuedIngestionWorker(
            session_factory=session_factory,
            process_document=self._process_document,
        )
        self._recovery = IngestionRecovery(
            session_factory=session_factory,
        )
        self._retry = IngestionRetry(
            session_factory=session_factory,
        )

        self._worker_lock = Lock()

    @property
    def max_upload_bytes(self) -> int:
        return self._settings.max_upload_bytes

    def enqueue_document(
        self,
        *,
        filename: str,
        content: bytes,
    ) -> QueuedIngestion:
        normalized_filename = self._validate_upload(
            filename=filename,
            content=content,
        )

        content_type, _ = mimetypes.guess_type(
            normalized_filename,
        )

        return self._queue.enqueue(
            filename=normalized_filename,
            final_storage_path=(self._settings.document_directory / normalized_filename),
            content_type=content_type,
            content=content,
        )

    def retry_job(
        self,
        job_id: UUID,
    ) -> RetriedIngestion:
        return self._retry.retry(
            job_id,
        )

    def process_job(
        self,
        job_id: UUID,
    ) -> None:
        with self._worker_lock:
            self._worker.process(job_id)

    def recover_pending_jobs(self) -> int:
        job_ids = self._recovery.prepare()

        for job_id in job_ids:
            try:
                self.process_job(job_id)
            except Exception:
                logger.exception(
                    "Recovered ingestion job failed job_id=%s",
                    job_id,
                )

        return len(job_ids)

    def _process_document(
        self,
        filename: str,
        content: bytes,
    ) -> int:
        result = self._rag_service.ingest_document(
            filename=filename,
            content=content,
            track_ingestion=False,
        )

        if result.document_chunk_count is None:
            raise RuntimeError(
                "Document ingestion did not report its indexed chunk count.",
            )

        return result.document_chunk_count

    def _validate_upload(
        self,
        *,
        filename: str,
        content: bytes,
    ) -> str:
        normalized_filename = filename.strip()

        if not normalized_filename or "/" in normalized_filename or "\\" in normalized_filename:
            raise InvalidDocumentNameError(
                "A valid filename without directory components is required.",
            )

        extension = Path(
            normalized_filename,
        ).suffix.lower()

        if extension not in SUPPORTED_DOCUMENT_EXTENSIONS:
            supported = ", ".join(
                sorted(SUPPORTED_DOCUMENT_EXTENSIONS),
            )

            raise UnsupportedDocumentTypeError(
                f"Unsupported document type: "
                f"{extension or '<none>'}. "
                f"Supported extensions: {supported}.",
            )

        if not content:
            raise EmptyDocumentError(
                "Uploaded document cannot be empty.",
            )

        if len(content) > self.max_upload_bytes:
            raise DocumentTooLargeError(
                "Uploaded document exceeds the configured size limit.",
            )

        return normalized_filename
