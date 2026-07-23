import logging
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session

from backend.database.models import (
    DocumentStatus,
    IngestionJobStatus,
    IngestionOperation,
)
from backend.database.payload_repository import (
    IngestionPayloadRepository,
)
from backend.database.repositories import (
    DocumentRepository,
    IngestionJobRepository,
)

logger = logging.getLogger(__name__)

SessionFactory = Callable[[], Session]
DocumentProcessor = Callable[[str, bytes], int]


@dataclass(frozen=True)
class ClaimedIngestion:
    document_id: UUID
    job_id: UUID
    operation: IngestionOperation
    filename: str
    staged_path: Path
    content_type: str | None
    size_bytes: int
    checksum_sha256: str


class QueuedIngestionWorker:
    """
    Process one durable queued ingestion job.

    The processor callback performs parsing, embedding, and Qdrant updates
    and returns the number of chunks indexed for this document.
    """

    def __init__(
        self,
        session_factory: SessionFactory,
        process_document: DocumentProcessor,
    ) -> None:
        self._session_factory = session_factory
        self._process_document = process_document

    def process(
        self,
        job_id: UUID,
    ) -> None:
        claimed = self._claim(job_id)

        try:
            content = self._read_staged_content(
                claimed,
            )

            chunk_count = self._process_document(
                claimed.filename,
                content,
            )

            if chunk_count <= 0:
                raise ValueError(
                    "Document processing must produce at least one chunk.",
                )

            self._mark_succeeded(
                claimed,
                chunk_count=chunk_count,
            )

        except Exception as error:
            self._record_failure(
                claimed,
                error,
            )
            raise

        try:
            claimed.staged_path.unlink(
                missing_ok=True,
            )
        except OSError:
            logger.exception(
                "Failed to remove completed staged upload job_id=%s",
                claimed.job_id,
            )

    def _claim(
        self,
        job_id: UUID,
    ) -> ClaimedIngestion:
        with self._session_factory() as session:
            with session.begin():
                documents = DocumentRepository(session)
                jobs = IngestionJobRepository(session)
                payloads = IngestionPayloadRepository(
                    session,
                )

                job = jobs.get(job_id)

                if job is None:
                    raise LookupError(
                        f"Ingestion job {job_id} does not exist.",
                    )

                if job.status is not IngestionJobStatus.QUEUED:
                    raise ValueError(
                        f"Ingestion job {job_id} is not queued.",
                    )

                document = documents.get(
                    job.document_id,
                )

                if document is None:
                    raise RuntimeError(
                        "The queued ingestion document no longer exists.",
                    )

                payload = payloads.get(job.id)

                if payload is None:
                    raise RuntimeError(
                        "The queued ingestion payload no longer exists.",
                    )

                jobs.mark_running(job)

                if job.operation is IngestionOperation.INDEX:
                    documents.update_status(
                        document,
                        DocumentStatus.INDEXING,
                    )

                return ClaimedIngestion(
                    document_id=document.id,
                    job_id=job.id,
                    operation=job.operation,
                    filename=document.filename,
                    staged_path=Path(
                        payload.staged_path,
                    ),
                    content_type=payload.content_type,
                    size_bytes=payload.size_bytes,
                    checksum_sha256=(payload.checksum_sha256),
                )

    @staticmethod
    def _read_staged_content(
        claimed: ClaimedIngestion,
    ) -> bytes:
        content = claimed.staged_path.read_bytes()

        if len(content) != claimed.size_bytes:
            raise ValueError(
                "Staged upload size does not match its database record.",
            )

        actual_checksum = sha256(content).hexdigest()

        if actual_checksum != claimed.checksum_sha256:
            raise ValueError(
                "Staged upload checksum does not match its database record.",
            )

        return content

    def _mark_succeeded(
        self,
        claimed: ClaimedIngestion,
        *,
        chunk_count: int,
    ) -> None:
        with self._session_factory() as session:
            with session.begin():
                documents = DocumentRepository(session)
                jobs = IngestionJobRepository(session)

                document = documents.get(
                    claimed.document_id,
                )
                job = jobs.get(claimed.job_id)

                if document is None:
                    raise RuntimeError(
                        "The processed document record no longer exists.",
                    )

                if job is None:
                    raise RuntimeError(
                        "The processed ingestion job no longer exists.",
                    )

                documents.update_content(
                    document,
                    filename=claimed.filename,
                    content_type=claimed.content_type,
                    size_bytes=claimed.size_bytes,
                    checksum_sha256=(claimed.checksum_sha256),
                )
                documents.update_status(
                    document,
                    DocumentStatus.READY,
                    chunk_count=chunk_count,
                )

                jobs.update_progress(
                    job,
                    processed_chunks=chunk_count,
                    total_chunks=chunk_count,
                )
                jobs.mark_succeeded(job)

    def _record_failure(
        self,
        claimed: ClaimedIngestion,
        error: Exception,
    ) -> None:
        error_message = str(error).strip() or error.__class__.__name__

        try:
            with self._session_factory() as session:
                with session.begin():
                    documents = DocumentRepository(
                        session,
                    )
                    jobs = IngestionJobRepository(
                        session,
                    )

                    document = documents.get(
                        claimed.document_id,
                    )
                    job = jobs.get(
                        claimed.job_id,
                    )

                    if document is None or job is None:
                        raise RuntimeError(
                            "Failed ingestion records no longer exist.",
                        )

                    if claimed.operation is IngestionOperation.INDEX:
                        documents.update_status(
                            document,
                            DocumentStatus.FAILED,
                            chunk_count=0,
                            error_message=error_message,
                        )

                    jobs.mark_failed(
                        job,
                        error_message=error_message,
                    )

        except Exception:
            logger.exception(
                "Failed to persist queued ingestion failure job_id=%s",
                claimed.job_id,
            )
