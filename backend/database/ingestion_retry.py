from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session

from backend.database.active_ingestion import (
    ActiveIngestionJobError,
    get_active_ingestion_job,
)
from backend.database.models import (
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

SessionFactory = Callable[[], Session]


class IngestionJobNotFoundError(LookupError):
    pass


class IngestionJobNotRetryableError(ValueError):
    pass


class IngestionPayloadUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class RetriedIngestion:
    source_job_id: UUID
    document_id: UUID
    job_id: UUID
    operation: IngestionOperation


class IngestionRetry:
    """
    Create a new queued attempt from a failed durable ingestion job.

    The failed job remains in history. Its staged-payload record is moved
    atomically to the new job, avoiding a second copy of the uploaded file.
    """

    def __init__(
        self,
        session_factory: SessionFactory,
    ) -> None:
        self._session_factory = session_factory

    def retry(
        self,
        job_id: UUID,
    ) -> RetriedIngestion:
        with self._session_factory() as session:
            with session.begin():
                documents = DocumentRepository(session)
                jobs = IngestionJobRepository(session)
                payloads = IngestionPayloadRepository(
                    session,
                )

                source_job = jobs.get(job_id)

                if source_job is None:
                    raise IngestionJobNotFoundError(
                        f"Ingestion job {job_id} does not exist.",
                    )

                if source_job.status is not IngestionJobStatus.FAILED:
                    raise IngestionJobNotRetryableError(
                        "Only failed ingestion jobs can be retried.",
                    )

                if source_job.operation is IngestionOperation.DELETE:
                    raise IngestionJobNotRetryableError(
                        "Delete jobs cannot be retried as document ingestion.",
                    )

                document = documents.get(
                    source_job.document_id,
                )

                if document is None:
                    raise IngestionJobNotRetryableError(
                        "The ingestion document no longer exists.",
                    )

                active_job = get_active_ingestion_job(
                    session,
                    document.id,
                )

                if active_job is not None:
                    raise ActiveIngestionJobError(
                        document_id=document.id,
                        job_id=active_job.id,
                    )

                source_payload = payloads.get(
                    source_job.id,
                )

                if source_payload is None:
                    raise IngestionPayloadUnavailableError(
                        "The failed ingestion payload is unavailable.",
                    )

                staged_path = Path(
                    source_payload.staged_path,
                )

                if not staged_path.is_file():
                    raise IngestionPayloadUnavailableError(
                        "The failed ingestion payload file is unavailable.",
                    )

                staged_path_value = source_payload.staged_path
                content_type = source_payload.content_type
                size_bytes = source_payload.size_bytes
                checksum_sha256 = source_payload.checksum_sha256

                retry_job = jobs.create(
                    document_id=document.id,
                    operation=source_job.operation,
                )

                payloads.delete(
                    source_payload,
                )
                payloads.create(
                    job_id=retry_job.id,
                    staged_path=staged_path_value,
                    content_type=content_type,
                    size_bytes=size_bytes,
                    checksum_sha256=checksum_sha256,
                )

                return RetriedIngestion(
                    source_job_id=source_job.id,
                    document_id=document.id,
                    job_id=retry_job.id,
                    operation=retry_job.operation,
                )
