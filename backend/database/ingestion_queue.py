from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from uuid import UUID

from sqlalchemy.exc import IntegrityError
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


@dataclass(frozen=True)
class QueuedIngestion:
    document_id: UUID
    job_id: UUID
    operation: IngestionOperation
    staged_path: Path
    checksum_sha256: str
    size_bytes: int
    status: IngestionJobStatus = IngestionJobStatus.QUEUED
    is_new_job: bool = True


class IngestionQueue:
    """
    Persist uploaded content before background processing.

    The database transaction and staged file are treated as one logical
    operation. If either database persistence or file staging fails, the
    staged files are removed and the database transaction rolls back.
    """

    def __init__(
        self,
        session_factory: SessionFactory,
        staging_directory: Path,
    ) -> None:
        self._session_factory = session_factory
        self._staging_directory = staging_directory

    def enqueue(
        self,
        *,
        filename: str,
        final_storage_path: Path,
        content_type: str | None,
        content: bytes,
    ) -> QueuedIngestion:
        normalized_filename = filename.strip()

        if not normalized_filename or "/" in normalized_filename or "\\" in normalized_filename:
            raise ValueError(
                "A valid filename without directory components is required.",
            )

        if not content:
            raise ValueError(
                "Queued document content cannot be empty.",
            )

        self._staging_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        resolved_storage_path = final_storage_path.resolve()
        checksum_sha256 = sha256(content).hexdigest()

        temporary_path: Path | None = None
        staged_path: Path | None = None
        document_id: UUID | None = None

        try:
            with self._session_factory() as session:
                with session.begin():
                    documents = DocumentRepository(session)
                    jobs = IngestionJobRepository(session)
                    payloads = IngestionPayloadRepository(
                        session,
                    )

                    document = documents.get_by_storage_path(
                        str(resolved_storage_path),
                    )

                    if document is None:
                        document = documents.create(
                            filename=normalized_filename,
                            storage_path=str(
                                resolved_storage_path,
                            ),
                            content_type=content_type,
                            size_bytes=len(content),
                            checksum_sha256=checksum_sha256,
                        )
                        operation = IngestionOperation.INDEX
                    else:
                        active_job = get_active_ingestion_job(
                            session,
                            document.id,
                        )

                        if active_job is not None:
                            active_payload = payloads.get(
                                active_job.id,
                            )

                            if (
                                active_payload is not None
                                and active_payload.size_bytes == len(content)
                                and active_payload.checksum_sha256 == checksum_sha256
                            ):
                                return QueuedIngestion(
                                    document_id=document.id,
                                    job_id=active_job.id,
                                    operation=active_job.operation,
                                    staged_path=Path(
                                        active_payload.staged_path,
                                    ),
                                    checksum_sha256=(active_payload.checksum_sha256),
                                    size_bytes=active_payload.size_bytes,
                                    status=active_job.status,
                                    is_new_job=False,
                                )

                            raise ActiveIngestionJobError(
                                document_id=document.id,
                                job_id=active_job.id,
                            )

                        operation = IngestionOperation.REINDEX

                    document_id = document.id

                    job = jobs.create(
                        document_id=document.id,
                        operation=operation,
                    )

                    suffix = Path(
                        normalized_filename,
                    ).suffix.lower()

                    staged_path = (self._staging_directory / f"{job.id}{suffix}").resolve()

                    temporary_path = (self._staging_directory / f".{job.id}.tmp").resolve()

                    temporary_path.write_bytes(content)
                    temporary_path.replace(staged_path)

                    payloads.create(
                        job_id=job.id,
                        staged_path=str(staged_path),
                        content_type=content_type,
                        size_bytes=len(content),
                        checksum_sha256=checksum_sha256,
                    )

                    queued_ingestion = QueuedIngestion(
                        document_id=document.id,
                        job_id=job.id,
                        operation=operation,
                        staged_path=staged_path,
                        checksum_sha256=checksum_sha256,
                        size_bytes=len(content),
                    )

            return queued_ingestion

        except IntegrityError as error:
            if temporary_path is not None:
                temporary_path.unlink(
                    missing_ok=True,
                )

            if staged_path is not None:
                staged_path.unlink(
                    missing_ok=True,
                )

            if document_id is not None:
                with self._session_factory() as session:
                    active_job = get_active_ingestion_job(
                        session,
                        document_id,
                    )

                    if active_job is not None:
                        raise ActiveIngestionJobError(
                            document_id=document_id,
                            job_id=active_job.id,
                        ) from error

            raise

        except Exception:
            if temporary_path is not None:
                temporary_path.unlink(
                    missing_ok=True,
                )

            if staged_path is not None:
                staged_path.unlink(
                    missing_ok=True,
                )

            raise
