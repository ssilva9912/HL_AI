from collections.abc import Callable
from pathlib import Path
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.database.models import (
    DocumentStatus,
    IngestionJob,
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


class IngestionRecovery:
    """
    Prepare durable ingestion jobs for processing after an application restart.

    Jobs left running by an interrupted process are returned to the queue.
    Jobs whose staged upload is missing are failed instead of being retried
    forever on every application start.
    """

    def __init__(
        self,
        session_factory: SessionFactory,
    ) -> None:
        self._session_factory = session_factory

    def prepare(self) -> list[UUID]:
        recoverable_job_ids: list[UUID] = []

        with self._session_factory() as session:
            with session.begin():
                documents = DocumentRepository(session)
                jobs = IngestionJobRepository(session)
                payloads = IngestionPayloadRepository(
                    session,
                )

                statement = (
                    select(IngestionJob)
                    .where(
                        IngestionJob.status.in_(
                            (
                                IngestionJobStatus.QUEUED,
                                IngestionJobStatus.RUNNING,
                            ),
                        ),
                    )
                    .order_by(
                        IngestionJob.created_at,
                        IngestionJob.id,
                    )
                )

                recoverable_jobs = list(
                    session.scalars(statement),
                )

                for job in recoverable_jobs:
                    payload = payloads.get(job.id)

                    if payload is None:
                        self._mark_unrecoverable(
                            documents=documents,
                            jobs=jobs,
                            job=job,
                            error_message=("The staged ingestion payload record is missing."),
                        )
                        continue

                    if not Path(payload.staged_path).is_file():
                        self._mark_unrecoverable(
                            documents=documents,
                            jobs=jobs,
                            job=job,
                            error_message=("The staged ingestion payload file is missing."),
                        )
                        continue

                    if job.status is IngestionJobStatus.RUNNING:
                        self._requeue_interrupted(
                            job,
                        )

                    recoverable_job_ids.append(
                        job.id,
                    )

                session.flush()

        return recoverable_job_ids

    @staticmethod
    def _requeue_interrupted(
        job: IngestionJob,
    ) -> None:
        job.status = IngestionJobStatus.QUEUED
        job.processed_chunks = 0
        job.total_chunks = None
        job.error_message = None
        job.started_at = None
        job.completed_at = None

    @staticmethod
    def _mark_unrecoverable(
        *,
        documents: DocumentRepository,
        jobs: IngestionJobRepository,
        job: IngestionJob,
        error_message: str,
    ) -> None:
        document = documents.get(
            job.document_id,
        )

        if document is not None and job.operation is IngestionOperation.INDEX:
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
