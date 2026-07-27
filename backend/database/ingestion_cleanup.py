import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.database.models import (
    DocumentStatus,
    IngestionJob,
    IngestionJobStatus,
    IngestionOperation,
    IngestionPayload,
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
Clock = Callable[[], datetime]


@dataclass(frozen=True)
class IngestionCleanupResult:
    abandoned_jobs: int = 0
    unrecoverable_jobs: int = 0
    removed_payloads: int = 0
    removed_orphaned_files: int = 0


class IngestionCleanup:
    """Apply retention rules to durable ingestion jobs and staged uploads."""

    def __init__(
        self,
        *,
        session_factory: SessionFactory,
        staging_directory: Path,
        abandoned_job_age: timedelta,
        failed_payload_retention: timedelta,
        orphan_file_grace_period: timedelta,
        clock: Clock | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._staging_directory = staging_directory.resolve()
        self._abandoned_job_age = abandoned_job_age
        self._failed_payload_retention = failed_payload_retention
        self._orphan_file_grace_period = orphan_file_grace_period
        self._clock = clock or (lambda: datetime.now(UTC))

    def run(self) -> IngestionCleanupResult:
        now = self._clock()
        abandoned_jobs = 0
        unrecoverable_jobs = 0
        removed_payloads = 0

        with self._session_factory() as session:
            with session.begin():
                documents = DocumentRepository(session)
                jobs = IngestionJobRepository(session)
                payloads = IngestionPayloadRepository(session)

                all_jobs = list(
                    session.scalars(
                        select(IngestionJob).order_by(
                            IngestionJob.created_at,
                            IngestionJob.id,
                        ),
                    ),
                )

                for job in all_jobs:
                    payload = payloads.get(job.id)

                    if job.status in {
                        IngestionJobStatus.QUEUED,
                        IngestionJobStatus.RUNNING,
                    }:
                        if payload is None or not Path(payload.staged_path).is_file():
                            self._mark_failed(
                                documents=documents,
                                jobs=jobs,
                                job=job,
                                error_message=(
                                    "The staged ingestion payload is unavailable; "
                                    "the job cannot be recovered."
                                ),
                            )
                            unrecoverable_jobs += 1
                        elif self._age(now, job.updated_at) >= self._abandoned_job_age:
                            self._mark_failed(
                                documents=documents,
                                jobs=jobs,
                                job=job,
                                error_message=(
                                    "The ingestion job was abandoned after exceeding "
                                    "the active-job retention limit."
                                ),
                            )
                            abandoned_jobs += 1

                    if payload is None:
                        continue

                    should_remove = job.status is IngestionJobStatus.SUCCEEDED
                    if job.status is IngestionJobStatus.FAILED:
                        completed_at = job.completed_at or job.updated_at
                        should_remove = (
                            self._age(now, completed_at) >= self._failed_payload_retention
                        )

                    if should_remove and self._remove_payload_file(
                        Path(payload.staged_path),
                    ):
                        payloads.delete(payload)
                        removed_payloads += 1

                referenced_paths = {
                    Path(path).resolve()
                    for path in session.scalars(
                        select(IngestionPayload.staged_path),
                    )
                }

        removed_orphaned_files = self._remove_orphaned_files(
            referenced_paths=referenced_paths,
            now=now,
        )

        return IngestionCleanupResult(
            abandoned_jobs=abandoned_jobs,
            unrecoverable_jobs=unrecoverable_jobs,
            removed_payloads=removed_payloads,
            removed_orphaned_files=removed_orphaned_files,
        )

    @staticmethod
    def _age(
        now: datetime,
        timestamp: datetime,
    ) -> timedelta:
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        return now - timestamp

    @staticmethod
    def _mark_failed(
        *,
        documents: DocumentRepository,
        jobs: IngestionJobRepository,
        job: IngestionJob,
        error_message: str,
    ) -> None:
        document = documents.get(job.document_id)
        if document is not None and job.operation is IngestionOperation.INDEX:
            documents.update_status(
                document,
                DocumentStatus.FAILED,
                chunk_count=0,
                error_message=error_message,
            )
        jobs.mark_failed(job, error_message=error_message)

    @staticmethod
    def _remove_payload_file(path: Path) -> bool:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.exception(
                "Failed to remove retained ingestion payload path=%s",
                path,
            )
            return False
        return True

    def _remove_orphaned_files(
        self,
        *,
        referenced_paths: set[Path],
        now: datetime,
    ) -> int:
        if not self._staging_directory.is_dir():
            return 0

        removed = 0
        cutoff = now - self._orphan_file_grace_period

        for path in self._staging_directory.iterdir():
            resolved_path = path.resolve()
            if (
                not path.is_file()
                or resolved_path in referenced_paths
                or datetime.fromtimestamp(path.stat().st_mtime, tz=UTC) > cutoff
            ):
                continue

            try:
                path.unlink()
            except OSError:
                logger.exception(
                    "Failed to remove orphaned staging file path=%s",
                    path,
                )
            else:
                removed += 1

        return removed
