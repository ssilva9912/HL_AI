import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.database import (
    Base,
    IngestionJobRepository,
    IngestionJobStatus,
    IngestionPayloadRepository,
    IngestionQueue,
)
from backend.database.ingestion_cleanup import IngestionCleanup


def _session_factory() -> Iterator[sessionmaker[Session]]:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    factory = sessionmaker(
        bind=engine,
        class_=Session,
        expire_on_commit=False,
    )
    Base.metadata.create_all(engine)
    try:
        yield factory
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


def _cleanup(
    *,
    factory: sessionmaker[Session],
    staging_directory: Path,
    now: datetime,
) -> IngestionCleanup:
    return IngestionCleanup(
        session_factory=factory,
        staging_directory=staging_directory,
        abandoned_job_age=timedelta(hours=1),
        failed_payload_retention=timedelta(days=7),
        orphan_file_grace_period=timedelta(hours=1),
        clock=lambda: now,
    )


def test_cleanup_marks_stale_active_job_abandoned(
    tmp_path: Path,
) -> None:
    generator = _session_factory()
    factory = next(generator)
    try:
        queued = IngestionQueue(
            session_factory=factory,
            staging_directory=tmp_path / "staging",
        ).enqueue(
            filename="abandoned.txt",
            final_storage_path=tmp_path / "documents" / "abandoned.txt",
            content_type="text/plain",
            content=b"abandoned",
        )
        now = datetime.now(UTC)
        with factory() as session, session.begin():
            job = IngestionJobRepository(session).get(queued.job_id)
            assert job is not None
            job.updated_at = now - timedelta(hours=2)

        result = _cleanup(
            factory=factory,
            staging_directory=tmp_path / "staging",
            now=now,
        ).run()

        assert result.abandoned_jobs == 1
        assert queued.staged_path.is_file()
        with factory() as session:
            job = IngestionJobRepository(session).get(queued.job_id)
            assert job is not None
            assert job.status is IngestionJobStatus.FAILED
            assert IngestionPayloadRepository(session).get(queued.job_id) is not None
    finally:
        generator.close()


def test_cleanup_expires_failed_payload_and_removes_orphan(
    tmp_path: Path,
) -> None:
    generator = _session_factory()
    factory = next(generator)
    try:
        staging_directory = tmp_path / "staging"
        queued = IngestionQueue(
            session_factory=factory,
            staging_directory=staging_directory,
        ).enqueue(
            filename="expired.txt",
            final_storage_path=tmp_path / "documents" / "expired.txt",
            content_type="text/plain",
            content=b"expired",
        )
        now = datetime.now(UTC)
        with factory() as session, session.begin():
            jobs = IngestionJobRepository(session)
            job = jobs.get(queued.job_id)
            assert job is not None
            jobs.mark_failed(job, error_message="failed")
            job.completed_at = now - timedelta(days=8)

        orphan = staging_directory / "orphan.tmp"
        orphan.write_bytes(b"orphan")
        old_timestamp = (now - timedelta(hours=2)).timestamp()
        os.utime(orphan, (old_timestamp, old_timestamp))

        result = _cleanup(
            factory=factory,
            staging_directory=staging_directory,
            now=now,
        ).run()

        assert result.removed_payloads == 1
        assert result.removed_orphaned_files == 1
        assert not queued.staged_path.exists()
        assert not orphan.exists()
        with factory() as session:
            assert IngestionPayloadRepository(session).get(queued.job_id) is None
    finally:
        generator.close()


def test_cleanup_marks_missing_active_payload_unrecoverable(
    tmp_path: Path,
) -> None:
    generator = _session_factory()
    factory = next(generator)
    try:
        queued = IngestionQueue(
            session_factory=factory,
            staging_directory=tmp_path / "staging",
        ).enqueue(
            filename="missing.txt",
            final_storage_path=tmp_path / "documents" / "missing.txt",
            content_type="text/plain",
            content=b"missing",
        )
        queued.staged_path.unlink()
        now = datetime.now(UTC)

        result = _cleanup(
            factory=factory,
            staging_directory=tmp_path / "staging",
            now=now,
        ).run()

        assert result.unrecoverable_jobs == 1
        with factory() as session:
            job = IngestionJobRepository(session).get(queued.job_id)
            assert job is not None
            assert job.status is IngestionJobStatus.FAILED
    finally:
        generator.close()
