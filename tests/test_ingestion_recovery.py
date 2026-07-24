from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.database import (
    Base,
    DocumentRepository,
    DocumentStatus,
    IngestionJobRepository,
    IngestionJobStatus,
    IngestionPayloadRepository,
    IngestionQueue,
    QueuedIngestionWorker,
)
from backend.database.ingestion_recovery import (
    IngestionRecovery,
)


@pytest.fixture
def session_factory() -> Iterator[sessionmaker[Session]]:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
    )
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


def test_recovery_prepares_queued_and_interrupted_jobs(
    tmp_path: Path,
    session_factory: sessionmaker[Session],
) -> None:
    queue = IngestionQueue(
        session_factory=session_factory,
        staging_directory=tmp_path / "staging",
    )

    queued = queue.enqueue(
        filename="queued.txt",
        final_storage_path=(tmp_path / "documents" / "queued.txt"),
        content_type="text/plain",
        content=b"Queued recovery content.",
    )
    interrupted = queue.enqueue(
        filename="interrupted.txt",
        final_storage_path=(tmp_path / "documents" / "interrupted.txt"),
        content_type="text/plain",
        content=b"Interrupted recovery content.",
    )

    with session_factory() as session:
        with session.begin():
            jobs = IngestionJobRepository(session)
            interrupted_job = jobs.get(
                interrupted.job_id,
            )

            assert interrupted_job is not None

            jobs.mark_running(
                interrupted_job,
                total_chunks=5,
            )
            jobs.update_progress(
                interrupted_job,
                processed_chunks=2,
            )

    recovery = IngestionRecovery(
        session_factory=session_factory,
    )
    recovered_job_ids = recovery.prepare()

    assert set(recovered_job_ids) == {
        queued.job_id,
        interrupted.job_id,
    }

    with session_factory() as session:
        jobs = IngestionJobRepository(session)
        interrupted_job = jobs.get(
            interrupted.job_id,
        )

        assert interrupted_job is not None
        assert interrupted_job.status is IngestionJobStatus.QUEUED
        assert interrupted_job.attempt_count == 1
        assert interrupted_job.processed_chunks == 0
        assert interrupted_job.total_chunks is None
        assert interrupted_job.started_at is None

    processor = Mock(return_value=3)
    worker = QueuedIngestionWorker(
        session_factory=session_factory,
        process_document=processor,
    )

    for job_id in recovered_job_ids:
        worker.process(job_id)

    assert processor.call_count == 2
    assert not queued.staged_path.exists()
    assert not interrupted.staged_path.exists()

    with session_factory() as session:
        documents = DocumentRepository(session)
        jobs = IngestionJobRepository(session)

        queued_document = documents.get(
            queued.document_id,
        )
        interrupted_document = documents.get(
            interrupted.document_id,
        )
        queued_job = jobs.get(
            queued.job_id,
        )
        interrupted_job = jobs.get(
            interrupted.job_id,
        )

        assert queued_document is not None
        assert interrupted_document is not None
        assert queued_job is not None
        assert interrupted_job is not None

        assert queued_document.status is DocumentStatus.READY
        assert interrupted_document.status is DocumentStatus.READY
        assert queued_job.status is IngestionJobStatus.SUCCEEDED
        assert interrupted_job.status is IngestionJobStatus.SUCCEEDED
        assert queued_job.attempt_count == 1
        assert interrupted_job.attempt_count == 2


def test_recovery_marks_missing_payload_as_failed(
    tmp_path: Path,
    session_factory: sessionmaker[Session],
) -> None:
    queue = IngestionQueue(
        session_factory=session_factory,
        staging_directory=tmp_path / "staging",
    )
    queued = queue.enqueue(
        filename="missing-payload.txt",
        final_storage_path=(tmp_path / "documents" / "missing-payload.txt"),
        content_type="text/plain",
        content=b"Missing payload content.",
    )

    with session_factory() as session:
        with session.begin():
            payloads = IngestionPayloadRepository(
                session,
            )
            payload = payloads.get(
                queued.job_id,
            )

            assert payload is not None
            payloads.delete(payload)

    recovery = IngestionRecovery(
        session_factory=session_factory,
    )

    assert recovery.prepare() == []

    with session_factory() as session:
        document = DocumentRepository(session).get(
            queued.document_id,
        )
        job = IngestionJobRepository(session).get(
            queued.job_id,
        )

        assert document is not None
        assert job is not None

        assert document.status is DocumentStatus.FAILED
        assert job.status is IngestionJobStatus.FAILED
        assert job.error_message == ("The staged ingestion payload record is missing.")
