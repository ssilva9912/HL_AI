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
from backend.database.active_ingestion import (
    ActiveIngestionJobError,
)
from backend.database.ingestion_retry import (
    IngestionJobNotRetryableError,
    IngestionRetry,
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


def test_retry_creates_new_attempt_and_preserves_history(
    tmp_path: Path,
    session_factory: sessionmaker[Session],
) -> None:
    queue = IngestionQueue(
        session_factory=session_factory,
        staging_directory=tmp_path / "staging",
    )
    queued = queue.enqueue(
        filename="retry.txt",
        final_storage_path=(tmp_path / "documents" / "retry.txt"),
        content_type="text/plain",
        content=b"Retry ingestion content.",
    )

    failing_worker = QueuedIngestionWorker(
        session_factory=session_factory,
        process_document=Mock(
            side_effect=RuntimeError(
                "Embedding service unavailable.",
            ),
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="Embedding service unavailable",
    ):
        failing_worker.process(
            queued.job_id,
        )

    assert queued.staged_path.is_file()

    retried = IngestionRetry(
        session_factory=session_factory,
    ).retry(
        queued.job_id,
    )

    assert retried.source_job_id == queued.job_id
    assert retried.document_id == queued.document_id
    assert retried.job_id != queued.job_id

    with session_factory() as session:
        jobs = IngestionJobRepository(session)
        payloads = IngestionPayloadRepository(
            session,
        )

        source_job = jobs.get(
            queued.job_id,
        )
        retry_job = jobs.get(
            retried.job_id,
        )
        retry_payload = payloads.get(
            retried.job_id,
        )

        assert source_job is not None
        assert retry_job is not None
        assert retry_payload is not None

        assert source_job.status is IngestionJobStatus.FAILED
        assert source_job.error_message == ("Embedding service unavailable.")
        assert payloads.get(queued.job_id) is None

        assert retry_job.status is IngestionJobStatus.QUEUED
        assert retry_job.attempt_count == 0
        assert retry_payload.staged_path == str(
            queued.staged_path,
        )

    successful_worker = QueuedIngestionWorker(
        session_factory=session_factory,
        process_document=Mock(return_value=4),
    )
    successful_worker.process(
        retried.job_id,
    )

    assert not queued.staged_path.exists()

    with session_factory() as session:
        document = DocumentRepository(session).get(
            queued.document_id,
        )
        jobs = IngestionJobRepository(session)
        source_job = jobs.get(
            queued.job_id,
        )
        retry_job = jobs.get(
            retried.job_id,
        )

        assert document is not None
        assert source_job is not None
        assert retry_job is not None

        assert document.status is DocumentStatus.READY
        assert document.chunk_count == 4
        assert source_job.status is IngestionJobStatus.FAILED
        assert retry_job.status is IngestionJobStatus.SUCCEEDED
        assert retry_job.attempt_count == 1


def test_retry_rejects_nonfailed_job(
    tmp_path: Path,
    session_factory: sessionmaker[Session],
) -> None:
    queued = IngestionQueue(
        session_factory=session_factory,
        staging_directory=tmp_path / "staging",
    ).enqueue(
        filename="queued.txt",
        final_storage_path=(tmp_path / "documents" / "queued.txt"),
        content_type="text/plain",
        content=b"Queued content.",
    )

    with pytest.raises(
        IngestionJobNotRetryableError,
        match="Only failed",
    ):
        IngestionRetry(
            session_factory=session_factory,
        ).retry(
            queued.job_id,
        )


def test_retry_rejects_document_with_active_job(
    tmp_path: Path,
    session_factory: sessionmaker[Session],
) -> None:
    queue = IngestionQueue(
        session_factory=session_factory,
        staging_directory=tmp_path / "staging",
    )
    queued = queue.enqueue(
        filename="failed.txt",
        final_storage_path=(tmp_path / "documents" / "failed.txt"),
        content_type="text/plain",
        content=b"Failed content.",
    )

    failing_worker = QueuedIngestionWorker(
        session_factory=session_factory,
        process_document=Mock(
            side_effect=RuntimeError(
                "Indexing failed.",
            ),
        ),
    )

    with pytest.raises(RuntimeError):
        failing_worker.process(
            queued.job_id,
        )

    with session_factory() as session:
        with session.begin():
            active_job = IngestionJobRepository(
                session,
            ).create(
                document_id=queued.document_id,
            )
            active_job_id = active_job.id

    with pytest.raises(
        ActiveIngestionJobError,
        match=str(active_job_id),
    ):
        IngestionRetry(
            session_factory=session_factory,
        ).retry(
            queued.job_id,
        )
