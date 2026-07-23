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


def test_worker_processes_queued_document(
    tmp_path: Path,
    session_factory: sessionmaker[Session],
) -> None:
    content = b"Queued worker content."

    queue = IngestionQueue(
        session_factory=session_factory,
        staging_directory=tmp_path / "staging",
    )
    queued = queue.enqueue(
        filename="queued.txt",
        final_storage_path=(tmp_path / "documents" / "queued.txt"),
        content_type="text/plain",
        content=content,
    )

    processor = Mock(return_value=3)

    worker = QueuedIngestionWorker(
        session_factory=session_factory,
        process_document=processor,
    )
    worker.process(queued.job_id)

    processor.assert_called_once_with(
        "queued.txt",
        content,
    )

    assert not queued.staged_path.exists()

    with session_factory() as session:
        document = DocumentRepository(session).get(
            queued.document_id,
        )
        job = IngestionJobRepository(session).get(
            queued.job_id,
        )
        payload = IngestionPayloadRepository(
            session,
        ).get(queued.job_id)

        assert document is not None
        assert job is not None
        assert payload is not None

        assert document.status is DocumentStatus.READY
        assert document.chunk_count == 3
        assert document.size_bytes == len(content)

        assert job.status is IngestionJobStatus.SUCCEEDED
        assert job.attempt_count == 1
        assert job.processed_chunks == 3
        assert job.total_chunks == 3
        assert job.completed_at is not None


def test_worker_records_new_document_failure(
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
        content=b"Failed worker content.",
    )

    processor = Mock(
        side_effect=RuntimeError(
            "Embedding service unavailable.",
        ),
    )

    worker = QueuedIngestionWorker(
        session_factory=session_factory,
        process_document=processor,
    )

    with pytest.raises(
        RuntimeError,
        match="Embedding service unavailable",
    ):
        worker.process(queued.job_id)

    assert queued.staged_path.exists()

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
        assert document.chunk_count == 0
        assert document.error_message == "Embedding service unavailable."

        assert job.status is IngestionJobStatus.FAILED
        assert job.error_message == "Embedding service unavailable."


def test_worker_preserves_ready_document_after_reindex_failure(
    tmp_path: Path,
    session_factory: sessionmaker[Session],
) -> None:
    final_path = (tmp_path / "documents" / "existing.txt").resolve()

    with session_factory() as session:
        with session.begin():
            documents = DocumentRepository(session)

            document = documents.create(
                filename="existing.txt",
                storage_path=str(final_path),
                content_type="text/plain",
                size_bytes=12,
                checksum_sha256="a" * 64,
            )
            documents.update_status(
                document,
                DocumentStatus.READY,
                chunk_count=4,
            )

            document_id = document.id

    queue = IngestionQueue(
        session_factory=session_factory,
        staging_directory=tmp_path / "staging",
    )
    queued = queue.enqueue(
        filename="existing.txt",
        final_storage_path=final_path,
        content_type="text/plain",
        content=b"Replacement content.",
    )

    worker = QueuedIngestionWorker(
        session_factory=session_factory,
        process_document=Mock(
            side_effect=RuntimeError(
                "Replacement indexing failed.",
            ),
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="Replacement indexing failed",
    ):
        worker.process(queued.job_id)

    assert queued.staged_path.exists()

    with session_factory() as session:
        document = DocumentRepository(session).get(
            document_id,
        )
        job = IngestionJobRepository(session).get(
            queued.job_id,
        )

        assert document is not None
        assert job is not None

        assert document.status is DocumentStatus.READY
        assert document.chunk_count == 4
        assert document.size_bytes == 12
        assert document.checksum_sha256 == "a" * 64
        assert document.error_message is None

        assert job.status is IngestionJobStatus.FAILED
        assert job.error_message == "Replacement indexing failed."
