from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.database import (
    Base,
    Document,
    DocumentRepository,
    DocumentStatus,
    IngestionJob,
    IngestionJobStatus,
    IngestionLifecycle,
    IngestionOperation,
)


@pytest.fixture
def session_factory() -> Iterator[sessionmaker[Session]]:
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


def test_successful_ingestion_lifecycle(
    session_factory: sessionmaker[Session],
) -> None:
    lifecycle = IngestionLifecycle(session_factory)

    handle = lifecycle.begin(
        filename="sample.txt",
        storage_path="data/documents/sample.txt",
        content_type="text/plain",
        size_bytes=42,
        checksum_sha256="a" * 64,
        operation=IngestionOperation.INDEX,
    )

    with session_factory() as session:
        document = session.get(Document, handle.document_id)
        job = session.get(IngestionJob, handle.job_id)

        assert document is not None
        assert job is not None
        assert document.status is DocumentStatus.INDEXING
        assert job.status is IngestionJobStatus.RUNNING
        assert job.attempt_count == 1

    lifecycle.succeed(
        handle,
        chunk_count=3,
    )

    with session_factory() as session:
        document = session.get(Document, handle.document_id)
        job = session.get(IngestionJob, handle.job_id)

        assert document is not None
        assert job is not None
        assert document.status is DocumentStatus.READY
        assert document.chunk_count == 3
        assert document.checksum_sha256 == "a" * 64
        assert job.status is IngestionJobStatus.SUCCEEDED
        assert job.processed_chunks == 3
        assert job.total_chunks == 3
        assert job.completed_at is not None


def test_failed_new_document_is_recorded(
    session_factory: sessionmaker[Session],
) -> None:
    lifecycle = IngestionLifecycle(session_factory)

    handle = lifecycle.begin(
        filename="broken.pdf",
        storage_path="data/documents/broken.pdf",
        content_type="application/pdf",
        size_bytes=100,
        checksum_sha256="b" * 64,
        operation=IngestionOperation.INDEX,
    )

    lifecycle.fail(
        handle,
        error_message="PDF parsing failed.",
    )

    with session_factory() as session:
        document = session.get(Document, handle.document_id)
        job = session.get(IngestionJob, handle.job_id)

        assert document is not None
        assert job is not None
        assert document.status is DocumentStatus.FAILED
        assert document.chunk_count == 0
        assert document.error_message == "PDF parsing failed."
        assert job.status is IngestionJobStatus.FAILED
        assert job.error_message == "PDF parsing failed."
        assert job.completed_at is not None


def test_failed_reindex_restores_previous_document_state(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        with session.begin():
            repository = DocumentRepository(session)

            document = repository.create(
                filename="existing.txt",
                storage_path="data/documents/existing.txt",
                content_type="text/plain",
                size_bytes=50,
                checksum_sha256="c" * 64,
            )
            repository.update_status(
                document,
                DocumentStatus.READY,
                chunk_count=4,
            )

            document_id = document.id

    lifecycle = IngestionLifecycle(session_factory)

    handle = lifecycle.begin(
        filename="existing.txt",
        storage_path="data/documents/existing.txt",
        content_type="text/plain",
        size_bytes=75,
        checksum_sha256="d" * 64,
        operation=IngestionOperation.REINDEX,
    )

    assert handle.document_id == document_id
    assert handle.is_new_document is False

    lifecycle.fail(
        handle,
        error_message="Embedding service unavailable.",
    )

    with session_factory() as session:
        document = session.get(Document, document_id)
        job = session.get(IngestionJob, handle.job_id)

        assert document is not None
        assert job is not None
        assert document.status is DocumentStatus.READY
        assert document.chunk_count == 4
        assert document.size_bytes == 50
        assert document.checksum_sha256 == "c" * 64
        assert document.error_message is None
        assert job.operation is IngestionOperation.REINDEX
        assert job.status is IngestionJobStatus.FAILED
        assert job.error_message == "Embedding service unavailable."
