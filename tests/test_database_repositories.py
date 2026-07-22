from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.database import (
    Base,
    DocumentRepository,
    DocumentStatus,
    IngestionJobRepository,
    IngestionJobStatus,
    IngestionOperation,
)


@pytest.fixture
def database_session() -> Iterator[Session]:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    session = Session(engine)

    try:
        Base.metadata.create_all(engine)
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


def test_document_repository_crud(
    database_session: Session,
) -> None:
    repository = DocumentRepository(database_session)

    document = repository.create(
        filename="sample.txt",
        storage_path="data/documents/sample.txt",
        content_type="text/plain",
        size_bytes=42,
        checksum_sha256="a" * 64,
    )

    assert repository.get(document.id) is document
    assert repository.get_by_storage_path(document.storage_path) is document
    assert repository.list_all() == [document]
    assert repository.list_by_checksum("A" * 64) == [document]

    repository.update_status(
        document,
        DocumentStatus.READY,
        chunk_count=3,
    )

    assert document.status is DocumentStatus.READY
    assert document.chunk_count == 3
    assert document.error_message is None

    document_id = document.id
    repository.delete(document)

    assert repository.get(document_id) is None


def test_document_repository_allows_duplicate_content(
    database_session: Session,
) -> None:
    repository = DocumentRepository(database_session)
    checksum = "b" * 64

    first_document = repository.create(
        filename="first.txt",
        storage_path="data/documents/first.txt",
        content_type="text/plain",
        size_bytes=10,
        checksum_sha256=checksum,
    )

    second_document = repository.create(
        filename="second.txt",
        storage_path="data/documents/second.txt",
        content_type="text/plain",
        size_bytes=10,
        checksum_sha256=checksum,
    )

    matching_documents = repository.list_by_checksum(checksum)

    assert set(matching_documents) == {
        first_document,
        second_document,
    }


def test_ingestion_job_repository_lifecycle(
    database_session: Session,
) -> None:
    document_repository = DocumentRepository(database_session)
    job_repository = IngestionJobRepository(database_session)

    document = document_repository.create(
        filename="sample.md",
        storage_path="data/documents/sample.md",
        content_type="text/markdown",
        size_bytes=100,
        checksum_sha256="c" * 64,
    )

    job = job_repository.create(
        document_id=document.id,
        operation=IngestionOperation.INDEX,
    )

    assert job.status is IngestionJobStatus.QUEUED
    assert job.attempt_count == 0
    assert job_repository.get(job.id) is job

    job_repository.mark_running(
        job,
        total_chunks=4,
    )

    assert job.status is IngestionJobStatus.RUNNING
    assert job.attempt_count == 1
    assert job.started_at is not None

    job_repository.update_progress(
        job,
        processed_chunks=2,
    )

    assert job.processed_chunks == 2
    assert job.total_chunks == 4

    job_repository.mark_succeeded(job)

    assert job.status is IngestionJobStatus.SUCCEEDED
    assert job.processed_chunks == 4
    assert job.completed_at is not None
    assert job.error_message is None

    assert job_repository.list_for_document(document.id) == [job]
