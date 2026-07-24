from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from backend.database import (
    Base,
    DocumentRepository,
    IngestionJobRepository,
    IngestionJobStatus,
    IngestionQueue,
)
from backend.database.active_ingestion import (
    ActiveIngestionJobError,
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


def test_same_upload_reuses_active_job(
    tmp_path: Path,
    session_factory: sessionmaker[Session],
) -> None:
    queue = IngestionQueue(
        session_factory=session_factory,
        staging_directory=tmp_path / "staging",
    )
    final_path = tmp_path / "documents" / "duplicate.txt"
    content = b"Duplicate upload content."

    first = queue.enqueue(
        filename="duplicate.txt",
        final_storage_path=final_path,
        content_type="text/plain",
        content=content,
    )
    second = queue.enqueue(
        filename="duplicate.txt",
        final_storage_path=final_path,
        content_type="text/plain",
        content=content,
    )

    assert first.is_new_job is True
    assert second.is_new_job is False
    assert second.document_id == first.document_id
    assert second.job_id == first.job_id
    assert second.staged_path == first.staged_path
    assert second.status is IngestionJobStatus.QUEUED

    with session_factory() as session:
        jobs = IngestionJobRepository(session).list_for_document(
            first.document_id,
        )

        assert len(jobs) == 1
        assert jobs[0].id == first.job_id


def test_different_upload_rejects_duplicate_active_job(
    tmp_path: Path,
    session_factory: sessionmaker[Session],
) -> None:
    queue = IngestionQueue(
        session_factory=session_factory,
        staging_directory=tmp_path / "staging",
    )
    final_path = tmp_path / "documents" / "duplicate.txt"

    first = queue.enqueue(
        filename="duplicate.txt",
        final_storage_path=final_path,
        content_type="text/plain",
        content=b"First upload content.",
    )

    with pytest.raises(
        ActiveIngestionJobError,
        match=str(first.job_id),
    ):
        queue.enqueue(
            filename="duplicate.txt",
            final_storage_path=final_path,
            content_type="text/plain",
            content=b"Different upload content.",
        )


def test_database_rejects_second_active_job(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        with session.begin():
            documents = DocumentRepository(session)
            jobs = IngestionJobRepository(session)

            document = documents.create(
                filename="concurrent.txt",
                storage_path="data/documents/concurrent.txt",
                content_type="text/plain",
                size_bytes=10,
                checksum_sha256="a" * 64,
            )
            jobs.create(
                document_id=document.id,
            )
            document_id = document.id

        with pytest.raises(IntegrityError):
            with session.begin():
                IngestionJobRepository(session).create(
                    document_id=document_id,
                )
