from collections.abc import Iterator
from hashlib import sha256
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.database import (
    Base,
    DocumentRepository,
    DocumentStatus,
    IngestionJobRepository,
    IngestionJobStatus,
    IngestionOperation,
    IngestionPayloadRepository,
    IngestionQueue,
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


def test_enqueue_new_document(
    tmp_path: Path,
    session_factory: sessionmaker[Session],
) -> None:
    queue = IngestionQueue(
        session_factory=session_factory,
        staging_directory=tmp_path / "staging",
    )

    content = b"Queued document content."

    result = queue.enqueue(
        filename="queued.txt",
        final_storage_path=(tmp_path / "documents" / "queued.txt"),
        content_type="text/plain",
        content=content,
    )

    assert result.operation is IngestionOperation.INDEX
    assert result.size_bytes == len(content)
    assert (
        result.checksum_sha256
        == sha256(
            content,
        ).hexdigest()
    )
    assert result.staged_path.read_bytes() == content

    with session_factory() as session:
        document = DocumentRepository(session).get(
            result.document_id,
        )
        job = IngestionJobRepository(session).get(
            result.job_id,
        )
        payload = IngestionPayloadRepository(
            session,
        ).get(result.job_id)

        assert document is not None
        assert job is not None
        assert payload is not None

        assert document.status is DocumentStatus.PENDING
        assert document.filename == "queued.txt"
        assert document.chunk_count == 0
        assert job.status is IngestionJobStatus.QUEUED
        assert job.attempt_count == 0

        assert payload.staged_path == str(
            result.staged_path,
        )
        assert payload.size_bytes == len(content)
        assert payload.checksum_sha256 == sha256(content).hexdigest()


def test_enqueue_existing_document_creates_reindex_job(
    tmp_path: Path,
    session_factory: sessionmaker[Session],
) -> None:
    final_storage_path = (tmp_path / "documents" / "existing.txt").resolve()

    with session_factory() as session:
        with session.begin():
            documents = DocumentRepository(session)

            document = documents.create(
                filename="existing.txt",
                storage_path=str(final_storage_path),
                content_type="text/plain",
                size_bytes=10,
                checksum_sha256="a" * 64,
            )
            documents.update_status(
                document,
                DocumentStatus.READY,
                chunk_count=3,
            )

            document_id = document.id

    queue = IngestionQueue(
        session_factory=session_factory,
        staging_directory=tmp_path / "staging",
    )

    replacement_content = b"Replacement content."

    result = queue.enqueue(
        filename="existing.txt",
        final_storage_path=final_storage_path,
        content_type="text/plain",
        content=replacement_content,
    )

    assert result.document_id == document_id
    assert result.operation is IngestionOperation.REINDEX

    with session_factory() as session:
        document = DocumentRepository(session).get(
            document_id,
        )
        job = IngestionJobRepository(session).get(
            result.job_id,
        )

        assert document is not None
        assert job is not None

        assert document.status is DocumentStatus.READY
        assert document.chunk_count == 3
        assert document.size_bytes == 10
        assert document.checksum_sha256 == "a" * 64

        assert job.status is IngestionJobStatus.QUEUED
        assert job.operation is IngestionOperation.REINDEX


def test_enqueue_rolls_back_when_staging_fails(
    tmp_path: Path,
    session_factory: sessionmaker[Session],
) -> None:
    invalid_staging_path = tmp_path / "staging-file"
    invalid_staging_path.write_text(
        "This is a file, not a directory.",
        encoding="utf-8",
    )

    queue = IngestionQueue(
        session_factory=session_factory,
        staging_directory=invalid_staging_path,
    )

    with pytest.raises(FileExistsError):
        queue.enqueue(
            filename="failed.txt",
            final_storage_path=(tmp_path / "documents" / "failed.txt"),
            content_type="text/plain",
            content=b"Failed staging content.",
        )

    with session_factory() as session:
        assert (
            DocumentRepository(
                session,
            ).list_all()
            == []
        )
