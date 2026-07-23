from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.database import (
    Base,
    DocumentRepository,
    IngestionJobRepository,
    IngestionOperation,
    IngestionPayloadRepository,
)


@pytest.fixture
def database_session() -> Iterator[Session]:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
    )
    session = Session(engine)

    Base.metadata.create_all(engine)

    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


def test_ingestion_payload_repository_crud(
    database_session: Session,
) -> None:
    documents = DocumentRepository(database_session)
    jobs = IngestionJobRepository(database_session)
    payloads = IngestionPayloadRepository(
        database_session,
    )

    document = documents.create(
        filename="manual.pdf",
        storage_path="data/documents/manual.pdf",
        content_type="application/pdf",
        size_bytes=1_024,
        checksum_sha256="a" * 64,
    )

    job = jobs.create(
        document_id=document.id,
        operation=IngestionOperation.INDEX,
    )

    payload = payloads.create(
        job_id=job.id,
        staged_path="data/staging/upload.pdf",
        content_type="application/pdf",
        size_bytes=1_024,
        checksum_sha256="B" * 64,
    )

    assert payloads.get(job.id) is payload
    assert payload.job_id == job.id
    assert payload.staged_path == "data/staging/upload.pdf"
    assert payload.content_type == "application/pdf"
    assert payload.size_bytes == 1_024
    assert payload.checksum_sha256 == "b" * 64

    payloads.delete(payload)

    assert payloads.get(job.id) is None


@pytest.mark.parametrize(
    ("staged_path", "size_bytes", "checksum"),
    [
        ("", 1, "a" * 64),
        ("data/staging/file.txt", -1, "a" * 64),
        ("data/staging/file.txt", 1, "invalid"),
    ],
)
def test_ingestion_payload_rejects_invalid_metadata(
    database_session: Session,
    staged_path: str,
    size_bytes: int,
    checksum: str,
) -> None:
    documents = DocumentRepository(database_session)
    jobs = IngestionJobRepository(database_session)
    payloads = IngestionPayloadRepository(
        database_session,
    )

    document = documents.create(
        filename="sample.txt",
        storage_path="data/documents/sample.txt",
        content_type="text/plain",
        size_bytes=1,
        checksum_sha256="c" * 64,
    )

    job = jobs.create(
        document_id=document.id,
    )

    with pytest.raises(ValueError):
        payloads.create(
            job_id=job.id,
            staged_path=staged_path,
            content_type="text/plain",
            size_bytes=size_bytes,
            checksum_sha256=checksum,
        )
