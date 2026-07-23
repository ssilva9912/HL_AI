from collections.abc import Iterator
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.api.app import app
from backend.database import (
    Base,
    DocumentRepository,
    DocumentStatus,
    IngestionJobRepository,
    get_database_session,
)


@pytest.fixture
def database_client() -> Iterator[
    tuple[
        TestClient,
        sessionmaker[Session],
    ]
]:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={
            "check_same_thread": False,
        },
        poolclass=StaticPool,
    )
    factory = sessionmaker(
        bind=engine,
        class_=Session,
        expire_on_commit=False,
    )

    Base.metadata.create_all(engine)

    def override_database_session() -> Iterator[Session]:
        with factory() as session:
            yield session

    app.dependency_overrides[get_database_session] = override_database_session

    try:
        with TestClient(app) as client:
            yield client, factory
    finally:
        app.dependency_overrides.pop(
            get_database_session,
            None,
        )
        Base.metadata.drop_all(engine)
        engine.dispose()


def create_document(
    factory: sessionmaker[Session],
    *,
    filename: str = "document.txt",
) -> UUID:
    with factory() as session:
        with session.begin():
            documents = DocumentRepository(session)

            document = documents.create(
                filename=filename,
                storage_path=f"/documents/{filename}",
                content_type="text/plain",
                size_bytes=128,
                checksum_sha256="a" * 64,
            )

            documents.update_status(
                document,
                DocumentStatus.READY,
                chunk_count=3,
            )

            document_id = document.id

    return document_id


def test_get_document_returns_persisted_metadata(
    database_client: tuple[
        TestClient,
        sessionmaker[Session],
    ],
) -> None:
    client, factory = database_client
    document_id = create_document(factory)

    response = client.get(
        f"/documents/{document_id}",
    )

    assert response.status_code == 200

    body = response.json()

    assert body["id"] == str(document_id)
    assert body["filename"] == "document.txt"
    assert body["content_type"] == "text/plain"
    assert body["size_bytes"] == 128
    assert body["checksum_sha256"] == "a" * 64
    assert body["status"] == "ready"
    assert body["chunk_count"] == 3
    assert body["error_message"] is None
    assert body["created_at"] is not None
    assert body["updated_at"] is not None


def test_get_document_returns_not_found(
    database_client: tuple[
        TestClient,
        sessionmaker[Session],
    ],
) -> None:
    client, _ = database_client

    response = client.get(
        f"/documents/{uuid4()}",
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Document not found.",
    }


def test_get_document_rejects_invalid_uuid(
    database_client: tuple[
        TestClient,
        sessionmaker[Session],
    ],
) -> None:
    client, _ = database_client

    response = client.get(
        "/documents/not-a-uuid",
    )

    assert response.status_code == 422


def test_list_document_jobs_returns_history(
    database_client: tuple[
        TestClient,
        sessionmaker[Session],
    ],
) -> None:
    client, factory = database_client
    document_id = create_document(factory)

    with factory() as session:
        with session.begin():
            jobs = IngestionJobRepository(session)

            job = jobs.create(
                document_id=document_id,
            )
            jobs.mark_running(
                job,
                total_chunks=4,
            )
            jobs.update_progress(
                job,
                processed_chunks=2,
            )

            job_id = job.id

    response = client.get(
        f"/documents/{document_id}/jobs",
    )

    assert response.status_code == 200

    body = response.json()

    assert len(body) == 1
    assert body[0]["id"] == str(job_id)
    assert body[0]["document_id"] == str(document_id)
    assert body[0]["operation"] == "index"
    assert body[0]["status"] == "running"
    assert body[0]["attempt_count"] == 1
    assert body[0]["processed_chunks"] == 2
    assert body[0]["total_chunks"] == 4
    assert body[0]["error_message"] is None
    assert body[0]["created_at"] is not None
    assert body[0]["started_at"] is not None
    assert body[0]["completed_at"] is None
    assert body[0]["updated_at"] is not None


def test_list_document_jobs_returns_not_found(
    database_client: tuple[
        TestClient,
        sessionmaker[Session],
    ],
) -> None:
    client, _ = database_client

    response = client.get(
        f"/documents/{uuid4()}/jobs",
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Document not found.",
    }
