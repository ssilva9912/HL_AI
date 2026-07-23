from collections.abc import Iterator
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.api.ingestion_routes import router
from backend.database import (
    Base,
    DocumentRepository,
    IngestionJobRepository,
    IngestionOperation,
    get_database_session,
)


@pytest.fixture
def client_and_session_factory() -> Iterator[tuple[TestClient, sessionmaker[Session]]]:
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

    app = FastAPI()
    app.include_router(router)

    def override_database_session() -> Iterator[Session]:
        with factory() as session:
            yield session

    app.dependency_overrides[get_database_session] = override_database_session

    try:
        with TestClient(app) as client:
            yield client, factory
    finally:
        app.dependency_overrides.clear()
        Base.metadata.drop_all(engine)
        engine.dispose()


def test_get_ingestion_job_returns_progress(
    client_and_session_factory: tuple[
        TestClient,
        sessionmaker[Session],
    ],
) -> None:
    client, factory = client_and_session_factory

    with factory() as session:
        with session.begin():
            documents = DocumentRepository(session)
            jobs = IngestionJobRepository(session)

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
            jobs.mark_running(
                job,
                total_chunks=4,
            )
            jobs.update_progress(
                job,
                processed_chunks=2,
            )

            job_id = job.id
            document_id = document.id

    response = client.get(
        f"/ingestion-jobs/{job_id}",
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["id"] == str(job_id)
    assert payload["document_id"] == str(document_id)
    assert payload["operation"] == "index"
    assert payload["status"] == "running"
    assert payload["attempt_count"] == 1
    assert payload["processed_chunks"] == 2
    assert payload["total_chunks"] == 4
    assert payload["error_message"] is None
    assert payload["created_at"] is not None
    assert payload["started_at"] is not None
    assert payload["completed_at"] is None
    assert payload["updated_at"] is not None


def test_get_ingestion_job_returns_not_found(
    client_and_session_factory: tuple[
        TestClient,
        sessionmaker[Session],
    ],
) -> None:
    client, _ = client_and_session_factory

    response = client.get(
        f"/ingestion-jobs/{uuid4()}",
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Ingestion job not found.",
    }


def test_get_ingestion_job_rejects_invalid_uuid(
    client_and_session_factory: tuple[
        TestClient,
        sessionmaker[Session],
    ],
) -> None:
    client, _ = client_and_session_factory

    response = client.get(
        "/ingestion-jobs/not-a-uuid",
    )

    assert response.status_code == 422
