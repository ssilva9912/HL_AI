from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.api.routes import router
from backend.database import (
    Base,
    DocumentRepository,
    DocumentStatus,
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


def test_list_documents_returns_persisted_documents(
    client_and_session_factory: tuple[
        TestClient,
        sessionmaker[Session],
    ],
) -> None:
    client, factory = client_and_session_factory

    with factory() as session:
        with session.begin():
            repository = DocumentRepository(session)

            document = repository.create(
                filename="manual.pdf",
                storage_path="data/documents/manual.pdf",
                content_type="application/pdf",
                size_bytes=1_024,
                checksum_sha256="a" * 64,
            )
            repository.update_status(
                document,
                DocumentStatus.READY,
                chunk_count=8,
            )

            document_id = document.id

    response = client.get("/documents")

    assert response.status_code == 200

    payload = response.json()

    assert len(payload) == 1
    assert payload[0]["id"] == str(document_id)
    assert payload[0]["filename"] == "manual.pdf"
    assert payload[0]["content_type"] == "application/pdf"
    assert payload[0]["size_bytes"] == 1_024
    assert payload[0]["checksum_sha256"] == "a" * 64
    assert payload[0]["status"] == "ready"
    assert payload[0]["chunk_count"] == 8
    assert payload[0]["error_message"] is None
    assert payload[0]["created_at"] is not None
    assert payload[0]["updated_at"] is not None


def test_list_documents_supports_pagination(
    client_and_session_factory: tuple[
        TestClient,
        sessionmaker[Session],
    ],
) -> None:
    client, factory = client_and_session_factory

    with factory() as session:
        with session.begin():
            repository = DocumentRepository(session)

            for index in range(3):
                repository.create(
                    filename=f"document-{index}.txt",
                    storage_path=(f"data/documents/document-{index}.txt"),
                    content_type="text/plain",
                    size_bytes=index + 1,
                    checksum_sha256=str(index + 1) * 64,
                )

    response = client.get(
        "/documents",
        params={
            "offset": 1,
            "limit": 1,
        },
    )

    assert response.status_code == 200
    assert len(response.json()) == 1


def test_list_documents_rejects_invalid_pagination(
    client_and_session_factory: tuple[
        TestClient,
        sessionmaker[Session],
    ],
) -> None:
    client, _ = client_and_session_factory

    response = client.get(
        "/documents",
        params={
            "offset": -1,
            "limit": 0,
        },
    )

    assert response.status_code == 422
