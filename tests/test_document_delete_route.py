from collections.abc import Iterator
from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.api.app import app
from backend.api.dependencies import get_rag_service
from backend.api.rag_service import (
    DocumentDeletionResult,
    DocumentNotFoundError,
    UnsafeDocumentPathError,
)
from backend.database import get_database_session


class FakeDeletionService:
    def __init__(self) -> None:
        self.error: Exception | None = None
        self.deleted_document_id: UUID | None = None

    def delete_document(
        self,
        document_id: UUID,
        _session: Session,
    ) -> DocumentDeletionResult:
        if self.error is not None:
            raise self.error

        self.deleted_document_id = document_id

        return DocumentDeletionResult(
            document_id=document_id,
            document="deleted.txt",
            deleted_chunk_count=3,
        )


@pytest.fixture
def deletion_client() -> Iterator[
    tuple[
        TestClient,
        FakeDeletionService,
    ]
]:
    fake_service = FakeDeletionService()
    database_session = Mock(
        spec=Session,
    )

    def override_rag_service() -> FakeDeletionService:
        return fake_service

    def override_database_session() -> Iterator[Session]:
        yield database_session

    previous_rag_override = app.dependency_overrides.get(
        get_rag_service,
    )
    previous_database_override = app.dependency_overrides.get(
        get_database_session,
    )

    app.dependency_overrides[get_rag_service] = override_rag_service
    app.dependency_overrides[get_database_session] = override_database_session

    try:
        with TestClient(app) as client:
            yield client, fake_service
    finally:
        if previous_rag_override is None:
            app.dependency_overrides.pop(
                get_rag_service,
                None,
            )
        else:
            app.dependency_overrides[get_rag_service] = previous_rag_override

        if previous_database_override is None:
            app.dependency_overrides.pop(
                get_database_session,
                None,
            )
        else:
            app.dependency_overrides[get_database_session] = previous_database_override


def test_delete_document_returns_no_content(
    deletion_client: tuple[
        TestClient,
        FakeDeletionService,
    ],
) -> None:
    client, fake_service = deletion_client
    document_id = uuid4()

    response = client.delete(
        f"/documents/{document_id}",
    )

    assert response.status_code == 204
    assert response.content == b""
    assert fake_service.deleted_document_id == document_id


def test_delete_document_returns_not_found(
    deletion_client: tuple[
        TestClient,
        FakeDeletionService,
    ],
) -> None:
    client, fake_service = deletion_client
    fake_service.error = DocumentNotFoundError(
        "Document not found.",
    )

    response = client.delete(
        f"/documents/{uuid4()}",
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Document not found.",
    }


def test_delete_document_rejects_unmanaged_path(
    deletion_client: tuple[
        TestClient,
        FakeDeletionService,
    ],
) -> None:
    client, fake_service = deletion_client
    fake_service.error = UnsafeDocumentPathError(
        "Document path is outside managed storage.",
    )

    response = client.delete(
        f"/documents/{uuid4()}",
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": ("The document could not be deleted safely."),
    }


def test_delete_document_handles_unexpected_failure(
    deletion_client: tuple[
        TestClient,
        FakeDeletionService,
    ],
) -> None:
    client, fake_service = deletion_client
    fake_service.error = RuntimeError(
        "Storage unavailable.",
    )

    response = client.delete(
        f"/documents/{uuid4()}",
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": ("The document could not be deleted."),
    }


def test_delete_document_rejects_invalid_uuid(
    deletion_client: tuple[
        TestClient,
        FakeDeletionService,
    ],
) -> None:
    client, fake_service = deletion_client

    response = client.delete(
        "/documents/not-a-uuid",
    )

    assert response.status_code == 422
    assert fake_service.deleted_document_id is None
