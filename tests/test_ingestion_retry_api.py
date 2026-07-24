from collections.abc import Iterator
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.dependencies import (
    get_queued_ingestion_service,
)
from backend.api.ingestion_routes import router
from backend.database import IngestionOperation
from backend.database.ingestion_retry import (
    IngestionJobNotFoundError,
    IngestionJobNotRetryableError,
    RetriedIngestion,
)


class FakeRetryService:
    def __init__(self) -> None:
        self.source_job_id = uuid4()
        self.document_id = uuid4()
        self.job_id = uuid4()
        self.processed_jobs: list[UUID] = []
        self.error: Exception | None = None

    def retry_job(
        self,
        job_id: UUID,
    ) -> RetriedIngestion:
        if self.error is not None:
            raise self.error

        return RetriedIngestion(
            source_job_id=job_id,
            document_id=self.document_id,
            job_id=self.job_id,
            operation=IngestionOperation.INDEX,
        )

    def process_job(
        self,
        job_id: UUID,
    ) -> None:
        self.processed_jobs.append(
            job_id,
        )


@pytest.fixture
def client_and_service() -> Iterator[tuple[TestClient, FakeRetryService]]:
    app = FastAPI()
    app.include_router(router)

    service = FakeRetryService()
    app.dependency_overrides[get_queued_ingestion_service] = lambda: service

    with TestClient(app) as client:
        yield client, service


def test_retry_endpoint_returns_new_queued_job(
    client_and_service: tuple[
        TestClient,
        FakeRetryService,
    ],
) -> None:
    client, service = client_and_service

    response = client.post(
        f"/ingestion-jobs/{service.source_job_id}/retry",
    )

    assert response.status_code == 202
    assert response.json() == {
        "source_job_id": str(service.source_job_id),
        "document_id": str(service.document_id),
        "job_id": str(service.job_id),
        "operation": "index",
        "status": "queued",
        "status_url": (f"/ingestion-jobs/{service.job_id}"),
    }
    assert service.processed_jobs == [
        service.job_id,
    ]


def test_retry_endpoint_returns_not_found(
    client_and_service: tuple[
        TestClient,
        FakeRetryService,
    ],
) -> None:
    client, service = client_and_service
    service.error = IngestionJobNotFoundError(
        "Ingestion job does not exist.",
    )

    response = client.post(
        f"/ingestion-jobs/{service.source_job_id}/retry",
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Ingestion job does not exist.",
    }
    assert service.processed_jobs == []


def test_retry_endpoint_returns_conflict(
    client_and_service: tuple[
        TestClient,
        FakeRetryService,
    ],
) -> None:
    client, service = client_and_service
    service.error = IngestionJobNotRetryableError(
        "Only failed ingestion jobs can be retried.",
    )

    response = client.post(
        f"/ingestion-jobs/{service.source_job_id}/retry",
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Only failed ingestion jobs can be retried.",
    }
    assert service.processed_jobs == []
