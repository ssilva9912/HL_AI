from pathlib import Path
from uuid import UUID, uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.dependencies import (
    get_queued_ingestion_service,
)
from backend.api.ingestion_routes import router
from backend.database import (
    IngestionJobStatus,
    IngestionOperation,
    QueuedIngestion,
)


class FakeDuplicateService:
    max_upload_bytes = 1_024

    def __init__(self) -> None:
        self.document_id = uuid4()
        self.job_id = uuid4()
        self.processed_jobs: list[UUID] = []

    def enqueue_document(
        self,
        *,
        filename: str,
        content: bytes,
    ) -> QueuedIngestion:
        return QueuedIngestion(
            document_id=self.document_id,
            job_id=self.job_id,
            operation=IngestionOperation.INDEX,
            staged_path=Path(
                "data/staging/active.txt",
            ),
            checksum_sha256="a" * 64,
            size_bytes=len(content),
            status=IngestionJobStatus.RUNNING,
            is_new_job=False,
        )

    def process_job(
        self,
        job_id: UUID,
    ) -> None:
        self.processed_jobs.append(
            job_id,
        )


def test_duplicate_upload_does_not_schedule_job_twice() -> None:
    app = FastAPI()
    app.include_router(router)

    service = FakeDuplicateService()
    app.dependency_overrides[get_queued_ingestion_service] = lambda: service

    with TestClient(app) as client:
        response = client.post(
            "/documents",
            files={
                "file": (
                    "active.txt",
                    b"Active upload content.",
                    "text/plain",
                ),
            },
        )

    assert response.status_code == 202
    assert response.json()["job_id"] == str(
        service.job_id,
    )
    assert response.json()["status"] == "running"
    assert service.processed_jobs == []
