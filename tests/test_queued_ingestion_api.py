from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.dependencies import (
    get_queued_ingestion_service,
)
from backend.api.ingestion_routes import router
from backend.database import (
    IngestionOperation,
    QueuedIngestion,
)


class FakeQueuedIngestionService:
    max_upload_bytes = 1_024

    def __init__(self) -> None:
        self.enqueued: list[tuple[str, bytes]] = []
        self.processed_jobs = []

        self.document_id = uuid4()
        self.job_id = uuid4()

    def enqueue_document(
        self,
        *,
        filename: str,
        content: bytes,
    ) -> QueuedIngestion:
        self.enqueued.append(
            (filename, content),
        )

        return QueuedIngestion(
            document_id=self.document_id,
            job_id=self.job_id,
            operation=IngestionOperation.INDEX,
            staged_path=Path(
                "data/staging/queued.txt",
            ),
            checksum_sha256="a" * 64,
            size_bytes=len(content),
        )

    def process_job(
        self,
        job_id,
    ) -> None:
        self.processed_jobs.append(job_id)


def test_queue_document_returns_accepted_job() -> None:
    app = FastAPI()
    app.include_router(router)

    service = FakeQueuedIngestionService()

    app.dependency_overrides[get_queued_ingestion_service] = lambda: service

    with TestClient(app) as client:
        response = client.post(
            "/documents",
            files={
                "file": (
                    "queued.txt",
                    b"Queued API content.",
                    "text/plain",
                ),
            },
        )

    assert response.status_code == 202

    payload = response.json()

    assert payload == {
        "document": "queued.txt",
        "size_bytes": 19,
        "document_id": str(
            service.document_id,
        ),
        "job_id": str(service.job_id),
        "operation": "index",
        "status": "queued",
        "status_url": (f"/ingestion-jobs/{service.job_id}"),
    }

    assert service.enqueued == [
        (
            "queued.txt",
            b"Queued API content.",
        ),
    ]
    assert service.processed_jobs == [
        service.job_id,
    ]
