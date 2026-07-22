from uuid import UUID

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.database import (
    Base,
    Document,
    DocumentStatus,
    IngestionJob,
    IngestionJobStatus,
    IngestionOperation,
)


def test_document_and_ingestion_job_round_trip() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")

    try:
        Base.metadata.create_all(engine)

        with Session(engine) as session:
            document = Document(
                filename="sample.txt",
                storage_path="data/documents/sample.txt",
                content_type="text/plain",
                size_bytes=42,
                checksum_sha256="a" * 64,
            )

            job = IngestionJob(
                document=document,
                operation=IngestionOperation.INDEX,
            )

            session.add(document)
            session.add(job)
            session.commit()

            assert isinstance(document.id, UUID)
            assert isinstance(job.id, UUID)
            assert job.document_id == document.id

            assert document.status is DocumentStatus.PENDING
            assert document.chunk_count == 0

            assert job.status is IngestionJobStatus.QUEUED
            assert job.attempt_count == 0
            assert job.processed_chunks == 0

            assert document.ingestion_jobs == [job]
            assert job.document is document
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()
