from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.database.models import (
    IngestionJob,
    IngestionJobStatus,
)


class ActiveIngestionJobError(RuntimeError):
    def __init__(
        self,
        *,
        document_id: UUID,
        job_id: UUID,
    ) -> None:
        self.document_id = document_id
        self.job_id = job_id

        super().__init__(
            f"Document {document_id} already has active ingestion job {job_id}.",
        )


def get_active_ingestion_job(
    session: Session,
    document_id: UUID,
) -> IngestionJob | None:
    statement = (
        select(IngestionJob)
        .where(
            IngestionJob.document_id == document_id,
            IngestionJob.status.in_(
                (
                    IngestionJobStatus.QUEUED,
                    IngestionJobStatus.RUNNING,
                ),
            ),
        )
        .order_by(
            IngestionJob.created_at,
            IngestionJob.id,
        )
        .limit(1)
    )

    return session.scalar(statement)
