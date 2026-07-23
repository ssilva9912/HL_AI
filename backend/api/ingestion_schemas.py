from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class QueuedIngestionResponse(BaseModel):
    document: str
    size_bytes: int
    document_id: UUID
    job_id: UUID
    operation: str
    status: str
    status_url: str


class IngestionJobResponse(BaseModel):
    id: UUID
    document_id: UUID
    operation: str
    status: str
    attempt_count: int
    processed_chunks: int
    total_chunks: int | None
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    updated_at: datetime
