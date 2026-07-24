from sqlalchemy import Index, text

from backend.database.models import IngestionJob

active_ingestion_job_index = Index(
    "uq_ingestion_jobs_active_document",
    IngestionJob.document_id,
    unique=True,
    postgresql_where=text(
        "status IN ('queued', 'running')",
    ),
    sqlite_where=text(
        "status IN ('queued', 'running')",
    ),
)
