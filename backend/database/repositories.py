from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.database.models import (
    Document,
    DocumentStatus,
    IngestionJob,
    IngestionJobStatus,
    IngestionOperation,
)


def _validate_sha256(checksum: str) -> str:
    normalized_checksum = checksum.lower()

    if len(normalized_checksum) != 64:
        raise ValueError("SHA-256 checksum must contain exactly 64 characters.")

    try:
        int(normalized_checksum, 16)
    except ValueError as error:
        raise ValueError("SHA-256 checksum must be hexadecimal.") from error

    return normalized_checksum


def _validate_pagination(offset: int, limit: int) -> None:
    if offset < 0:
        raise ValueError("Pagination offset cannot be negative.")

    if limit < 1 or limit > 1000:
        raise ValueError("Pagination limit must be between 1 and 1000.")


def _normalize_content_type(content_type: str | None) -> str | None:
    if content_type is None:
        return None

    normalized_content_type = content_type.strip()

    return normalized_content_type or None


class DocumentRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        *,
        filename: str,
        storage_path: str,
        content_type: str | None,
        size_bytes: int,
        checksum_sha256: str,
    ) -> Document:
        normalized_filename = filename.strip()
        normalized_storage_path = storage_path.strip()

        if not normalized_filename:
            raise ValueError("Document filename cannot be empty.")

        if not normalized_storage_path:
            raise ValueError("Document storage path cannot be empty.")

        if size_bytes < 0:
            raise ValueError("Document size cannot be negative.")

        document = Document(
            filename=normalized_filename,
            storage_path=normalized_storage_path,
            content_type=_normalize_content_type(content_type),
            size_bytes=size_bytes,
            checksum_sha256=_validate_sha256(checksum_sha256),
        )

        self._session.add(document)
        self._session.flush()

        return document

    def update_content(
        self,
        document: Document,
        *,
        filename: str,
        content_type: str | None,
        size_bytes: int,
        checksum_sha256: str,
    ) -> Document:
        normalized_filename = filename.strip()

        if not normalized_filename:
            raise ValueError("Document filename cannot be empty.")

        if size_bytes < 0:
            raise ValueError("Document size cannot be negative.")

        document.filename = normalized_filename
        document.content_type = _normalize_content_type(content_type)
        document.size_bytes = size_bytes
        document.checksum_sha256 = _validate_sha256(checksum_sha256)

        self._session.flush()

        return document

    def get(self, document_id: UUID) -> Document | None:
        return self._session.get(Document, document_id)

    def get_by_storage_path(self, storage_path: str) -> Document | None:
        statement = select(Document).where(
            Document.storage_path == storage_path,
        )
        return self._session.scalar(statement)

    def list_by_checksum(self, checksum_sha256: str) -> list[Document]:
        statement = (
            select(Document)
            .where(
                Document.checksum_sha256 == _validate_sha256(checksum_sha256),
            )
            .order_by(Document.created_at.desc(), Document.id)
        )
        return list(self._session.scalars(statement))

    def list_all(
        self,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> list[Document]:
        _validate_pagination(offset, limit)

        statement = (
            select(Document)
            .order_by(Document.created_at.desc(), Document.id)
            .offset(offset)
            .limit(limit)
        )
        return list(self._session.scalars(statement))

    def update_status(
        self,
        document: Document,
        status: DocumentStatus,
        *,
        chunk_count: int | None = None,
        error_message: str | None = None,
    ) -> Document:
        if chunk_count is not None and chunk_count < 0:
            raise ValueError("Document chunk count cannot be negative.")

        document.status = status
        document.error_message = error_message

        if chunk_count is not None:
            document.chunk_count = chunk_count

        self._session.flush()

        return document

    def delete(self, document: Document) -> None:
        self._session.delete(document)
        self._session.flush()


class IngestionJobRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        *,
        document_id: UUID,
        operation: IngestionOperation = IngestionOperation.INDEX,
    ) -> IngestionJob:
        job = IngestionJob(
            document_id=document_id,
            operation=operation,
        )

        self._session.add(job)
        self._session.flush()

        return job

    def get(self, job_id: UUID) -> IngestionJob | None:
        return self._session.get(IngestionJob, job_id)

    def list_for_document(
        self,
        document_id: UUID,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> list[IngestionJob]:
        _validate_pagination(offset, limit)

        statement = (
            select(IngestionJob)
            .where(IngestionJob.document_id == document_id)
            .order_by(IngestionJob.created_at.desc(), IngestionJob.id)
            .offset(offset)
            .limit(limit)
        )
        return list(self._session.scalars(statement))

    def mark_running(
        self,
        job: IngestionJob,
        *,
        total_chunks: int | None = None,
    ) -> IngestionJob:
        if total_chunks is not None and total_chunks < 0:
            raise ValueError("Total chunk count cannot be negative.")

        job.status = IngestionJobStatus.RUNNING
        job.attempt_count += 1
        job.total_chunks = total_chunks
        job.processed_chunks = 0
        job.error_message = None
        job.started_at = datetime.now(UTC)
        job.completed_at = None

        self._session.flush()

        return job

    def update_progress(
        self,
        job: IngestionJob,
        *,
        processed_chunks: int,
        total_chunks: int | None = None,
    ) -> IngestionJob:
        effective_total = total_chunks
        if effective_total is None:
            effective_total = job.total_chunks

        if processed_chunks < 0:
            raise ValueError("Processed chunk count cannot be negative.")

        if effective_total is not None:
            if effective_total < 0:
                raise ValueError("Total chunk count cannot be negative.")

            if processed_chunks > effective_total:
                raise ValueError(
                    "Processed chunk count cannot exceed the total.",
                )

        job.processed_chunks = processed_chunks
        job.total_chunks = effective_total

        self._session.flush()

        return job

    def mark_succeeded(self, job: IngestionJob) -> IngestionJob:
        job.status = IngestionJobStatus.SUCCEEDED
        job.error_message = None
        job.completed_at = datetime.now(UTC)

        if job.total_chunks is not None:
            job.processed_chunks = job.total_chunks

        self._session.flush()

        return job

    def mark_failed(
        self,
        job: IngestionJob,
        *,
        error_message: str,
    ) -> IngestionJob:
        normalized_error = error_message.strip()

        if not normalized_error:
            raise ValueError("Failed ingestion jobs require an error message.")

        job.status = IngestionJobStatus.FAILED
        job.error_message = normalized_error
        job.completed_at = datetime.now(UTC)

        self._session.flush()

        return job

    def delete(self, job: IngestionJob) -> None:
        self._session.delete(job)
        self._session.flush()
