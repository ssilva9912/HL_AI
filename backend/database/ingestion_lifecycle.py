from collections.abc import Callable
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.orm import Session

from backend.database.models import (
    DocumentStatus,
    IngestionOperation,
)
from backend.database.repositories import (
    DocumentRepository,
    IngestionJobRepository,
)

SessionFactory = Callable[[], Session]


@dataclass(frozen=True)
class IngestionHandle:
    document_id: UUID
    job_id: UUID
    is_new_document: bool

    filename: str
    storage_path: str
    content_type: str | None
    size_bytes: int
    checksum_sha256: str

    previous_status: DocumentStatus
    previous_chunk_count: int
    previous_error_message: str | None


class IngestionLifecycle:
    """
    Persist ingestion state using short database transactions.

    Slow parsing, embedding, and vector-store operations happen between
    begin() and succeed()/fail(), outside an open database transaction.
    """

    def __init__(self, session_factory: SessionFactory) -> None:
        self._session_factory = session_factory

    def begin(
        self,
        *,
        filename: str,
        storage_path: str,
        content_type: str | None,
        size_bytes: int,
        checksum_sha256: str,
        operation: IngestionOperation,
    ) -> IngestionHandle:
        with self._session_factory() as session:
            with session.begin():
                documents = DocumentRepository(session)
                jobs = IngestionJobRepository(session)

                document = documents.get_by_storage_path(storage_path)
                is_new_document = document is None

                if document is None:
                    document = documents.create(
                        filename=filename,
                        storage_path=storage_path,
                        content_type=content_type,
                        size_bytes=size_bytes,
                        checksum_sha256=checksum_sha256,
                    )

                previous_status = document.status
                previous_chunk_count = document.chunk_count
                previous_error_message = document.error_message

                job = jobs.create(
                    document_id=document.id,
                    operation=operation,
                )

                documents.update_status(
                    document,
                    DocumentStatus.INDEXING,
                )
                jobs.mark_running(job)

                handle = IngestionHandle(
                    document_id=document.id,
                    job_id=job.id,
                    is_new_document=is_new_document,
                    filename=filename,
                    storage_path=storage_path,
                    content_type=content_type,
                    size_bytes=size_bytes,
                    checksum_sha256=checksum_sha256,
                    previous_status=previous_status,
                    previous_chunk_count=previous_chunk_count,
                    previous_error_message=previous_error_message,
                )

        return handle

    def succeed(
        self,
        handle: IngestionHandle,
        *,
        chunk_count: int,
    ) -> None:
        if chunk_count < 0:
            raise ValueError("Document chunk count cannot be negative.")

        with self._session_factory() as session:
            with session.begin():
                documents = DocumentRepository(session)
                jobs = IngestionJobRepository(session)

                document = documents.get(handle.document_id)
                if document is None:
                    raise RuntimeError(
                        "The ingestion document record no longer exists.",
                    )

                job = jobs.get(handle.job_id)
                if job is None:
                    raise RuntimeError(
                        "The ingestion job record no longer exists.",
                    )

                documents.update_content(
                    document,
                    filename=handle.filename,
                    content_type=handle.content_type,
                    size_bytes=handle.size_bytes,
                    checksum_sha256=handle.checksum_sha256,
                )
                documents.update_status(
                    document,
                    DocumentStatus.READY,
                    chunk_count=chunk_count,
                )

                jobs.update_progress(
                    job,
                    processed_chunks=chunk_count,
                    total_chunks=chunk_count,
                )
                jobs.mark_succeeded(job)

    def fail(
        self,
        handle: IngestionHandle,
        *,
        error_message: str,
    ) -> None:
        normalized_error = error_message.strip() or "Document ingestion failed."

        with self._session_factory() as session:
            with session.begin():
                documents = DocumentRepository(session)
                jobs = IngestionJobRepository(session)

                document = documents.get(handle.document_id)
                if document is None:
                    raise RuntimeError(
                        "The ingestion document record no longer exists.",
                    )

                job = jobs.get(handle.job_id)
                if job is None:
                    raise RuntimeError(
                        "The ingestion job record no longer exists.",
                    )

                if handle.is_new_document:
                    documents.update_status(
                        document,
                        DocumentStatus.FAILED,
                        chunk_count=0,
                        error_message=normalized_error,
                    )
                else:
                    documents.update_status(
                        document,
                        handle.previous_status,
                        chunk_count=handle.previous_chunk_count,
                        error_message=handle.previous_error_message,
                    )

                jobs.mark_failed(
                    job,
                    error_message=normalized_error,
                )
