import logging
from typing import Annotated
from uuid import UUID

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)
from sqlalchemy.orm import Session

from backend.api.dependencies import (
    get_queued_ingestion_service,
)
from backend.api.ingestion_schemas import (
    IngestionJobResponse,
    QueuedIngestionResponse,
)
from backend.api.queued_ingestion_service import (
    QueuedDocumentIngestionService,
)
from backend.api.rag_service import (
    DocumentTooLargeError,
    EmptyDocumentError,
    InvalidDocumentNameError,
    UnsupportedDocumentTypeError,
)
from backend.database import (
    IngestionJobRepository,
    get_database_session,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/documents",
    response_model=QueuedIngestionResponse,
    tags=["documents"],
    status_code=status.HTTP_202_ACCEPTED,
)
def queue_document(
    background_tasks: BackgroundTasks,
    file: Annotated[
        UploadFile,
        File(
            description=("Text, Markdown, or text-based PDF document."),
        ),
    ],
    ingestion_service: Annotated[
        QueuedDocumentIngestionService,
        Depends(get_queued_ingestion_service),
    ],
) -> QueuedIngestionResponse:
    filename = file.filename or ""

    try:
        content = file.file.read(
            ingestion_service.max_upload_bytes + 1,
        )

        queued = ingestion_service.enqueue_document(
            filename=filename,
            content=content,
        )
    except InvalidDocumentNameError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    except UnsupportedDocumentTypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        ) from exc
    except EmptyDocumentError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    except DocumentTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        logger.exception(
            "Queued ingestion is unavailable",
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception(
            "Unexpected queued ingestion error",
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The document could not be queued.",
        ) from exc
    finally:
        file.file.close()

    background_tasks.add_task(
        ingestion_service.process_job,
        queued.job_id,
    )

    logger.info(
        "Document queued document=%s job_id=%s size_bytes=%d",
        filename,
        queued.job_id,
        queued.size_bytes,
    )

    return QueuedIngestionResponse(
        document=filename,
        size_bytes=queued.size_bytes,
        document_id=queued.document_id,
        job_id=queued.job_id,
        operation=queued.operation.value,
        status="queued",
        status_url=(f"/ingestion-jobs/{queued.job_id}"),
    )


@router.get(
    "/ingestion-jobs/{job_id}",
    response_model=IngestionJobResponse,
    tags=["ingestion"],
    status_code=status.HTTP_200_OK,
)
def get_ingestion_job(
    job_id: UUID,
    session: Annotated[
        Session,
        Depends(get_database_session),
    ],
) -> IngestionJobResponse:
    job = IngestionJobRepository(session).get(job_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ingestion job not found.",
        )

    return IngestionJobResponse(
        id=job.id,
        document_id=job.document_id,
        operation=job.operation.value,
        status=job.status.value,
        attempt_count=job.attempt_count,
        processed_chunks=job.processed_chunks,
        total_chunks=job.total_chunks,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        updated_at=job.updated_at,
    )
