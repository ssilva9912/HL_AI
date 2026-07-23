import logging
from time import perf_counter
from typing import Annotated
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)
from sqlalchemy.orm import Session

from backend.api.dependencies import get_rag_service
from backend.api.rag_service import (
    DocumentNotFoundError,
    DocumentTooLargeError,
    EmptyDocumentError,
    HomelabRAGService,
    InvalidDocumentNameError,
    UnsafeDocumentPathError,
    UnsupportedDocumentTypeError,
)
from backend.api.schemas import (
    DocumentResponse,
    EvaluationMetricsResponse,
    EvaluationRequest,
    EvaluationResponse,
    HealthResponse,
    IngestionJobResponse,
    IngestResponse,
    QuestionEvaluationResponse,
    SearchMetadata,
    SearchRequest,
    SearchResponse,
    SourceResponse,
)
from backend.database import (
    Document,
    DocumentRepository,
    IngestionJob,
    IngestionJobRepository,
    get_database_session,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _document_response(
    document: Document,
) -> DocumentResponse:
    return DocumentResponse(
        id=document.id,
        filename=document.filename,
        content_type=document.content_type,
        size_bytes=document.size_bytes,
        checksum_sha256=document.checksum_sha256,
        status=document.status.value,
        chunk_count=document.chunk_count,
        error_message=document.error_message,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


def _ingestion_job_response(
    job: IngestionJob,
) -> IngestionJobResponse:
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


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["system"],
)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="homelab-ai",
    )


@router.get(
    "/documents",
    response_model=list[DocumentResponse],
    tags=["documents"],
    status_code=status.HTTP_200_OK,
)
def list_documents(
    session: Annotated[
        Session,
        Depends(get_database_session),
    ],
    offset: Annotated[
        int,
        Query(ge=0),
    ] = 0,
    limit: Annotated[
        int,
        Query(ge=1, le=1000),
    ] = 100,
) -> list[DocumentResponse]:
    documents = DocumentRepository(
        session,
    ).list_all(
        offset=offset,
        limit=limit,
    )

    return [_document_response(document) for document in documents]


@router.get(
    "/documents/{document_id}",
    response_model=DocumentResponse,
    tags=["documents"],
    status_code=status.HTTP_200_OK,
)
def get_document(
    document_id: UUID,
    session: Annotated[
        Session,
        Depends(get_database_session),
    ],
) -> DocumentResponse:
    document = DocumentRepository(
        session,
    ).get(document_id)

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    return _document_response(document)


@router.get(
    "/documents/{document_id}/jobs",
    response_model=list[IngestionJobResponse],
    tags=["documents"],
    status_code=status.HTTP_200_OK,
)
def list_document_jobs(
    document_id: UUID,
    session: Annotated[
        Session,
        Depends(get_database_session),
    ],
    offset: Annotated[
        int,
        Query(ge=0),
    ] = 0,
    limit: Annotated[
        int,
        Query(ge=1, le=1000),
    ] = 100,
) -> list[IngestionJobResponse]:
    document = DocumentRepository(
        session,
    ).get(document_id)

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    jobs = IngestionJobRepository(
        session,
    ).list_for_document(
        document_id,
        offset=offset,
        limit=limit,
    )

    return [_ingestion_job_response(job) for job in jobs]


@router.delete(
    "/documents/{document_id}",
    tags=["documents"],
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
def delete_document(
    document_id: UUID,
    session: Annotated[
        Session,
        Depends(get_database_session),
    ],
    rag_service: Annotated[
        HomelabRAGService,
        Depends(get_rag_service),
    ],
) -> Response:
    try:
        result = rag_service.delete_document(
            document_id,
            session,
        )
    except DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        ) from exc
    except UnsafeDocumentPathError as exc:
        logger.exception(
            "Document deletion rejected because the storage path is unmanaged",
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=("The document could not be deleted safely."),
        ) from exc
    except Exception as exc:
        logger.exception(
            "Unexpected document deletion error",
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=("The document could not be deleted."),
        ) from exc

    logger.info(
        "Document deleted document_id=%s document=%s deleted_chunk_count=%d",
        result.document_id,
        result.document,
        result.deleted_chunk_count,
    )

    return Response(
        status_code=status.HTTP_204_NO_CONTENT,
    )


@router.post(
    "/search",
    response_model=SearchResponse,
    tags=["rag"],
    status_code=status.HTTP_200_OK,
)
def search(
    request: SearchRequest,
    rag_service: Annotated[
        HomelabRAGService,
        Depends(get_rag_service),
    ],
) -> SearchResponse:
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=(status.HTTP_422_UNPROCESSABLE_CONTENT),
            detail="Question cannot be empty.",
        )

    started_at = perf_counter()

    try:
        result = rag_service.ask(
            question=question,
            top_k=request.top_k,
        )
    except RuntimeError as exc:
        logger.exception(
            "RAG pipeline configuration error",
        )

        raise HTTPException(
            status_code=(status.HTTP_503_SERVICE_UNAVAILABLE),
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception(
            "Unexpected error while processing RAG request",
        )

        raise HTTPException(
            status_code=(status.HTTP_500_INTERNAL_SERVER_ERROR),
            detail=("The question could not be processed."),
        ) from exc

    elapsed_ms = (perf_counter() - started_at) * 1_000

    sources = [
        SourceResponse(
            text=source.text,
            score=source.score,
            document=source.document,
            chunk_id=source.chunk_id,
        )
        for source in result.sources
    ]

    logger.info(
        "RAG search completed question_length=%d top_k=%d source_count=%d elapsed_ms=%.2f",
        len(question),
        request.top_k,
        len(sources),
        elapsed_ms,
    )

    return SearchResponse(
        answer=result.answer,
        sources=sources,
        metadata=SearchMetadata(
            top_k=request.top_k,
            source_count=len(sources),
            elapsed_ms=round(
                elapsed_ms,
                2,
            ),
        ),
    )


@router.post(
    "/evaluate",
    response_model=EvaluationResponse,
    tags=["evaluation"],
    status_code=status.HTTP_200_OK,
)
def evaluate(
    request: EvaluationRequest,
    rag_service: Annotated[
        HomelabRAGService,
        Depends(get_rag_service),
    ],
) -> EvaluationResponse:
    started_at = perf_counter()

    try:
        report = rag_service.evaluate(
            top_k=request.top_k,
        )
    except FileNotFoundError as exc:
        logger.exception(
            "Evaluation dataset was not found",
        )

        raise HTTPException(
            status_code=(status.HTTP_500_INTERNAL_SERVER_ERROR),
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        logger.exception(
            "Evaluation configuration is invalid",
        )

        raise HTTPException(
            status_code=(status.HTTP_422_UNPROCESSABLE_CONTENT),
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        logger.exception(
            "Evaluation pipeline configuration error",
        )

        raise HTTPException(
            status_code=(status.HTTP_503_SERVICE_UNAVAILABLE),
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception(
            "Unexpected error while running retrieval evaluation",
        )

        raise HTTPException(
            status_code=(status.HTTP_500_INTERNAL_SERVER_ERROR),
            detail=("The retrieval evaluation could not be completed."),
        ) from exc

    elapsed_ms = (perf_counter() - started_at) * 1_000

    logger.info(
        "Retrieval evaluation completed question_count=%d top_k=%d elapsed_ms=%.2f",
        report.metrics.question_count,
        request.top_k,
        elapsed_ms,
    )

    return EvaluationResponse(
        metrics=EvaluationMetricsResponse(
            question_count=(report.metrics.question_count),
            hit_at_1=report.metrics.hit_at_1,
            hit_at_5=report.metrics.hit_at_5,
            precision_at_5=(report.metrics.precision_at_5),
            recall_at_5=(report.metrics.recall_at_5),
            mean_reciprocal_rank=(report.metrics.mean_reciprocal_rank),
        ),
        questions=[
            QuestionEvaluationResponse(
                question=result.question,
                relevant_documents=list(
                    result.relevant_documents,
                ),
                retrieved_documents=list(
                    result.retrieved_documents,
                ),
                hit_at_1=result.hit_at_1,
                hit_at_5=result.hit_at_5,
                precision_at_5=(result.precision_at_5),
                recall_at_5=result.recall_at_5,
                reciprocal_rank=(result.reciprocal_rank),
            )
            for result in report.questions
        ],
        top_k=request.top_k,
        elapsed_ms=round(
            elapsed_ms,
            2,
        ),
    )


@router.post(
    "/ingest",
    response_model=IngestResponse,
    tags=["documents"],
    status_code=status.HTTP_201_CREATED,
)
def ingest(
    file: Annotated[
        UploadFile,
        File(
            description=("Text, Markdown, or text-based PDF document."),
        ),
    ],
    rag_service: Annotated[
        HomelabRAGService,
        Depends(get_rag_service),
    ],
) -> IngestResponse:
    filename = file.filename or ""

    try:
        content = file.file.read(
            rag_service.max_upload_bytes + 1,
        )

        result = rag_service.ingest_document(
            filename=filename,
            content=content,
        )
    except InvalidDocumentNameError as exc:
        raise HTTPException(
            status_code=(status.HTTP_422_UNPROCESSABLE_CONTENT),
            detail=str(exc),
        ) from exc
    except UnsupportedDocumentTypeError as exc:
        raise HTTPException(
            status_code=(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE),
            detail=str(exc),
        ) from exc
    except EmptyDocumentError as exc:
        raise HTTPException(
            status_code=(status.HTTP_422_UNPROCESSABLE_CONTENT),
            detail=str(exc),
        ) from exc
    except DocumentTooLargeError as exc:
        raise HTTPException(
            status_code=(status.HTTP_413_CONTENT_TOO_LARGE),
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        logger.exception(
            "Document could not be parsed",
        )

        raise HTTPException(
            status_code=(status.HTTP_422_UNPROCESSABLE_CONTENT),
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception(
            "Unexpected document ingestion error",
        )

        raise HTTPException(
            status_code=(status.HTTP_500_INTERNAL_SERVER_ERROR),
            detail=("The document could not be ingested."),
        ) from exc
    finally:
        file.file.close()

    logger.info(
        "Document ingested document=%s size_bytes=%d document_count=%d chunk_count=%d",
        result.document,
        result.size_bytes,
        result.document_count,
        result.chunk_count,
    )

    return IngestResponse(
        document=result.document,
        size_bytes=result.size_bytes,
        document_count=result.document_count,
        chunk_count=result.chunk_count,
        status=result.status,
    )
