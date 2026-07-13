import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from backend.api.dependencies import get_rag_service
from backend.api.rag_service import HomelabRAGService
from backend.api.schemas import (
    HealthResponse,
    SearchRequest,
    SearchResponse,
    SourceResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


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
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question cannot be empty.",
        )

    try:
        result = rag_service.ask(
            question=question,
            top_k=request.top_k,
        )
    except RuntimeError as exc:
        logger.exception("RAG pipeline configuration error")

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error while processing RAG request")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The question could not be processed.",
        ) from exc

    return SearchResponse(
        answer=result.answer,
        sources=[
            SourceResponse(
                text=source.text,
                score=source.score,
                document=source.document,
                chunk_id=source.chunk_id,
            )
            for source in result.sources
        ],
    )
