from functools import lru_cache

from backend.api.queued_ingestion_service import (
    QueuedDocumentIngestionService,
)
from backend.api.rag_service import HomelabRAGService
from backend.config import get_settings
from backend.database import get_session_factory


@lru_cache(maxsize=1)
def get_rag_service() -> HomelabRAGService:
    """
    Return one shared RAG service instance for the life of the application.

    The underlying retrieval indexes and models are loaded lazily on the
    first search request instead of being recreated for every request.
    """

    return HomelabRAGService(
        settings=get_settings(),
    )


@lru_cache(maxsize=1)
def get_queued_ingestion_service() -> QueuedDocumentIngestionService:
    settings = get_settings()

    return QueuedDocumentIngestionService(
        settings=settings,
        rag_service=get_rag_service(),
        session_factory=get_session_factory(),
    )
