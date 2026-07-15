from functools import lru_cache

from backend.api.rag_service import HomelabRAGService
from backend.config import get_settings


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
