import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.api.conversation_routes import (
    router as conversation_router,
)
from backend.api.dependencies import (
    get_queued_ingestion_service,
    get_rag_service,
)
from backend.api.ingestion_routes import (
    router as ingestion_router,
)
from backend.api.rag_service import HomelabRAGService
from backend.api.routes import router as api_router
from backend.config import get_settings
from backend.observability import configure_logging
from backend.observability.http import ObservabilityMiddleware, metrics_router

configure_logging(get_settings().log_format)

logger = logging.getLogger(__name__)


def _recover_pending_ingestion() -> None:
    try:
        attempted_count = get_queued_ingestion_service().recover_pending_jobs()
    except Exception:
        logger.exception(
            "Failed to recover pending ingestion jobs during startup",
        )
        return

    if attempted_count:
        logger.info(
            "Startup ingestion recovery finished attempted_jobs=%d",
            attempted_count,
        )


def _warm_up_rag_service(rag_service: HomelabRAGService) -> None:
    try:
        rag_service.warm_up()
    except Exception:
        logger.exception("RAG warm-up failed; the first question will pay start-up costs")


def _start_warm_up() -> asyncio.Task[None] | None:
    if not get_settings().warmup_on_startup:
        return None

    # Create the shared service here, before any worker thread asks for it.
    # get_rag_service() is an unlocked lru_cache, so two threads calling it for
    # the first time could build two services; ingestion would then invalidate
    # a different instance's caches than the one answering questions.
    try:
        rag_service = get_rag_service()
    except Exception:
        logger.exception("Could not create the RAG service; skipping warm-up")
        return None

    return asyncio.create_task(
        asyncio.to_thread(_warm_up_rag_service, rag_service),
        name="rag-warm-up",
    )


@asynccontextmanager
async def lifespan(
    _: FastAPI,
) -> AsyncIterator[None]:
    warm_up_task = _start_warm_up()
    recovery_task = asyncio.create_task(
        asyncio.to_thread(
            _recover_pending_ingestion,
        ),
        name="recover-pending-ingestion",
    )

    try:
        yield
    finally:
        await recovery_task
        if warm_up_task is not None:
            await warm_up_task


def create_app() -> FastAPI:
    application = FastAPI(
        title="Homelab AI",
        description="Local retrieval-augmented generation API.",
        version="0.1.0",
        lifespan=lifespan,
    )

    application.include_router(api_router)
    application.include_router(ingestion_router)
    application.include_router(conversation_router)

    if get_settings().metrics_enabled:
        application.include_router(metrics_router)
    application.add_middleware(ObservabilityMiddleware)

    return application


app = create_app()
