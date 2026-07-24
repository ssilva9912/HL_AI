import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.api.dependencies import (
    get_queued_ingestion_service,
)
from backend.api.ingestion_routes import (
    router as ingestion_router,
)
from backend.api.routes import router as api_router

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s - %(message)s",
)

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


@asynccontextmanager
async def lifespan(
    _: FastAPI,
) -> AsyncIterator[None]:
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


def create_app() -> FastAPI:
    application = FastAPI(
        title="Homelab AI",
        description="Local retrieval-augmented generation API.",
        version="0.1.0",
        lifespan=lifespan,
    )

    application.include_router(api_router)
    application.include_router(ingestion_router)

    return application


app = create_app()
