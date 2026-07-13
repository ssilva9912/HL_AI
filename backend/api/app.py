import logging

from fastapi import FastAPI

from backend.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s - %(message)s",
)


def create_app() -> FastAPI:
    application = FastAPI(
        title="Homelab AI",
        description="Local retrieval-augmented generation API.",
        version="0.1.0",
    )

    application.include_router(router)

    return application


app = create_app()
