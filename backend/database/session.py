from collections.abc import Generator
from functools import lru_cache

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from backend.config import get_settings


class DatabaseNotConfiguredError(RuntimeError):
    """Raised when database access is attempted without a configured URL."""


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    settings = get_settings()

    if settings.database_url is None:
        raise DatabaseNotConfiguredError(
            "HOMELAB_DATABASE_URL is not configured.",
        )

    return create_engine(
        settings.database_url.get_secret_value(),
        pool_pre_ping=True,
        echo=settings.database_echo,
    )


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(
        bind=get_engine(),
        class_=Session,
        autoflush=False,
        expire_on_commit=False,
    )


def get_database_session() -> Generator[Session, None, None]:
    session = get_session_factory()()

    try:
        yield session
    finally:
        session.close()


def check_database_connection() -> None:
    with get_engine().connect() as connection:
        connection.execute(text("SELECT 1"))
