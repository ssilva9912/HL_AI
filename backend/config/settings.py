from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="HOMELAB_",
        case_sensitive=False,
        extra="ignore",
    )

    api_host: str = Field(default="127.0.0.1")
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_url: str = Field(
        default="http://127.0.0.1:8000",
    )

    ollama_url: str = Field(
        default="http://localhost:11434",
    )
    llm_model: str = Field(default="llama3.2")
    embedding_model: str = Field(
        default="nomic-embed-text",
    )

    request_timeout: float = Field(
        default=120.0,
        gt=0,
    )
    embedding_timeout: float = Field(
        default=30.0,
        gt=0,
    )

    default_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    max_upload_bytes: int = Field(
        default=10 * 1024 * 1024,
        gt=0,
        description=("Maximum accepted document upload size in bytes."),
    )

    document_directory: Path = Field(
        default=DATA_DIR / "documents",
    )

    staging_directory: Path = Field(
        default=DATA_DIR / "staging",
        description=("Directory containing uploads awaiting ingestion."),
    )

    vector_store_path: Path = Field(
        default=DATA_DIR / "index" / "qdrant",
        description=("Directory containing the persistent local Qdrant database."),
    )

    vector_collection_name: str = Field(
        default="homelab_documents",
        min_length=1,
        description=("Qdrant collection containing document chunk embeddings."),
    )

    database_url: SecretStr | None = Field(
        default=None,
        description=("SQLAlchemy connection URL for the PostgreSQL product database."),
    )

    database_echo: bool = Field(
        default=False,
        description=("Log generated SQL statements. Keep disabled outside local debugging."),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
