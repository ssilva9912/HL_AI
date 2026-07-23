from collections.abc import Iterator
from hashlib import sha256
from pathlib import Path
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.api.rag_service import HomelabRAGService
from backend.config import Settings
from backend.database import (
    Base,
    DocumentRepository,
    DocumentStatus,
    IngestionJobRepository,
    IngestionJobStatus,
    IngestionLifecycle,
)


@pytest.fixture
def session_factory() -> Iterator[sessionmaker[Session]]:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    factory = sessionmaker(
        bind=engine,
        class_=Session,
        expire_on_commit=False,
    )

    Base.metadata.create_all(engine)

    try:
        yield factory
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


def build_service(
    *,
    tmp_path: Path,
    session_factory: sessionmaker[Session],
    indexer: Mock,
) -> HomelabRAGService:
    settings = Settings(
        document_directory=tmp_path / "documents",
        vector_store_path=tmp_path / "qdrant",
        database_url=None,
    )

    return HomelabRAGService(
        settings=settings,
        indexer_factory=lambda: indexer,
        ingestion_lifecycle=IngestionLifecycle(
            session_factory,
        ),
    )


def test_ingestion_writes_document_and_job_records(
    tmp_path: Path,
    session_factory: sessionmaker[Session],
) -> None:
    indexed_corpus = Mock()
    indexed_corpus.document_count = 1
    indexed_corpus.chunk_count = 2

    indexer = Mock()
    indexer.index_directory.return_value = indexed_corpus

    service = build_service(
        tmp_path=tmp_path,
        session_factory=session_factory,
        indexer=indexer,
    )

    content = b"Persistent document ingestion."

    result = service.ingest_document(
        filename="persistent.txt",
        content=content,
    )

    assert result.status == "indexed"
    assert result.document_count == 1
    assert result.chunk_count == 2

    with session_factory() as session:
        documents = DocumentRepository(session).list_all()

        assert len(documents) == 1

        document = documents[0]

        assert document.filename == "persistent.txt"
        assert document.status is DocumentStatus.READY
        assert document.size_bytes == len(content)
        assert document.chunk_count == 2
        assert document.checksum_sha256 == sha256(content).hexdigest()

        jobs = IngestionJobRepository(
            session,
        ).list_for_document(document.id)

        assert len(jobs) == 1
        assert jobs[0].status is IngestionJobStatus.SUCCEEDED
        assert jobs[0].processed_chunks == 2
        assert jobs[0].total_chunks == 2


def test_failed_ingestion_writes_failed_state(
    tmp_path: Path,
    session_factory: sessionmaker[Session],
) -> None:
    indexer = Mock()
    indexer.index_directory.side_effect = RuntimeError(
        "Embedding service unavailable.",
    )

    service = build_service(
        tmp_path=tmp_path,
        session_factory=session_factory,
        indexer=indexer,
    )

    with pytest.raises(
        RuntimeError,
        match="Embedding service unavailable",
    ):
        service.ingest_document(
            filename="broken.txt",
            content=b"This upload will fail.",
        )

    assert not (tmp_path / "documents" / "broken.txt").exists()

    with session_factory() as session:
        documents = DocumentRepository(session).list_all()

        assert len(documents) == 1

        document = documents[0]

        assert document.status is DocumentStatus.FAILED
        assert document.chunk_count == 0
        assert document.error_message == "Embedding service unavailable."

        jobs = IngestionJobRepository(
            session,
        ).list_for_document(document.id)

        assert len(jobs) == 1
        assert jobs[0].status is IngestionJobStatus.FAILED
        assert jobs[0].error_message == "Embedding service unavailable."
