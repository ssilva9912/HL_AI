from collections.abc import Iterator
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
from backend.ingestion.scanner import FileMetadata
from backend.interfaces.chunker import DocumentChunk
from backend.interfaces.embedder import EmbeddedChunk
from backend.interfaces.parser import ParsedDocument


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


def make_embedded_chunk(
    document_path: Path,
    content: str,
) -> EmbeddedChunk:
    metadata = FileMetadata(
        name=document_path.name,
        path=document_path,
        extension=document_path.suffix,
        size_bytes=len(content.encode("utf-8")),
    )

    document = ParsedDocument(
        source_path=document_path,
        file_type="text",
        content=content,
        metadata=metadata,
    )

    chunk = DocumentChunk(
        source_document=document,
        content=content,
        chunk_index=0,
        start_char=0,
        end_char=len(content),
    )

    return EmbeddedChunk(
        chunk=chunk,
        vector=[1.0, 0.0],
    )


def test_new_document_failure_removes_qdrant_points(
    tmp_path: Path,
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document_directory = tmp_path / "documents"
    vector_store_path = tmp_path / "qdrant"

    settings = Settings(
        document_directory=document_directory,
        vector_store_path=vector_store_path,
        database_url=None,
    )

    service = HomelabRAGService(
        settings=settings,
        ingestion_lifecycle=IngestionLifecycle(
            session_factory,
        ),
    )

    filename = "rollback.txt"
    content = b"Document that must be rolled back."
    destination = document_directory / filename

    staged_corpus = Mock()
    staged_corpus.embedded_chunks = [
        make_embedded_chunk(
            destination,
            content.decode("utf-8"),
        )
    ]
    staged_corpus.chunk_count = 1

    fake_indexer = Mock()
    fake_indexer.index_file.return_value = staged_corpus

    monkeypatch.setattr(
        "backend.api.rag_service.Indexer",
        lambda **_: fake_indexer,
    )

    load_persisted_corpus = Mock(
        side_effect=[
            None,
            RuntimeError("Persisted corpus reload failed."),
        ],
    )

    monkeypatch.setattr(
        service,
        "_load_persisted_corpus",
        load_persisted_corpus,
    )

    with pytest.raises(
        RuntimeError,
        match="Persisted corpus reload failed",
    ):
        service.ingest_document(
            filename=filename,
            content=content,
        )

    assert not destination.exists()

    vector_store = service._get_vector_store()

    try:
        assert vector_store.count() == 0
    finally:
        vector_store.close()

    with session_factory() as session:
        documents = DocumentRepository(session).list_all()

        assert len(documents) == 1

        document = documents[0]

        assert document.filename == filename
        assert document.status is DocumentStatus.FAILED
        assert document.chunk_count == 0
        assert document.error_message == "Persisted corpus reload failed."

        jobs = IngestionJobRepository(
            session,
        ).list_for_document(document.id)

        assert len(jobs) == 1
        assert jobs[0].status is IngestionJobStatus.FAILED
        assert jobs[0].error_message == "Persisted corpus reload failed."
