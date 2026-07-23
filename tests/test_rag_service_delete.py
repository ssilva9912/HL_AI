from collections.abc import Iterator
from hashlib import sha256
from pathlib import Path
from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.api.rag_service import (
    DocumentNotFoundError,
    HomelabRAGService,
    UnsafeDocumentPathError,
)
from backend.config import Settings
from backend.database import (
    Base,
    DocumentRepository,
    DocumentStatus,
)
from backend.ingestion.scanner import FileMetadata
from backend.interfaces.chunker import DocumentChunk
from backend.interfaces.embedder import EmbeddedChunk
from backend.interfaces.parser import ParsedDocument


@pytest.fixture
def session_factory() -> Iterator[sessionmaker[Session]]:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
    )
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
    tmp_path: Path,
) -> HomelabRAGService:
    return HomelabRAGService(
        settings=Settings(
            document_directory=tmp_path / "documents",
            vector_store_path=tmp_path / "qdrant",
            database_url=None,
        ),
    )


def create_document(
    factory: sessionmaker[Session],
    document_directory: Path,
    *,
    filename: str = "delete.txt",
    create_file: bool = True,
) -> tuple[UUID, Path, bytes]:
    document_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = document_directory / filename
    content = b"Document scheduled for deletion."

    if create_file:
        destination.write_bytes(content)

    with factory() as session:
        with session.begin():
            documents = DocumentRepository(session)

            document = documents.create(
                filename=filename,
                storage_path=str(
                    destination.resolve(),
                ),
                content_type="text/plain",
                size_bytes=len(content),
                checksum_sha256=sha256(
                    content,
                ).hexdigest(),
            )

            documents.update_status(
                document,
                DocumentStatus.READY,
                chunk_count=1,
            )

            document_id = document.id

    return document_id, destination, content


def make_embedded_chunk(
    destination: Path,
    content: bytes,
) -> EmbeddedChunk:
    decoded_content = content.decode("utf-8")

    metadata = FileMetadata(
        name=destination.name,
        path=destination,
        extension=destination.suffix,
        size_bytes=len(content),
    )

    document = ParsedDocument(
        source_path=destination,
        file_type="text",
        content=decoded_content,
        metadata=metadata,
    )

    chunk = DocumentChunk(
        source_document=document,
        content=decoded_content,
        chunk_index=0,
        start_char=0,
        end_char=len(decoded_content),
    )

    return EmbeddedChunk(
        chunk=chunk,
        vector=[1.0, 0.0],
    )


def test_delete_document_removes_database_file_and_vectors(
    tmp_path: Path,
    session_factory: sessionmaker[Session],
) -> None:
    service = build_service(tmp_path)
    document_directory = tmp_path / "documents"

    document_id, destination, content = create_document(
        session_factory,
        document_directory,
    )

    vector_store = service._get_vector_store()
    vector_store.replace_document(
        document_name=destination.name,
        embedded_chunks=[
            make_embedded_chunk(
                destination,
                content,
            )
        ],
        document_id=document_id,
    )

    service._corpus = Mock()

    with session_factory() as session:
        result = service.delete_document(
            document_id,
            session,
        )

    assert result.document_id == document_id
    assert result.document == destination.name
    assert result.deleted_chunk_count == 1

    assert not destination.exists()
    assert service._corpus is None
    assert (
        vector_store.document_items(
            destination.name,
            document_id=document_id,
        )
        == []
    )

    with session_factory() as session:
        assert (
            DocumentRepository(
                session,
            ).get(document_id)
            is None
        )

    vector_store.close()


def test_delete_document_tolerates_missing_file_and_vectors(
    tmp_path: Path,
    session_factory: sessionmaker[Session],
) -> None:
    service = build_service(tmp_path)
    document_directory = tmp_path / "documents"

    document_id, destination, _ = create_document(
        session_factory,
        document_directory,
        filename="missing.txt",
        create_file=False,
    )

    with session_factory() as session:
        result = service.delete_document(
            document_id,
            session,
        )

    assert result.deleted_chunk_count == 0
    assert not destination.exists()

    with session_factory() as session:
        assert (
            DocumentRepository(
                session,
            ).get(document_id)
            is None
        )

    service._get_vector_store().close()


def test_delete_document_restores_file_and_vectors_on_failure(
    tmp_path: Path,
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = build_service(tmp_path)
    document_directory = tmp_path / "documents"

    document_id, destination, content = create_document(
        session_factory,
        document_directory,
    )

    vector_store = service._get_vector_store()
    vector_store.replace_document(
        document_name=destination.name,
        embedded_chunks=[
            make_embedded_chunk(
                destination,
                content,
            )
        ],
        document_id=document_id,
    )

    previous_corpus = Mock()
    service._corpus = previous_corpus

    def fail_database_delete(
        repository: DocumentRepository,
        document: object,
    ) -> None:
        raise RuntimeError(
            "Database deletion failed.",
        )

    monkeypatch.setattr(
        DocumentRepository,
        "delete",
        fail_database_delete,
    )

    with session_factory() as session:
        with pytest.raises(
            RuntimeError,
            match="Database deletion failed",
        ):
            service.delete_document(
                document_id,
                session,
            )

    assert destination.read_bytes() == content
    assert service._corpus is previous_corpus

    restored_chunks = vector_store.document_items(
        destination.name,
        document_id=document_id,
    )

    assert len(restored_chunks) == 1
    assert restored_chunks[0].chunk.content == content.decode("utf-8")

    with session_factory() as session:
        document = DocumentRepository(
            session,
        ).get(document_id)

        assert document is not None
        assert document.status is DocumentStatus.READY

    vector_store.close()


def test_delete_document_returns_not_found(
    tmp_path: Path,
    session_factory: sessionmaker[Session],
) -> None:
    service = build_service(tmp_path)

    with session_factory() as session:
        with pytest.raises(
            DocumentNotFoundError,
            match="Document not found",
        ):
            service.delete_document(
                uuid4(),
                session,
            )


def test_delete_document_rejects_unmanaged_storage_path(
    tmp_path: Path,
    session_factory: sessionmaker[Session],
) -> None:
    service = build_service(tmp_path)
    outside_path = tmp_path / "outside.txt"
    outside_content = b"Outside managed storage."
    outside_path.write_bytes(outside_content)

    with session_factory() as session:
        with session.begin():
            document = DocumentRepository(
                session,
            ).create(
                filename=outside_path.name,
                storage_path=str(
                    outside_path.resolve(),
                ),
                content_type="text/plain",
                size_bytes=len(outside_content),
                checksum_sha256=sha256(
                    outside_content,
                ).hexdigest(),
            )

            document_id = document.id

    with session_factory() as session:
        with pytest.raises(
            UnsafeDocumentPathError,
            match="outside",
        ):
            service.delete_document(
                document_id,
                session,
            )

    assert outside_path.exists()
