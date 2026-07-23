from pathlib import Path
from uuid import uuid4

import pytest

from backend.ingestion.scanner import FileMetadata
from backend.interfaces.chunker import DocumentChunk
from backend.interfaces.embedder import EmbeddedChunk
from backend.interfaces.parser import ParsedDocument
from backend.storage.qdrant_vector_store import (
    QdrantVectorStore,
)


def make_embedded_chunk(
    filename: str,
    content: str,
    vector: list[float],
    chunk_index: int = 0,
) -> EmbeddedChunk:
    path = Path(filename)

    metadata = FileMetadata(
        name=filename,
        path=path,
        extension=path.suffix,
        size_bytes=len(content.encode("utf-8")),
    )

    document = ParsedDocument(
        source_path=path,
        file_type="text",
        content=content,
        metadata=metadata,
    )

    chunk = DocumentChunk(
        source_document=document,
        content=content,
        chunk_index=chunk_index,
        start_char=0,
        end_char=len(content),
    )

    return EmbeddedChunk(
        chunk=chunk,
        vector=vector,
    )


def test_qdrant_store_adds_and_searches(
    tmp_path: Path,
) -> None:
    store = QdrantVectorStore(
        storage_path=tmp_path / "qdrant",
    )

    first = make_embedded_chunk(
        filename="first.txt",
        content="First document.",
        vector=[1.0, 0.0],
    )

    second = make_embedded_chunk(
        filename="second.txt",
        content="Second document.",
        vector=[0.0, 1.0],
    )

    store.add_many(
        [
            first,
            second,
        ]
    )

    results = store.search(
        query_vector=[1.0, 0.0],
        top_k=1,
    )

    assert store.count() == 2
    assert len(results) == 1
    assert results[0].embedded_chunk.chunk.content == "First document."
    assert results[0].score == pytest.approx(1.0)

    store.close()


def test_qdrant_store_survives_restart(
    tmp_path: Path,
) -> None:
    storage_path = tmp_path / "qdrant"

    store = QdrantVectorStore(
        storage_path=storage_path,
    )

    store.add_many(
        [
            make_embedded_chunk(
                filename="persistent.txt",
                content="Persistent document.",
                vector=[1.0, 0.0],
            )
        ]
    )

    store.close()

    restored_store = QdrantVectorStore(
        storage_path=storage_path,
    )

    items = restored_store.items()

    assert restored_store.count() == 1
    assert len(items) == 1
    assert items[0].chunk.source_document.metadata.name == "persistent.txt"
    assert items[0].chunk.content == "Persistent document."

    restored_store.close()


def test_qdrant_store_replaces_old_document_chunks(
    tmp_path: Path,
) -> None:
    store = QdrantVectorStore(
        storage_path=tmp_path / "qdrant",
    )

    store.add_many(
        [
            make_embedded_chunk(
                filename="replace.txt",
                content="Old first chunk.",
                vector=[1.0, 0.0],
                chunk_index=0,
            ),
            make_embedded_chunk(
                filename="replace.txt",
                content="Old second chunk.",
                vector=[0.9, 0.1],
                chunk_index=1,
            ),
            make_embedded_chunk(
                filename="keep.txt",
                content="Keep this document.",
                vector=[0.0, 1.0],
            ),
        ]
    )

    store.replace_document(
        "replace.txt",
        [
            make_embedded_chunk(
                filename="replace.txt",
                content="New replacement chunk.",
                vector=[1.0, 0.0],
            )
        ],
    )

    items = store.items()

    assert store.count() == 2

    stored_contents = {
        (
            item.chunk.source_document.metadata.name,
            item.chunk.content,
        )
        for item in items
    }

    assert stored_contents == {
        (
            "replace.txt",
            "New replacement chunk.",
        ),
        (
            "keep.txt",
            "Keep this document.",
        ),
    }

    store.close()


def test_qdrant_store_scopes_documents_by_database_id(
    tmp_path: Path,
) -> None:
    store = QdrantVectorStore(
        storage_path=tmp_path / "qdrant",
    )

    first_document_id = uuid4()
    second_document_id = uuid4()

    store.replace_document(
        "shared.txt",
        [
            make_embedded_chunk(
                filename="shared.txt",
                content="First database document.",
                vector=[1.0, 0.0],
            )
        ],
        document_id=first_document_id,
    )

    store.replace_document(
        "shared.txt",
        [
            make_embedded_chunk(
                filename="shared.txt",
                content="Second database document.",
                vector=[0.0, 1.0],
            )
        ],
        document_id=second_document_id,
    )

    store.replace_document(
        "shared.txt",
        [
            make_embedded_chunk(
                filename="shared.txt",
                content="Updated first document.",
                vector=[0.9, 0.1],
            )
        ],
        document_id=first_document_id,
    )

    assert store.count() == 2

    stored_contents = {item.chunk.content for item in store.items()}

    assert stored_contents == {
        "Updated first document.",
        "Second database document.",
    }

    store.delete_document(
        "shared.txt",
        document_id=first_document_id,
    )

    remaining_items = store.items()

    assert store.count() == 1
    assert len(remaining_items) == 1
    assert remaining_items[0].chunk.content == "Second database document."

    store.close()


def test_qdrant_store_migrates_legacy_name_scoped_points(
    tmp_path: Path,
) -> None:
    store = QdrantVectorStore(
        storage_path=tmp_path / "qdrant",
    )

    store.add_many(
        [
            make_embedded_chunk(
                filename="legacy.txt",
                content="Legacy document version.",
                vector=[1.0, 0.0],
            )
        ]
    )

    store.replace_document(
        "legacy.txt",
        [
            make_embedded_chunk(
                filename="legacy.txt",
                content="Database-linked version.",
                vector=[1.0, 0.0],
            )
        ],
        document_id=uuid4(),
    )

    items = store.items()

    assert store.count() == 1
    assert len(items) == 1
    assert items[0].chunk.content == "Database-linked version."

    store.close()


def test_qdrant_store_rejects_invalid_top_k(
    tmp_path: Path,
) -> None:
    store = QdrantVectorStore(
        storage_path=tmp_path / "qdrant",
    )

    with pytest.raises(
        ValueError,
        match="top_k must be positive",
    ):
        store.search(
            query_vector=[1.0, 0.0],
            top_k=0,
        )

    store.close()
