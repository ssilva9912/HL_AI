from pathlib import Path
from typing import cast

import pytest

from backend.api.rag_service import (
    DocumentTooLargeError,
    EmptyDocumentError,
    HomelabRAGService,
    InvalidDocumentNameError,
    UnsupportedDocumentTypeError,
)
from backend.config import Settings
from backend.indexing.indexer import IndexedCorpus


class FakeCorpus:
    document_count = 1
    chunk_count = 2


class FakeIndexer:
    def __init__(self) -> None:
        self.indexed_directory: Path | None = None

    def index_directory(
        self,
        directory: Path,
        recursive: bool = True,
    ) -> IndexedCorpus:
        self.indexed_directory = directory
        return cast(IndexedCorpus, FakeCorpus())


def build_service(
    tmp_path: Path,
    max_upload_bytes: int = 100,
) -> tuple[HomelabRAGService, FakeIndexer]:
    fake_indexer = FakeIndexer()

    settings = Settings(
        document_directory=tmp_path,
        max_upload_bytes=max_upload_bytes,
    )

    service = HomelabRAGService(
        settings=settings,
        indexer_factory=lambda: fake_indexer,
    )

    return service, fake_indexer


def test_ingest_document_saves_file_and_rebuilds_corpus(
    tmp_path: Path,
) -> None:
    service, fake_indexer = build_service(tmp_path)

    result = service.ingest_document(
        filename="notes.txt",
        content=b"Homelab AI document.",
    )

    saved_path = tmp_path / "notes.txt"

    assert saved_path.read_bytes() == b"Homelab AI document."
    assert fake_indexer.indexed_directory == tmp_path

    assert result.document == "notes.txt"
    assert result.size_bytes == 20
    assert result.document_count == 1
    assert result.chunk_count == 2
    assert result.status == "indexed"


@pytest.mark.parametrize(
    "filename",
    [
        "",
        "   ",
        "../notes.txt",
        r"..\notes.txt",
        "folder/notes.txt",
        r"folder\notes.txt",
    ],
)
def test_ingest_rejects_invalid_filename(
    tmp_path: Path,
    filename: str,
) -> None:
    service, _ = build_service(tmp_path)

    with pytest.raises(InvalidDocumentNameError):
        service.ingest_document(
            filename=filename,
            content=b"content",
        )


def test_ingest_rejects_unsupported_extension(
    tmp_path: Path,
) -> None:
    service, _ = build_service(tmp_path)

    with pytest.raises(
        UnsupportedDocumentTypeError,
        match="Unsupported document type",
    ):
        service.ingest_document(
            filename="data.csv",
            content=b"name,value",
        )


def test_ingest_rejects_empty_document(
    tmp_path: Path,
) -> None:
    service, _ = build_service(tmp_path)

    with pytest.raises(
        EmptyDocumentError,
        match="cannot be empty",
    ):
        service.ingest_document(
            filename="empty.txt",
            content=b"",
        )


def test_ingest_rejects_document_over_size_limit(
    tmp_path: Path,
) -> None:
    service, _ = build_service(
        tmp_path,
        max_upload_bytes=5,
    )

    with pytest.raises(
        DocumentTooLargeError,
        match="size limit",
    ):
        service.ingest_document(
            filename="large.txt",
            content=b"123456",
        )
