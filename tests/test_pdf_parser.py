from pathlib import Path

import pytest

import backend.parser.pdf_parser as pdf_parser_module
from backend.ingestion.scanner import FileMetadata
from backend.parser.pdf_parser import PdfParser


class FakePage:
    def __init__(self, text: str | None) -> None:
        self._text = text

    def extract_text(self) -> str | None:
        return self._text


class FakePdfReader:
    pages: list[FakePage] = []

    def __init__(self, path: Path) -> None:
        self.path = path
        self.pages = type(self).pages


def build_metadata(path: Path) -> FileMetadata:
    return FileMetadata(
        name=path.name,
        path=path,
        extension=path.suffix.lower(),
        size_bytes=path.stat().st_size,
    )


def install_fake_reader(
    monkeypatch: pytest.MonkeyPatch,
    page_texts: list[str | None],
) -> None:
    FakePdfReader.pages = [FakePage(text) for text in page_texts]

    monkeypatch.setattr(
        pdf_parser_module,
        "PdfReader",
        FakePdfReader,
    )


def test_pdf_parser_supports_pdf_extension(tmp_path: Path) -> None:
    path = tmp_path / "document.pdf"
    path.write_bytes(b"%PDF-test")

    parser = PdfParser()

    assert parser.can_parse(build_metadata(path)) is True


def test_pdf_parser_rejects_other_extensions(tmp_path: Path) -> None:
    path = tmp_path / "document.txt"
    path.write_text("Text document.", encoding="utf-8")

    parser = PdfParser()

    assert parser.can_parse(build_metadata(path)) is False


def test_pdf_parser_extracts_text_from_pages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "document.pdf"
    path.write_bytes(b"%PDF-test")

    install_fake_reader(
        monkeypatch,
        [
            "First page content.",
            "Second page content.",
        ],
    )

    parser = PdfParser()
    document = parser.parse(build_metadata(path))

    assert document.source_path == path
    assert document.file_type == "pdf"
    assert document.content == "First page content.\n\nSecond page content."
    assert document.metadata.name == "document.pdf"


def test_pdf_parser_skips_blank_pages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "document.pdf"
    path.write_bytes(b"%PDF-test")

    install_fake_reader(
        monkeypatch,
        [
            "First page.",
            None,
            "",
            "   ",
            "Last page.",
        ],
    )

    parser = PdfParser()
    document = parser.parse(build_metadata(path))

    assert document.content == "First page.\n\nLast page."


def test_pdf_parser_rejects_pdf_without_extractable_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "scanned.pdf"
    path.write_bytes(b"%PDF-test")

    install_fake_reader(
        monkeypatch,
        [
            None,
            "",
            "   ",
        ],
    )

    parser = PdfParser()

    with pytest.raises(
        ValueError,
        match="PDF contains no extractable text",
    ):
        parser.parse(build_metadata(path))
