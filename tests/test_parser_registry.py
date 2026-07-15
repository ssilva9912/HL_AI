from pathlib import Path

import pytest

from backend.ingestion.scanner import FileMetadata
from backend.parser.markdown_parser import MarkdownParser
from backend.parser.registry import ParserRegistry
from backend.parser.text_parser import TextParser


def build_metadata(path: Path) -> FileMetadata:
    return FileMetadata(
        name=path.name,
        path=path,
        extension=path.suffix.lower(),
        size_bytes=path.stat().st_size,
    )


def test_registry_selects_text_parser(tmp_path: Path) -> None:
    path = tmp_path / "document.txt"
    path.write_text(
        "Plain text content.",
        encoding="utf-8",
    )

    registry = ParserRegistry(
        parsers=[
            TextParser(),
            MarkdownParser(),
        ]
    )

    parser = registry.get_parser(build_metadata(path))

    assert isinstance(parser, TextParser)


def test_registry_selects_markdown_parser(tmp_path: Path) -> None:
    path = tmp_path / "document.md"
    path.write_text(
        "# Markdown content",
        encoding="utf-8",
    )

    registry = ParserRegistry(
        parsers=[
            TextParser(),
            MarkdownParser(),
        ]
    )

    parser = registry.get_parser(build_metadata(path))

    assert isinstance(parser, MarkdownParser)


def test_registry_parses_text_document(tmp_path: Path) -> None:
    path = tmp_path / "document.txt"
    path.write_text(
        "Homelab AI text document.",
        encoding="utf-8",
    )

    registry = ParserRegistry(
        parsers=[
            TextParser(),
            MarkdownParser(),
        ]
    )

    document = registry.parse(build_metadata(path))

    assert document.file_type == "text"
    assert document.content == "Homelab AI text document."
    assert document.source_path == path


def test_registry_parses_markdown_document(tmp_path: Path) -> None:
    path = tmp_path / "document.md"
    path.write_text(
        "# Homelab AI",
        encoding="utf-8",
    )

    registry = ParserRegistry(
        parsers=[
            TextParser(),
            MarkdownParser(),
        ]
    )

    document = registry.parse(build_metadata(path))

    assert document.file_type == "markdown"
    assert document.content == "# Homelab AI"
    assert document.source_path == path


def test_registry_reports_supported_file(tmp_path: Path) -> None:
    path = tmp_path / "document.md"
    path.write_text(
        "# Supported",
        encoding="utf-8",
    )

    registry = ParserRegistry(
        parsers=[
            TextParser(),
            MarkdownParser(),
        ]
    )

    assert registry.can_parse(build_metadata(path)) is True


def test_registry_reports_unsupported_file(tmp_path: Path) -> None:
    path = tmp_path / "document.csv"
    path.write_text(
        "name,value",
        encoding="utf-8",
    )

    registry = ParserRegistry(
        parsers=[
            TextParser(),
            MarkdownParser(),
        ]
    )

    assert registry.can_parse(build_metadata(path)) is False


def test_registry_rejects_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "document.csv"
    path.write_text(
        "name,value",
        encoding="utf-8",
    )

    registry = ParserRegistry(
        parsers=[
            TextParser(),
            MarkdownParser(),
        ]
    )

    with pytest.raises(
        ValueError,
        match="no parser is registered for extension: .csv",
    ):
        registry.parse(build_metadata(path))


def test_registry_requires_at_least_one_parser() -> None:
    with pytest.raises(
        ValueError,
        match="at least one parser is required",
    ):
        ParserRegistry(parsers=[])
