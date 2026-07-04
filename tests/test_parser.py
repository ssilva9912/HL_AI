from pathlib import Path

from backend.ingestion.scanner import scan_directory
from backend.parser.code_parser import CodeParser
from backend.parser.markdown_parser import MarkdownParser
from backend.parser.text_parser import TextParser


def test_text_parser_parses_txt_file(tmp_path: Path) -> None:
    file_path = tmp_path / "note.txt"
    file_path.write_text("hello world", encoding="utf-8")

    metadata = scan_directory(tmp_path)[0]
    parser = TextParser()

    parsed = parser.parse(metadata)

    assert parser.can_parse(metadata)
    assert parsed.content == "hello world"
    assert parsed.file_type == "text"
    assert parsed.source_path == file_path


def test_markdown_parser_parses_md_file(tmp_path: Path) -> None:
    file_path = tmp_path / "README.md"
    file_path.write_text("# Title", encoding="utf-8")

    metadata = scan_directory(tmp_path)[0]
    parser = MarkdownParser()

    parsed = parser.parse(metadata)

    assert parser.can_parse(metadata)
    assert parsed.content == "# Title"
    assert parsed.file_type == "markdown"
    assert parsed.source_path == file_path


def test_code_parser_parses_python_file(tmp_path: Path) -> None:
    file_path = tmp_path / "main.py"
    file_path.write_text("print('hello')", encoding="utf-8")

    metadata = scan_directory(tmp_path)[0]
    parser = CodeParser()

    parsed = parser.parse(metadata)

    assert parser.can_parse(metadata)
    assert parsed.content == "print('hello')"
    assert parsed.file_type == "python"
    assert parsed.source_path == file_path


def test_parsers_reject_wrong_extensions(tmp_path: Path) -> None:
    file_path = tmp_path / "image.png"
    file_path.write_text("fake image", encoding="utf-8")

    metadata = scan_directory(tmp_path)[0]

    assert not TextParser().can_parse(metadata)
    assert not MarkdownParser().can_parse(metadata)
    assert not CodeParser().can_parse(metadata)
