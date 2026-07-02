from pathlib import Path

import pytest

from backend.ingestion.scanner import scan_directory


def test_scan_directory_finds_file(tmp_path: Path) -> None:
    test_file = tmp_path / "note.txt"
    test_file.write_text("hello", encoding="utf-8")

    results = scan_directory(tmp_path)

    assert len(results) == 1
    assert results[0].name == "note.txt"
    assert results[0].extension == ".txt"
    assert results[0].size_bytes == 5


def test_scan_directory_missing_path_raises_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        scan_directory(missing_path)
