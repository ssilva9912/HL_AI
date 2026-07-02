from dataclasses import dataclass
from pathlib import Path

from backend.config.settings import EXCLUDED_DIRS
from backend.logging.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class FileMetadata:
    name: str
    path: Path
    extension: str
    size_bytes: int


def should_exclude(path: Path) -> bool:
    return any(part in EXCLUDED_DIRS for part in path.parts)


def scan_directory(root: Path) -> list[FileMetadata]:
    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root}")

    files: list[FileMetadata] = []

    for path in root.rglob("*"):
        if should_exclude(path):
            continue

        if path.is_file():
            files.append(
                FileMetadata(
                    name=path.name,
                    path=path,
                    extension=path.suffix.lower(),
                    size_bytes=path.stat().st_size,
                )
            )

    logger.info("Scanned %s and found %d files", root, len(files))
    return files
