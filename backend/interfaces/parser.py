from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from backend.ingestion.scanner import FileMetadata


@dataclass(frozen=True)
class ParsedDocument:
    source_path: Path
    file_type: str
    content: str
    metadata: FileMetadata


class Parser(Protocol):
    def can_parse(self, file: FileMetadata) -> bool: ...

    def parse(self, file: FileMetadata) -> ParsedDocument: ...
