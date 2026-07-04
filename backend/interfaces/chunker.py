from dataclasses import dataclass
from typing import Protocol

from backend.interfaces.parser import ParsedDocument


@dataclass(frozen=True)
class DocumentChunk:
    source_document: ParsedDocument
    content: str
    chunk_index: int
    start_char: int
    end_char: int


class Chunker(Protocol):
    def chunk(self, document: ParsedDocument) -> list[DocumentChunk]: ...
