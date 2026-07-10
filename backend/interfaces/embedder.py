from dataclasses import dataclass
from typing import Protocol

from backend.interfaces.chunker import DocumentChunk


@dataclass(frozen=True)
class EmbeddedChunk:
    chunk: DocumentChunk
    vector: list[float]


class EmbeddingProvider(Protocol):
    def embed_text(self, text: str) -> list[float]: ...

    def embed(self, chunk: DocumentChunk) -> EmbeddedChunk: ...

    def embed_many(self, chunks: list[DocumentChunk]) -> list[EmbeddedChunk]: ...
