from dataclasses import dataclass
from typing import Protocol

from backend.interfaces.embedder import EmbeddedChunk


@dataclass(frozen=True)
class SearchResult:
    embedded_chunk: EmbeddedChunk
    score: float


class VectorStore(Protocol):
    def add(self, embedded_chunk: EmbeddedChunk) -> None: ...

    def add_many(self, embedded_chunks: list[EmbeddedChunk]) -> None: ...

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]: ...
