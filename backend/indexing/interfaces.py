from typing import Protocol

from backend.interfaces.embedder import EmbeddedChunk
from backend.interfaces.vector_store import VectorStore


class IndexVectorStore(VectorStore, Protocol):
    def add_many(
        self,
        embedded_chunks: list[EmbeddedChunk],
    ) -> None: ...

    def count(self) -> int: ...
