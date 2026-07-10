from dataclasses import dataclass
from typing import Protocol

from backend.interfaces.chunker import DocumentChunk


@dataclass(frozen=True)
class RetrievalResult:
    chunk: DocumentChunk
    score: float
    retriever: str


class Retriever(Protocol):
    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]: ...
