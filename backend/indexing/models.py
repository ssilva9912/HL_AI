from dataclasses import dataclass

from backend.indexing.interfaces import IndexVectorStore
from backend.interfaces.chunker import DocumentChunk
from backend.interfaces.embedder import EmbeddedChunk, EmbeddingProvider
from backend.interfaces.parser import ParsedDocument


@dataclass(frozen=True)
class ProcessedDocument:
    document: ParsedDocument
    chunks: list[DocumentChunk]


@dataclass(frozen=True)
class IndexedCorpus:
    documents: list[ParsedDocument]
    chunks: list[DocumentChunk]
    embedded_chunks: list[EmbeddedChunk]
    vector_store: IndexVectorStore
    embedder: EmbeddingProvider

    @property
    def document_count(self) -> int:
        return len(self.documents)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)
