from backend.indexing.interfaces import IndexVectorStore
from backend.indexing.models import IndexedCorpus, ProcessedDocument
from backend.interfaces.chunker import DocumentChunk
from backend.interfaces.embedder import EmbeddedChunk, EmbeddingProvider


class EmbeddingPipeline:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        vector_store: IndexVectorStore,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    def embed_processed_documents(
        self,
        processed_documents: list[ProcessedDocument],
    ) -> IndexedCorpus:
        documents = [p.document for p in processed_documents]

        chunks: list[DocumentChunk] = []

        for processed in processed_documents:
            chunks.extend(processed.chunks)

        embedded_chunks: list[EmbeddedChunk] = []

        if chunks:
            embedded_chunks = self._embedder.embed_many(chunks)
            self._vector_store.add_many(embedded_chunks)

        return IndexedCorpus(
            documents=documents,
            chunks=chunks,
            embedded_chunks=embedded_chunks,
            vector_store=self._vector_store,
            embedder=self._embedder,
        )
