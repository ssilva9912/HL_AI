from pathlib import Path

import pytest

from backend.ingestion.scanner import FileMetadata
from backend.interfaces.chunker import DocumentChunk
from backend.interfaces.embedder import EmbeddedChunk
from backend.interfaces.parser import ParsedDocument
from backend.retrieval.dense_retriever import DenseRetriever
from backend.storage.in_memory_vector_store import InMemoryVectorStore


class FakeEmbedder:
    def embed_text(self, text: str) -> list[float]:
        if text == "local AI":
            return [1.0, 0.0]

        return [0.0, 1.0]

    def embed(self, chunk: DocumentChunk) -> EmbeddedChunk:
        return EmbeddedChunk(
            chunk=chunk,
            vector=self.embed_text(chunk.content),
        )

    def embed_many(
        self,
        chunks: list[DocumentChunk],
    ) -> list[EmbeddedChunk]:
        return [self.embed(chunk) for chunk in chunks]


def create_chunk(
    content: str,
    chunk_index: int,
) -> DocumentChunk:
    source_path = Path("data/test.txt")

    metadata = FileMetadata(
        name="test.txt",
        path=source_path,
        extension=".txt",
        size_bytes=len(content),
    )

    document = ParsedDocument(
        source_path=source_path,
        file_type="text",
        content=content,
        metadata=metadata,
    )

    return DocumentChunk(
        source_document=document,
        content=content,
        chunk_index=chunk_index,
        start_char=0,
        end_char=len(content),
    )


def test_dense_retriever_returns_semantic_match() -> None:
    local_chunk = create_chunk(
        "Homelab AI runs locally.",
        chunk_index=0,
    )

    garden_chunk = create_chunk(
        "Tomatoes require sunlight.",
        chunk_index=1,
    )

    store = InMemoryVectorStore()
    store.add_many(
        [
            EmbeddedChunk(
                chunk=local_chunk,
                vector=[1.0, 0.0],
            ),
            EmbeddedChunk(
                chunk=garden_chunk,
                vector=[0.0, 1.0],
            ),
        ]
    )

    retriever = DenseRetriever(
        embedder=FakeEmbedder(),
        vector_store=store,
    )

    results = retriever.search(
        query="local AI",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].chunk.chunk_index == 0
    assert results[0].retriever == "dense"
    assert results[0].score == pytest.approx(1.0)


def test_dense_retriever_rejects_empty_query() -> None:
    retriever = DenseRetriever(
        embedder=FakeEmbedder(),
        vector_store=InMemoryVectorStore(),
    )

    with pytest.raises(
        ValueError,
        match="query must not be empty",
    ):
        retriever.search("   ")


def test_dense_retriever_rejects_invalid_top_k() -> None:
    retriever = DenseRetriever(
        embedder=FakeEmbedder(),
        vector_store=InMemoryVectorStore(),
    )

    with pytest.raises(
        ValueError,
        match="top_k must be positive",
    ):
        retriever.search("local AI", top_k=0)
