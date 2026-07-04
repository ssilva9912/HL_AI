from pathlib import Path

import httpx
import pytest

from backend.embeddings.ollama_embedder import OllamaEmbedder
from backend.ingestion.scanner import FileMetadata
from backend.interfaces.chunker import DocumentChunk
from backend.interfaces.parser import ParsedDocument


def make_chunk(content: str) -> DocumentChunk:
    metadata = FileMetadata(
        path=Path("test.txt"),
        name="test.txt",
        size_bytes=len(content),
        extension=".txt",
    )

    document = ParsedDocument(
        source_path=Path("test.txt"),
        file_type="text",
        content=content,
        metadata=metadata,
    )

    return DocumentChunk(
        source_document=document,
        content=content,
        chunk_index=0,
        start_char=0,
        end_char=len(content),
    )


def ollama_available() -> bool:
    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


@pytest.mark.skipif(not ollama_available(), reason="Ollama is not running")
def test_ollama_embedder_embeds_chunk() -> None:
    chunk = make_chunk("hello world")
    embedder = OllamaEmbedder(timeout=120.0)

    embedded = embedder.embed(chunk)

    assert embedded.chunk == chunk
    assert len(embedded.vector) > 0
    assert all(isinstance(value, float) for value in embedded.vector)
