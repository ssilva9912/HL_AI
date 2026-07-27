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


def test_ollama_embedder_reports_batch_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    progress: list[tuple[int, int]] = []

    def fake_post(
        url: str,
        *,
        json: dict[str, object],
        timeout: float,
    ) -> httpx.Response:
        assert url.endswith("/api/embed")
        assert timeout == 30.0
        inputs = json["input"]
        assert isinstance(inputs, list)
        calls.append(inputs)
        return httpx.Response(
            200,
            json={
                "embeddings": [[float(index), 1.0] for index, _ in enumerate(inputs)],
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    chunks = [make_chunk(f"chunk {index}") for index in range(5)]
    embedder = OllamaEmbedder(
        batch_size=2,
        progress_callback=lambda processed, total: progress.append(
            (processed, total),
        ),
    )

    embedded = embedder.embed_many(chunks)

    assert len(embedded) == 5
    assert [len(call) for call in calls] == [2, 2, 1]
    assert progress == [(2, 5), (4, 5), (5, 5)]
