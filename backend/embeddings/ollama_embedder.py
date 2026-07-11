from typing import Any

import httpx

from backend.interfaces.chunker import DocumentChunk
from backend.interfaces.embedder import EmbeddedChunk


class OllamaEmbedder:
    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        timeout: float = 30.0,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")

        if not base_url.strip():
            raise ValueError("base_url must not be empty")

        if timeout <= 0:
            raise ValueError("timeout must be positive")

        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def embed_text(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("text must not be empty")

        response = httpx.post(
            f"{self._base_url}/api/embed",
            json={
                "model": self._model,
                "input": text,
            },
            timeout=self._timeout,
        )
        response.raise_for_status()

        return self._parse_embedding(response.json())

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

    @staticmethod
    def _parse_embedding(payload: Any) -> list[float]:
        if not isinstance(payload, dict):
            raise ValueError("Ollama response must be a JSON object")

        embeddings = payload.get("embeddings")

        if not isinstance(embeddings, list) or not embeddings:
            raise ValueError("Ollama response does not contain embeddings")

        first_embedding = embeddings[0]

        if not isinstance(first_embedding, list):
            raise ValueError("Ollama embedding must be a list")

        if not all(
            isinstance(value, int | float) and not isinstance(value, bool)
            for value in first_embedding
        ):
            raise ValueError("Ollama embedding contains non-numeric values")

        return [float(value) for value in first_embedding]
