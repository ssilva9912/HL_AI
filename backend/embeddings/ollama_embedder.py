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
        batch_size: int = 8,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")

        if not base_url.strip():
            raise ValueError("base_url must not be empty")

        if timeout <= 0:
            raise ValueError("timeout must be positive")

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._batch_size = batch_size

    def embed_text(
        self,
        text: str,
    ) -> list[float]:
        if not text.strip():
            raise ValueError("text must not be empty")

        return self._request_embeddings([text])[0]

    def embed(
        self,
        chunk: DocumentChunk,
    ) -> EmbeddedChunk:
        return EmbeddedChunk(
            chunk=chunk,
            vector=self.embed_text(chunk.content),
        )

    def embed_many(
        self,
        chunks: list[DocumentChunk],
    ) -> list[EmbeddedChunk]:
        if not chunks:
            return []

        embedded_chunks: list[EmbeddedChunk] = []

        for start_index in range(
            0,
            len(chunks),
            self._batch_size,
        ):
            batch = chunks[start_index : start_index + self._batch_size]

            vectors = self._request_embeddings([chunk.content for chunk in batch])

            embedded_chunks.extend(
                EmbeddedChunk(
                    chunk=chunk,
                    vector=vector,
                )
                for chunk, vector in zip(
                    batch,
                    vectors,
                    strict=True,
                )
            )

        return embedded_chunks

    def _request_embeddings(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        if not texts:
            return []

        if any(not text.strip() for text in texts):
            raise ValueError("embedding input must not contain empty text")

        response = httpx.post(
            f"{self._base_url}/api/embed",
            json={
                "model": self._model,
                "input": texts,
            },
            timeout=self._timeout,
        )

        response.raise_for_status()

        embeddings = self._parse_embeddings(response.json())

        if len(embeddings) != len(texts):
            raise ValueError(
                "Ollama returned a different number of embeddings "
                "than the number of requested inputs"
            )

        return embeddings

    @staticmethod
    def _parse_embedding(
        payload: Any,
    ) -> list[float]:
        return OllamaEmbedder._parse_embeddings(payload)[0]

    @staticmethod
    def _parse_embeddings(
        payload: Any,
    ) -> list[list[float]]:
        if not isinstance(payload, dict):
            raise ValueError("Ollama response must be a JSON object")

        raw_embeddings = payload.get("embeddings")

        if not isinstance(raw_embeddings, list) or not raw_embeddings:
            raise ValueError("Ollama response does not contain embeddings")

        embeddings: list[list[float]] = []

        for raw_embedding in raw_embeddings:
            if not isinstance(raw_embedding, list) or not raw_embedding:
                raise ValueError("Ollama response contains an invalid embedding")

            if not all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in raw_embedding
            ):
                raise ValueError("Ollama embedding contains non-numeric values")

            embeddings.append([float(value) for value in raw_embedding])

        return embeddings
