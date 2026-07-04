import httpx

from backend.interfaces.chunker import DocumentChunk
from backend.interfaces.embedder import EmbeddedChunk


class OllamaEmbedder:
    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def embed(self, chunk: DocumentChunk) -> EmbeddedChunk:
        response = httpx.post(
            f"{self.base_url}/api/embed",
            json={
                "model": self.model,
                "input": chunk.content,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()
        vector = data["embeddings"][0]

        return EmbeddedChunk(chunk=chunk, vector=vector)

    def embed_many(self, chunks: list[DocumentChunk]) -> list[EmbeddedChunk]:
        return [self.embed(chunk) for chunk in chunks]
