from pathlib import Path

from backend.chunking.fixed_size_chunker import FixedSizeChunker
from backend.embeddings.ollama_embedder import OllamaEmbedder
from backend.ingestion.scanner import FileMetadata
from backend.interfaces.vector_store import SearchResult
from backend.parser.text_parser import TextParser
from backend.storage.in_memory_vector_store import InMemoryVectorStore


def retrieve(
    question: str,
    embedder: OllamaEmbedder,
    vector_store: InMemoryVectorStore,
    top_k: int = 3,
) -> list[SearchResult]:
    if not question.strip():
        raise ValueError("question must not be empty")

    query_vector = embedder.embed_text(question)

    return vector_store.search(
        query_vector=query_vector,
        top_k=top_k,
    )


def main() -> None:
    sample_path = Path("data/sample.txt")

    if not sample_path.exists():
        sample_path.parent.mkdir(exist_ok=True)
        sample_path.write_text(
            "Homelab AI is a local retrieval augmented generation system. "
            "It scans documents, chunks text, embeds chunks, and stores vectors. "
            "The system uses Ollama to generate embeddings locally. "
            "Cosine similarity is used to locate relevant document chunks.",
            encoding="utf-8",
        )

    parser = TextParser()
    chunker = FixedSizeChunker()
    embedder = OllamaEmbedder()
    vector_store = InMemoryVectorStore()

    file = FileMetadata(
        name=sample_path.name,
        path=sample_path,
        extension=sample_path.suffix.lower(),
        size_bytes=sample_path.stat().st_size,
    )

    parsed_document = parser.parse(file)
    chunks = chunker.chunk(parsed_document)
    embedded_chunks = embedder.embed_many(chunks)
    vector_store.add_many(embedded_chunks)

    question = "What does Homelab AI do?"
    results = retrieve(
        question=question,
        embedder=embedder,
        vector_store=vector_store,
        top_k=3,
    )

    print(f"Indexed chunks: {vector_store.count()}")
    print(f"Question: {question}")
    print("\nRetrieved results:")

    for position, result in enumerate(results, start=1):
        chunk = result.embedded_chunk.chunk

        print(f"\nResult {position}")
        print(f"Score: {result.score:.4f}")
        print(f"Source: {chunk.source_document.metadata.name}")
        print(f"Chunk: {chunk.chunk_index}")
        print(f"Content: {chunk.content}")


if __name__ == "__main__":
    main()
