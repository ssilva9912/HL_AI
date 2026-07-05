from pathlib import Path

from backend.chunking.fixed_size_chunker import FixedSizeChunker
from backend.embeddings.ollama_embedder import OllamaEmbedder
from backend.ingestion.scanner import FileMetadata
from backend.parser.text_parser import TextParser
from backend.storage.in_memory_vector_store import InMemoryVectorStore


def main() -> None:
    sample_path = Path("data/sample.txt")

    if not sample_path.exists():
        sample_path.parent.mkdir(exist_ok=True)
        sample_path.write_text(
            "Homelab AI is a local retrieval augmented generation system. "
            "It scans documents, chunks text, embeds chunks, and stores vectors.",
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

    document_text = parser.parse(file)
    chunks = chunker.chunk(document_text)

    print(f"Loaded document: {sample_path}")
    print(f"Created chunks: {len(chunks)}")

    embedded_chunks = embedder.embed_many(chunks)
    vector_store.add_many(embedded_chunks)

    print(f"Embedded and stored {len(embedded_chunks)} chunks.")


if __name__ == "__main__":
    main()
