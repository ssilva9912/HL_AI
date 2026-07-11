from pathlib import Path

from backend.chunking.fixed_size_chunker import FixedSizeChunker
from backend.embeddings.ollama_embedder import OllamaEmbedder
from backend.ingestion.scanner import FileMetadata
from backend.interfaces.chunker import DocumentChunk
from backend.parser.text_parser import TextParser
from backend.retrieval.bm25_retriever import BM25Retriever
from backend.retrieval.cross_encoder_reranker import CrossEncoderReranker
from backend.retrieval.dense_retriever import DenseRetriever
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.reranking_retriever import RerankingRetriever
from backend.storage.in_memory_vector_store import InMemoryVectorStore


def create_demo_documents(directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)

    documents = {
        "architecture.txt": (
            "Homelab AI is a local retrieval augmented generation platform. "
            "The system separates document parsing, chunking, embedding, "
            "storage, retrieval, reranking, and generation into modular components."
        ),
        "lexical_retrieval.txt": (
            "BM25 performs lexical retrieval. It is especially useful for exact "
            "terms, filenames, function names, error messages, and identifiers "
            "such as source_document, api_embed, and chunk_index."
        ),
        "dense_retrieval.txt": (
            "Dense retrieval uses vector embeddings to find semantically related "
            "content. It can retrieve relevant passages even when the query uses "
            "different wording from the original document."
        ),
        "fusion.txt": (
            "Reciprocal Rank Fusion combines ranked results from BM25 and dense "
            "retrieval. RRF uses the rank position from each retriever rather than "
            "comparing incompatible raw similarity scores."
        ),
        "gardening.txt": (
            "Tomato plants require sunlight, fertile soil, consistent irrigation, "
            "and protection from frost. Soil moisture sensors can help automate "
            "garden watering systems."
        ),
    }

    paths: list[Path] = []

    for filename, content in documents.items():
        path = directory / filename
        path.write_text(content, encoding="utf-8")
        paths.append(path)

    return paths


def parse_and_chunk(paths: list[Path]) -> list[DocumentChunk]:
    parser = TextParser()
    chunker = FixedSizeChunker(
        chunk_size=120,
        overlap=20,
    )

    chunks: list[DocumentChunk] = []

    for path in paths:
        metadata = FileMetadata(
            name=path.name,
            path=path,
            extension=path.suffix.lower(),
            size_bytes=path.stat().st_size,
        )

        parsed_document = parser.parse(metadata)
        chunks.extend(chunker.chunk(parsed_document))

    return chunks


def main() -> None:
    document_paths = create_demo_documents(Path("data/hybrid_demo"))
    chunks = parse_and_chunk(document_paths)

    embedder = OllamaEmbedder()
    vector_store = InMemoryVectorStore()

    embedded_chunks = embedder.embed_many(chunks)
    vector_store.add_many(embedded_chunks)

    bm25_retriever = BM25Retriever(chunks)

    dense_retriever = DenseRetriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    hybrid_retriever = HybridRetriever(
        retrievers=[
            bm25_retriever,
            dense_retriever,
        ],
        rrf_k=60,
        candidate_multiplier=1,
    )

    reranker = CrossEncoderReranker()

    retrieval_pipeline = RerankingRetriever(
        retriever=hybrid_retriever,
        reranker=reranker,
        candidate_multiplier=4,
    )

    question = "How does the system combine semantic search with exact identifier matching?"

    results = retrieval_pipeline.search(
        query=question,
        top_k=5,
    )

    print("=" * 70)
    print("HLAI Production Retrieval Pipeline")
    print("=" * 70)

    print(f"\nIndexed documents : {len(document_paths)}")
    print(f"Indexed chunks    : {len(chunks)}")

    print("\nPipeline")
    print("Query")
    print("  ↓")
    print("BM25 + Dense Retrieval")
    print("  ↓")
    print("Reciprocal Rank Fusion")
    print("  ↓")
    print("Cross-Encoder Reranking")
    print("  ↓")
    print("Final Top-K Context")

    print(f"\nQuestion:\n{question}")

    print("\nFinal Results")
    print("-" * 70)

    for position, result in enumerate(results, start=1):
        chunk = result.chunk

        print(f"\n[{position}] {chunk.source_document.metadata.name}")
        print(f"Chunk Index : {chunk.chunk_index}")
        print(f"Retriever   : {result.retriever}")
        print(f"Cross-Encoder Logit : {result.score:.3f}")

        if result.original_score is not None:
            print(f"Original RRF Score  : {result.original_score:.6f}")

        print("Content:")
        print(chunk.content)

if __name__ == "__main__":
    main()
