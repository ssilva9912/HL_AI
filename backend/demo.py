from pathlib import Path

from backend.indexing.indexer import IndexedCorpus, Indexer
from backend.llm.ollama_generator import OllamaGenerator
from backend.rag.pipeline import RAGPipeline
from backend.rag.prompt_builder import PromptBuilder
from backend.retrieval.bm25_retriever import BM25Retriever
from backend.retrieval.cross_encoder_reranker import CrossEncoderReranker
from backend.retrieval.dense_retriever import DenseRetriever
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.reranking_retriever import RerankingRetriever


def create_demo_documents(directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)

    documents = {
        "architecture.txt": (
            "Homelab AI is a local retrieval augmented generation platform. "
            "The system separates document parsing, semantic chunking, embedding, "
            "storage, retrieval, reranking, prompt construction, and generation "
            "into modular components."
        ),
        "bm25.txt": (
            "BM25 performs lexical retrieval. It is useful for exact terms, "
            "filenames, identifiers, error messages, and function names."
        ),
        "dense_retrieval.txt": (
            "Dense retrieval uses vector embeddings to locate semantically related "
            "content. It can find relevant passages even when a question uses "
            "different wording from the original document."
        ),
        "fusion.txt": (
            "Reciprocal Rank Fusion combines ranked results from BM25 and dense "
            "retrieval. RRF uses the rank position from each retriever instead of "
            "directly comparing incompatible raw similarity scores."
        ),
        "reranking.txt": (
            "A cross-encoder reranker evaluates the question and each retrieved "
            "passage together. It reorders candidate passages according to their "
            "estimated relevance to the question."
        ),
    }

    paths: list[Path] = []

    for filename, content in documents.items():
        path = directory / filename
        path.write_text(content, encoding="utf-8")
        paths.append(path)

    return paths


def build_rag_pipeline(corpus: IndexedCorpus) -> RAGPipeline:
    bm25_retriever = BM25Retriever(corpus.chunks)

    dense_retriever = DenseRetriever(
        embedder=corpus.embedder,
        vector_store=corpus.vector_store,
    )

    hybrid_retriever = HybridRetriever(
        retrievers=[
            bm25_retriever,
            dense_retriever,
        ],
        rrf_k=60,
        candidate_multiplier=2,
    )

    reranking_retriever = RerankingRetriever(
        retriever=hybrid_retriever,
        reranker=CrossEncoderReranker(),
        candidate_multiplier=4,
    )

    return RAGPipeline(
        retriever=reranking_retriever,
        prompt_builder=PromptBuilder(),
        generator=OllamaGenerator(
            model="llama3.1:8b",
            temperature=0.1,
        ),
        top_k=3,
    )


def main() -> None:
    print("=" * 70)
    print("Homelab AI — Indexed End-to-End RAG Demo")
    print("=" * 70)

    demo_directory = Path("data/rag_demo")

    create_demo_documents(demo_directory)

    print("\nIndexing documents...")

    indexer = Indexer()
    corpus = indexer.index_directory(demo_directory)

    print(f"Indexed documents: {corpus.document_count}")
    print(f"Indexed chunks: {corpus.chunk_count}")
    print(f"Stored vectors: {corpus.vector_store.count()}")

    pipeline = build_rag_pipeline(corpus)

    question = (
        "Why does Homelab AI use Reciprocal Rank Fusion "
        "instead of directly averaging retrieval scores?"
    )

    print("\nQuestion")
    print("-" * 70)
    print(question)

    print("\nRetrieving context and generating answer...")

    response = pipeline.ask(question)

    print("\nAnswer")
    print("-" * 70)
    print(response.answer)

    print("\nSources")
    print("-" * 70)

    if not response.sources:
        print("No sources retrieved.")
        return

    for position, result in enumerate(
        response.sources,
        start=1,
    ):
        chunk = result.chunk

        print(
            f"{position}. "
            f"{chunk.source_document.metadata.name} "
            f"(chunk {chunk.chunk_index}, "
            f"score {result.score:.4f})"
        )

        print(f"   {chunk.content}")


if __name__ == "__main__":
    main()
