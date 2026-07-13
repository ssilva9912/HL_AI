from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.demo import build_rag_pipeline, create_demo_documents
from backend.indexing.indexer import IndexedCorpus, Indexer


@dataclass(frozen=True)
class RAGSource:
    text: str
    score: float
    document: str | None = None
    chunk_id: str | None = None


@dataclass(frozen=True)
class RAGAnswer:
    answer: str
    sources: list[RAGSource]


class HomelabRAGService:
    """
    API-facing adapter for the existing Homelab AI RAG pipeline.

    The corpus is indexed once and reused across requests.
    The pipeline is rebuilt per request so each request can use its own top_k.
    """

    def __init__(
        self,
        document_directory: Path | None = None,
    ) -> None:
        self._document_directory = document_directory or Path("data/rag_demo")
        self._corpus: IndexedCorpus | None = None

    def _get_corpus(self) -> IndexedCorpus:
        if self._corpus is None:
            create_demo_documents(self._document_directory)

            indexer = Indexer()

            self._corpus = indexer.index_directory(self._document_directory)

        return self._corpus

    def ask(
        self,
        question: str,
        top_k: int = 5,
    ) -> RAGAnswer:
        normalized_question = question.strip()

        if not normalized_question:
            raise ValueError("question must not be empty")

        corpus = self._get_corpus()

        pipeline = build_rag_pipeline(
            corpus=corpus,
            top_k=top_k,
        )

        result = pipeline.ask(normalized_question)

        return RAGAnswer(
            answer=result.answer,
            sources=[self._normalize_source(source) for source in result.sources],
        )

    @staticmethod
    def _normalize_source(source: Any) -> RAGSource:
        chunk = source.chunk

        document_name: str | None = None

        source_document = getattr(
            chunk,
            "source_document",
            None,
        )

        if source_document is not None:
            metadata = getattr(
                source_document,
                "metadata",
                None,
            )

            if metadata is not None:
                name = getattr(metadata, "name", None)

                if name is not None:
                    document_name = str(name)

        chunk_index = getattr(
            chunk,
            "chunk_index",
            None,
        )

        return RAGSource(
            text=str(chunk.content),
            score=float(source.score),
            document=document_name,
            chunk_id=(str(chunk_index) if chunk_index is not None else None),
        )
