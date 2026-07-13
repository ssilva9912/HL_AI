from pathlib import Path

import pytest

from backend.ingestion.scanner import FileMetadata
from backend.interfaces.chunker import DocumentChunk
from backend.interfaces.parser import ParsedDocument
from backend.interfaces.retriever import RetrievalResult
from backend.rag.pipeline import RAGPipeline
from backend.rag.prompt_builder import PromptBuilder


class FakeRetriever:
    def __init__(
        self,
        results: list[RetrievalResult],
    ) -> None:
        self.results = results
        self.received_query: str | None = None
        self.received_top_k: int | None = None

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        self.received_query = query
        self.received_top_k = top_k

        return self.results[:top_k]


class FakeGenerator:
    def __init__(
        self,
        answer: str = "Generated answer.",
    ) -> None:
        self.answer = answer
        self.received_prompt: str | None = None

    def generate(self, prompt: str) -> str:
        self.received_prompt = prompt
        return self.answer


def make_result(
    content: str,
    filename: str = "source.txt",
    chunk_index: int = 0,
    score: float = 1.0,
) -> RetrievalResult:
    path = Path(filename)

    metadata = FileMetadata(
        name=filename,
        path=path,
        extension=path.suffix.lower(),
        size_bytes=len(content.encode("utf-8")),
    )

    document = ParsedDocument(
        source_path=path,
        file_type=path.suffix.lower(),
        content=content,
        metadata=metadata,
    )

    chunk = DocumentChunk(
        source_document=document,
        content=content,
        chunk_index=chunk_index,
        start_char=0,
        end_char=len(content),
    )

    return RetrievalResult(
        chunk=chunk,
        score=score,
        retriever="test",
    )


def test_ask_returns_answer_and_sources() -> None:
    results = [
        make_result(
            content="RRF combines ranked retrieval results.",
            filename="fusion.txt",
        )
    ]

    retriever = FakeRetriever(results)
    generator = FakeGenerator(
        answer="RRF combines results using rank positions.",
    )

    pipeline = RAGPipeline(
        retriever=retriever,
        prompt_builder=PromptBuilder(),
        generator=generator,
        top_k=3,
    )

    response = pipeline.ask(
        "How does RRF work?",
    )

    assert response.question == "How does RRF work?"
    assert response.answer == ("RRF combines results using rank positions.")
    assert response.sources == results


def test_ask_passes_question_to_retriever() -> None:
    retriever = FakeRetriever([])
    generator = FakeGenerator()

    pipeline = RAGPipeline(
        retriever=retriever,
        prompt_builder=PromptBuilder(),
        generator=generator,
        top_k=4,
    )

    pipeline.ask("What is BM25?")

    assert retriever.received_query == "What is BM25?"
    assert retriever.received_top_k == 4


def test_ask_builds_prompt_from_retrieved_context() -> None:
    results = [
        make_result(
            content="BM25 performs lexical retrieval.",
            filename="bm25.txt",
        )
    ]

    retriever = FakeRetriever(results)
    generator = FakeGenerator()

    pipeline = RAGPipeline(
        retriever=retriever,
        prompt_builder=PromptBuilder(),
        generator=generator,
    )

    pipeline.ask("What does BM25 do?")

    assert generator.received_prompt is not None
    assert "What does BM25 do?" in generator.received_prompt
    assert "BM25 performs lexical retrieval." in (generator.received_prompt)
    assert "bm25.txt" in generator.received_prompt


def test_ask_handles_no_retrieval_results() -> None:
    retriever = FakeRetriever([])
    generator = FakeGenerator(
        answer="I do not know based on the available context.",
    )

    pipeline = RAGPipeline(
        retriever=retriever,
        prompt_builder=PromptBuilder(),
        generator=generator,
    )

    response = pipeline.ask("What is unavailable?")

    assert response.sources == []
    assert response.answer == ("I do not know based on the available context.")
    assert generator.received_prompt is not None
    assert "[No context was retrieved.]" in (generator.received_prompt)


def test_question_is_stripped_before_processing() -> None:
    retriever = FakeRetriever([])
    generator = FakeGenerator()

    pipeline = RAGPipeline(
        retriever=retriever,
        prompt_builder=PromptBuilder(),
        generator=generator,
    )

    response = pipeline.ask("  What is dense retrieval?  ")

    assert response.question == "What is dense retrieval?"
    assert retriever.received_query == "What is dense retrieval?"


@pytest.mark.parametrize(
    "question",
    [
        "",
        " ",
        "\n\t",
    ],
)
def test_empty_question_raises_value_error(
    question: str,
) -> None:
    pipeline = RAGPipeline(
        retriever=FakeRetriever([]),
        prompt_builder=PromptBuilder(),
        generator=FakeGenerator(),
    )

    with pytest.raises(
        ValueError,
        match="question must not be empty",
    ):
        pipeline.ask(question)


@pytest.mark.parametrize(
    "top_k",
    [
        0,
        -1,
        -10,
    ],
)
def test_invalid_top_k_raises_value_error(
    top_k: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="top_k must be positive",
    ):
        RAGPipeline(
            retriever=FakeRetriever([]),
            prompt_builder=PromptBuilder(),
            generator=FakeGenerator(),
            top_k=top_k,
        )
