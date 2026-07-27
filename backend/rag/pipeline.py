from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from backend.interfaces.generator import Generator
from backend.interfaces.retriever import RetrievalResult, Retriever
from backend.rag.prompt_builder import PromptBuilder


@dataclass(frozen=True)
class RAGResponse:
    question: str
    answer: str
    sources: list[RetrievalResult]
    answer_mode: str = "documents"


@dataclass(frozen=True)
class PreparedRAGRequest:
    question: str
    prompt: str
    sources: list[RetrievalResult]
    answer_mode: str


class StreamingGenerator(Protocol):
    def stream(self, prompt: str) -> Iterator[str]: ...


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        prompt_builder: PromptBuilder,
        generator: Generator,
        top_k: int = 5,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        self._retriever = retriever
        self._prompt_builder = prompt_builder
        self._generator = generator
        self._top_k = top_k

    def ask(
        self,
        question: str,
        history: Sequence[tuple[str, str]] = (),
        *,
        hybrid: bool = False,
        minimum_relevance_score: float = 0.0,
    ) -> RAGResponse:
        question = question.strip()

        if not question:
            raise ValueError("question must not be empty")

        prepared = self.prepare(
            question,
            history=history,
            hybrid=hybrid,
            minimum_relevance_score=minimum_relevance_score,
        )
        answer = self._generator.generate(prepared.prompt)

        return RAGResponse(
            question=prepared.question,
            answer=answer,
            sources=prepared.sources,
            answer_mode=prepared.answer_mode,
        )

    def prepare(
        self,
        question: str,
        history: Sequence[tuple[str, str]] = (),
        *,
        hybrid: bool = False,
        minimum_relevance_score: float = 0.0,
    ) -> PreparedRAGRequest:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")

        results = self._retriever.search(
            query=question,
            top_k=self._top_k,
        )

        relevant_results = [
            result for result in results if not hybrid or result.score >= minimum_relevance_score
        ]
        if hybrid and not relevant_results:
            prompt = self._prompt_builder.build_general(question=question)
            answer_mode = "general"
        else:
            prompt = self._prompt_builder.build(
                question=question,
                results=relevant_results,
                history=history,
            )
            answer_mode = "documents"

        return PreparedRAGRequest(
            question=question,
            prompt=prompt,
            sources=relevant_results,
            answer_mode=answer_mode,
        )

    def stream(self, prepared: PreparedRAGRequest) -> Iterator[str]:
        generator = cast(StreamingGenerator, self._generator)
        yield from generator.stream(prepared.prompt)
