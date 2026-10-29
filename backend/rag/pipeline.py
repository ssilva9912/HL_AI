from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from backend.interfaces.generator import Generator
from backend.interfaces.retriever import RetrievalResult, Retriever
from backend.observability import stage
from backend.observability.metrics import RAG_PROMPT_CHARS
from backend.rag.corrective import Correction, Corrector, is_web_result
from backend.rag.prompt_builder import PromptBuilder


@dataclass(frozen=True)
class RAGResponse:
    question: str
    answer: str
    sources: list[RetrievalResult]
    answer_mode: str = "documents"
    correction: Correction | None = None


@dataclass(frozen=True)
class PreparedRAGRequest:
    question: str
    prompt: str
    sources: list[RetrievalResult]
    answer_mode: str
    correction: Correction | None = None


def _sourced_answer_mode(results: Sequence[RetrievalResult]) -> str:
    web_count = sum(1 for result in results if is_web_result(result))

    if web_count == 0:
        return "documents"
    if web_count == len(results):
        return "web"
    return "documents_web"


class StreamingGenerator(Protocol):
    def stream(self, prompt: str) -> Iterator[str]: ...


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        prompt_builder: PromptBuilder,
        generator: Generator,
        top_k: int = 5,
        corrector: Corrector | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        self._retriever = retriever
        self._prompt_builder = prompt_builder
        self._generator = generator
        self._top_k = top_k
        self._corrector = corrector

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
        with stage("generation"):
            answer = self._generator.generate(prepared.prompt)

        return RAGResponse(
            question=prepared.question,
            answer=answer,
            sources=prepared.sources,
            answer_mode=prepared.answer_mode,
            correction=prepared.correction,
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

        correction: Correction | None = None
        if self._corrector is not None:
            correction = self._corrector.correct(
                question,
                results,
                top_k=self._top_k,
                history=history,
            )
            results = correction.results

        relevant_results = [
            result for result in results if not hybrid or result.score >= minimum_relevance_score
        ]
        with stage("prompt_build"):
            if hybrid and not relevant_results:
                prompt = self._prompt_builder.build_general(question=question)
                answer_mode = "general"
            else:
                prompt = self._prompt_builder.build(
                    question=question,
                    results=relevant_results,
                    history=history,
                )
                answer_mode = _sourced_answer_mode(relevant_results)
        RAG_PROMPT_CHARS.observe(len(prompt))

        return PreparedRAGRequest(
            question=question,
            prompt=prompt,
            sources=relevant_results,
            answer_mode=answer_mode,
            correction=correction,
        )

    def stream(self, prepared: PreparedRAGRequest) -> Iterator[str]:
        generator = cast(StreamingGenerator, self._generator)
        yield from generator.stream(prepared.prompt)
