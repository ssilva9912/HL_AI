from dataclasses import dataclass

from backend.interfaces.generator import Generator
from backend.interfaces.retriever import RetrievalResult, Retriever
from backend.rag.prompt_builder import PromptBuilder


@dataclass(frozen=True)
class RAGResponse:
    question: str
    answer: str
    sources: list[RetrievalResult]


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

    def ask(self, question: str) -> RAGResponse:
        question = question.strip()

        if not question:
            raise ValueError("question must not be empty")

        results = self._retriever.search(
            query=question,
            top_k=self._top_k,
        )

        prompt = self._prompt_builder.build(
            question=question,
            results=results,
        )

        answer = self._generator.generate(prompt)

        return RAGResponse(
            question=question,
            answer=answer,
            sources=results,
        )
