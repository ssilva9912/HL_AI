import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Protocol

from backend.interfaces.generator import Generator
from backend.interfaces.reranker import Reranker
from backend.interfaces.retriever import RetrievalResult, Retriever
from backend.observability import stage
from backend.observability.metrics import CRAG_DECISIONS
from backend.web.web_retriever import WEB_FILE_TYPE

logger = logging.getLogger(__name__)

_STRIP_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")
_MIN_STRIPS_TO_REFINE = 3
_MAX_QUERY_CHARS = 200
_HISTORY_TURNS = 4
_HISTORY_TURN_CHARS = 300


class Verdict(StrEnum):
    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


class PassageScorer(Protocol):
    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


@dataclass(frozen=True)
class Correction:
    results: list[RetrievalResult]
    verdict: Verdict
    web_used: bool = False
    search_query: str | None = None
    local_score: float | None = None


def is_web_result(result: RetrievalResult) -> bool:
    return result.chunk.source_document.file_type == WEB_FILE_TYPE


class RetrievalEvaluator:
    """
    Grades retrieved passages as correct, ambiguous or incorrect for a question.

    The grade comes from the best cross-encoder score: one clearly relevant passage
    is enough to answer from the documents, while no passage above the lower bound
    means the documents do not cover the question.
    """

    def __init__(
        self,
        correct_threshold: float = 2.0,
        incorrect_threshold: float = -6.0,
    ) -> None:
        if incorrect_threshold >= correct_threshold:
            raise ValueError("incorrect_threshold must be smaller than correct_threshold")

        self._correct_threshold = correct_threshold
        self._incorrect_threshold = incorrect_threshold

    def evaluate(self, results: Sequence[RetrievalResult]) -> Verdict:
        if not results:
            return Verdict.INCORRECT

        best_score = max(result.score for result in results)

        if best_score >= self._correct_threshold:
            return Verdict.CORRECT
        if best_score <= self._incorrect_threshold:
            return Verdict.INCORRECT
        return Verdict.AMBIGUOUS


class KnowledgeRefiner:
    """
    Trims each passage to the sentences that bear on the question.

    Retrieved passages often carry sentences that have nothing to do with the
    question; dropping them leaves the model less to be distracted by and shortens
    the prompt. A passage keeps at least its single best sentence.
    """

    def __init__(
        self,
        scorer: PassageScorer,
        strip_threshold: float = -4.0,
    ) -> None:
        self._scorer = scorer
        self._strip_threshold = strip_threshold

    def refine(
        self,
        query: str,
        results: Sequence[RetrievalResult],
    ) -> list[RetrievalResult]:
        strips_by_index = {
            index: self._split(result.chunk.content) for index, result in enumerate(results)
        }
        candidates = {
            index: strips
            for index, strips in strips_by_index.items()
            if len(strips) >= _MIN_STRIPS_TO_REFINE
        }
        if not candidates:
            return list(results)

        with stage("crag.refine"):
            scores = self._scorer.score(
                query,
                [strip for strips in candidates.values() for strip in strips],
            )

        refined = list(results)
        offset = 0
        for index, strips in candidates.items():
            strip_scores = scores[offset : offset + len(strips)]
            offset += len(strips)

            kept = [
                strip
                for strip, score in zip(strips, strip_scores, strict=True)
                if score >= self._strip_threshold
            ]
            if not kept:
                best = max(range(len(strips)), key=strip_scores.__getitem__)
                kept = [strips[best]]

            result = results[index]
            refined[index] = replace(result, chunk=replace(result.chunk, content=" ".join(kept)))

        return refined

    @staticmethod
    def _split(content: str) -> list[str]:
        return [strip.strip() for strip in _STRIP_BOUNDARY.split(content) if strip.strip()]


class QueryRewriter:
    """Turns a question and its conversation into one standalone web search query."""

    def __init__(self, generator: Generator) -> None:
        self._generator = generator

    def rewrite(
        self,
        question: str,
        history: Sequence[tuple[str, str]] = (),
    ) -> str:
        with stage("crag.rewrite"):
            try:
                raw = self._generator.generate(self._prompt(question, history))
            except Exception:
                logger.warning("Search query rewrite failed; using the question", exc_info=True)
                return question

        for line in raw.splitlines():
            query = line.strip().strip("\"'`").strip()
            if query:
                return query[:_MAX_QUERY_CHARS]
        return question

    @staticmethod
    def _prompt(question: str, history: Sequence[tuple[str, str]]) -> str:
        turns = [
            f"{'User' if role == 'user' else 'Assistant'}: {content.strip()[:_HISTORY_TURN_CHARS]}"
            for role, content in history[-_HISTORY_TURNS:]
            if role in {"user", "assistant"} and content.strip()
        ]
        conversation = "\n".join(turns) or "[No prior conversation.]"

        return (
            "Rewrite the question as one standalone web search query. Resolve pronouns "
            "and references using the conversation, and keep names, versions and dates. "
            "Reply with only the query, without quotes or explanation.\n\n"
            f"CONVERSATION\n{conversation}\n\n"
            f"QUESTION\n{question.strip()}\n\n"
            "SEARCH QUERY\n"
        )


class Corrector:
    """
    Corrective RAG: acts on how well the retrieved documents cover the question.

    - Correct: answer from the documents, trimmed to their relevant sentences.
    - Incorrect: discard them and answer from the web.
    - Ambiguous: use both, ranked together.

    Without a web retriever (web search not allowed for this question) the web
    branches are skipped and the documents are used as retrieved.
    """

    def __init__(
        self,
        evaluator: RetrievalEvaluator,
        refiner: KnowledgeRefiner,
        reranker: Reranker,
        web_retriever: Retriever | None = None,
        rewriter: QueryRewriter | None = None,
        candidate_multiplier: int = 4,
    ) -> None:
        if candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be positive")

        self._evaluator = evaluator
        self._refiner = refiner
        self._reranker = reranker
        self._web_retriever = web_retriever
        self._rewriter = rewriter
        self._candidate_multiplier = candidate_multiplier

    def correct(
        self,
        question: str,
        local_results: list[RetrievalResult],
        *,
        top_k: int,
        history: Sequence[tuple[str, str]] = (),
    ) -> Correction:
        with stage("crag.evaluate"):
            verdict = self._evaluator.evaluate(local_results)
        local_score = max((result.score for result in local_results), default=None)

        local = (
            [] if verdict is Verdict.INCORRECT else self._refiner.refine(question, local_results)
        )

        if verdict is Verdict.CORRECT or self._web_retriever is None:
            return self._finish(verdict, local or local_results, local_score=local_score)

        search_query = self._rewriter.rewrite(question, history) if self._rewriter else question
        web = self._search_web(self._web_retriever, search_query, top_k)

        if not web:
            return self._finish(
                verdict, local or local_results, search_query=search_query, local_score=local_score
            )

        merged = sorted([*local, *web], key=lambda result: -result.score)[:top_k]
        return self._finish(
            verdict, merged, web_used=True, search_query=search_query, local_score=local_score
        )

    def _search_web(
        self,
        web_retriever: Retriever,
        search_query: str,
        top_k: int,
    ) -> list[RetrievalResult]:
        candidates = web_retriever.search(
            query=search_query,
            top_k=top_k * self._candidate_multiplier,
        )
        if not candidates:
            return []

        ranked = self._reranker.rerank(query=search_query, results=candidates, top_k=top_k)
        return self._refiner.refine(search_query, ranked)

    @staticmethod
    def _finish(
        verdict: Verdict,
        results: list[RetrievalResult],
        *,
        web_used: bool = False,
        search_query: str | None = None,
        local_score: float | None = None,
    ) -> Correction:
        has_documents = any(not is_web_result(result) for result in results)
        if web_used and has_documents:
            sources = "documents_and_web"
        elif web_used:
            sources = "web"
        elif results and verdict is not Verdict.INCORRECT:
            sources = "documents"
        else:
            sources = "none"
        CRAG_DECISIONS.labels(verdict=verdict.value, sources=sources).inc()

        return Correction(
            results=results,
            verdict=verdict,
            web_used=web_used,
            search_query=search_query,
            local_score=local_score,
        )
