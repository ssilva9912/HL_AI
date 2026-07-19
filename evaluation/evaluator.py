from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from evaluation.dataset import EvaluationQuestion
from evaluation.metrics import (
    hit_at_k,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

DocumentRetriever = Callable[[str, int], Sequence[str]]


@dataclass(frozen=True)
class QuestionEvaluation:
    question: str
    relevant_documents: tuple[str, ...]
    retrieved_documents: tuple[str, ...]
    hit_at_1: float
    hit_at_5: float
    precision_at_5: float
    recall_at_5: float
    reciprocal_rank: float


@dataclass(frozen=True)
class EvaluationMetrics:
    question_count: int
    hit_at_1: float
    hit_at_5: float
    precision_at_5: float
    recall_at_5: float
    mean_reciprocal_rank: float


@dataclass(frozen=True)
class EvaluationReport:
    metrics: EvaluationMetrics
    questions: tuple[QuestionEvaluation, ...]


def evaluate_retriever(
    questions: Sequence[EvaluationQuestion],
    retrieve_documents: DocumentRetriever,
    *,
    top_k: int = 5,
) -> EvaluationReport:
    if not questions:
        raise ValueError("at least one evaluation question is required")

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    question_results: list[QuestionEvaluation] = []
    rankings: list[Sequence[str]] = []
    relevant_documents_by_query: list[frozenset[str]] = []

    for evaluation_question in questions:
        retrieved_documents = tuple(
            retrieve_documents(
                evaluation_question.question,
                top_k,
            )
        )

        relevant_documents = evaluation_question.relevant_documents

        rankings.append(retrieved_documents)
        relevant_documents_by_query.append(relevant_documents)

        question_results.append(
            QuestionEvaluation(
                question=evaluation_question.question,
                relevant_documents=tuple(sorted(relevant_documents)),
                retrieved_documents=retrieved_documents,
                hit_at_1=hit_at_k(
                    retrieved_documents=retrieved_documents,
                    relevant_documents=relevant_documents,
                    k=1,
                ),
                hit_at_5=hit_at_k(
                    retrieved_documents=retrieved_documents,
                    relevant_documents=relevant_documents,
                    k=5,
                ),
                precision_at_5=precision_at_k(
                    retrieved_documents=retrieved_documents,
                    relevant_documents=relevant_documents,
                    k=5,
                ),
                recall_at_5=recall_at_k(
                    retrieved_documents=retrieved_documents,
                    relevant_documents=relevant_documents,
                    k=5,
                ),
                reciprocal_rank=reciprocal_rank(
                    retrieved_documents=retrieved_documents,
                    relevant_documents=relevant_documents,
                ),
            )
        )

    question_count = len(question_results)

    metrics = EvaluationMetrics(
        question_count=question_count,
        hit_at_1=_mean(result.hit_at_1 for result in question_results),
        hit_at_5=_mean(result.hit_at_5 for result in question_results),
        precision_at_5=_mean(result.precision_at_5 for result in question_results),
        recall_at_5=_mean(result.recall_at_5 for result in question_results),
        mean_reciprocal_rank=mean_reciprocal_rank(
            rankings=rankings,
            relevant_documents_by_query=relevant_documents_by_query,
        ),
    )

    return EvaluationReport(
        metrics=metrics,
        questions=tuple(question_results),
    )


def _mean(values: Iterable[float]) -> float:
    resolved_values = list(values)

    if not resolved_values:
        raise ValueError("at least one metric value is required")

    return sum(resolved_values) / len(resolved_values)
