from __future__ import annotations

from collections.abc import Collection, Sequence


def hit_at_k(
    retrieved_documents: Sequence[str],
    relevant_documents: Collection[str],
    k: int,
) -> float:
    _validate_inputs(relevant_documents, k)

    retrieved_at_k = retrieved_documents[:k]

    return float(any(document in relevant_documents for document in retrieved_at_k))


def precision_at_k(
    retrieved_documents: Sequence[str],
    relevant_documents: Collection[str],
    k: int,
) -> float:
    _validate_inputs(relevant_documents, k)

    retrieved_at_k = retrieved_documents[:k]

    if not retrieved_at_k:
        return 0.0

    relevant_retrieved = sum(document in relevant_documents for document in retrieved_at_k)

    return relevant_retrieved / len(retrieved_at_k)


def recall_at_k(
    retrieved_documents: Sequence[str],
    relevant_documents: Collection[str],
    k: int,
) -> float:
    _validate_inputs(relevant_documents, k)

    retrieved_at_k = retrieved_documents[:k]
    unique_retrieved = set(retrieved_at_k)

    relevant_retrieved = len(unique_retrieved.intersection(relevant_documents))

    return relevant_retrieved / len(relevant_documents)


def reciprocal_rank(
    retrieved_documents: Sequence[str],
    relevant_documents: Collection[str],
) -> float:
    _validate_relevant_documents(relevant_documents)

    for rank, document in enumerate(retrieved_documents, start=1):
        if document in relevant_documents:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    rankings: Sequence[Sequence[str]],
    relevant_documents_by_query: Sequence[Collection[str]],
) -> float:
    if len(rankings) != len(relevant_documents_by_query):
        raise ValueError(
            "rankings and relevant document collections must contain the same number of queries"
        )

    if not rankings:
        raise ValueError("at least one ranking is required")

    reciprocal_ranks = [
        reciprocal_rank(
            retrieved_documents=retrieved_documents,
            relevant_documents=relevant_documents,
        )
        for retrieved_documents, relevant_documents in zip(
            rankings,
            relevant_documents_by_query,
            strict=True,
        )
    ]

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def _validate_inputs(
    relevant_documents: Collection[str],
    k: int,
) -> None:
    if k <= 0:
        raise ValueError("k must be positive")

    _validate_relevant_documents(relevant_documents)


def _validate_relevant_documents(
    relevant_documents: Collection[str],
) -> None:
    if not relevant_documents:
        raise ValueError("at least one relevant document is required")
