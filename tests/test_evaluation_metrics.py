import pytest

from evaluation.metrics import (
    hit_at_k,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_hit_at_k_returns_one_when_relevant_document_is_found() -> None:
    score = hit_at_k(
        retrieved_documents=[
            "dense.txt",
            "fusion.txt",
            "bm25.txt",
        ],
        relevant_documents={"fusion.txt"},
        k=2,
    )

    assert score == 1.0


def test_hit_at_k_returns_zero_when_relevant_document_is_not_found() -> None:
    score = hit_at_k(
        retrieved_documents=[
            "dense.txt",
            "architecture.txt",
            "bm25.txt",
        ],
        relevant_documents={"fusion.txt"},
        k=2,
    )

    assert score == 0.0


def test_precision_at_k() -> None:
    score = precision_at_k(
        retrieved_documents=[
            "fusion.txt",
            "dense.txt",
            "architecture.txt",
            "bm25.txt",
        ],
        relevant_documents={
            "fusion.txt",
            "architecture.txt",
        },
        k=3,
    )

    assert score == pytest.approx(2 / 3)


def test_precision_at_k_uses_available_results_when_fewer_than_k() -> None:
    score = precision_at_k(
        retrieved_documents=[
            "fusion.txt",
            "dense.txt",
        ],
        relevant_documents={
            "fusion.txt",
        },
        k=5,
    )

    assert score == pytest.approx(1 / 2)


def test_precision_at_k_returns_zero_for_empty_results() -> None:
    score = precision_at_k(
        retrieved_documents=[],
        relevant_documents={"fusion.txt"},
        k=5,
    )

    assert score == 0.0


def test_recall_at_k() -> None:
    score = recall_at_k(
        retrieved_documents=[
            "fusion.txt",
            "dense.txt",
            "architecture.txt",
        ],
        relevant_documents={
            "fusion.txt",
            "architecture.txt",
            "bm25.txt",
        },
        k=3,
    )

    assert score == pytest.approx(2 / 3)


def test_recall_at_k_does_not_count_duplicate_results_twice() -> None:
    score = recall_at_k(
        retrieved_documents=[
            "fusion.txt",
            "fusion.txt",
            "dense.txt",
        ],
        relevant_documents={
            "fusion.txt",
            "architecture.txt",
        },
        k=3,
    )

    assert score == pytest.approx(1 / 2)


def test_reciprocal_rank_returns_inverse_first_relevant_rank() -> None:
    score = reciprocal_rank(
        retrieved_documents=[
            "dense.txt",
            "architecture.txt",
            "fusion.txt",
        ],
        relevant_documents={"fusion.txt"},
    )

    assert score == pytest.approx(1 / 3)


def test_reciprocal_rank_returns_zero_when_no_relevant_result_exists() -> None:
    score = reciprocal_rank(
        retrieved_documents=[
            "dense.txt",
            "architecture.txt",
        ],
        relevant_documents={"fusion.txt"},
    )

    assert score == 0.0


def test_mean_reciprocal_rank() -> None:
    score = mean_reciprocal_rank(
        rankings=[
            [
                "fusion.txt",
                "dense.txt",
            ],
            [
                "dense.txt",
                "bm25.txt",
            ],
            [
                "architecture.txt",
                "reranking.txt",
            ],
        ],
        relevant_documents_by_query=[
            {"fusion.txt"},
            {"bm25.txt"},
            {"missing.txt"},
        ],
    )

    expected = (1.0 + 0.5 + 0.0) / 3

    assert score == pytest.approx(expected)


@pytest.mark.parametrize(
    "metric",
    [
        hit_at_k,
        precision_at_k,
        recall_at_k,
    ],
)
def test_at_k_metrics_reject_non_positive_k(
    metric: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="k must be positive",
    ):
        metric(  # type: ignore[operator]
            ["fusion.txt"],
            {"fusion.txt"},
            0,
        )


@pytest.mark.parametrize(
    "metric_call",
    [
        lambda: hit_at_k(
            ["fusion.txt"],
            set(),
            1,
        ),
        lambda: precision_at_k(
            ["fusion.txt"],
            set(),
            1,
        ),
        lambda: recall_at_k(
            ["fusion.txt"],
            set(),
            1,
        ),
        lambda: reciprocal_rank(
            ["fusion.txt"],
            set(),
        ),
    ],
)
def test_metrics_reject_empty_relevant_documents(
    metric_call: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="at least one relevant document",
    ):
        metric_call()  # type: ignore[operator]


def test_mean_reciprocal_rank_rejects_different_query_counts() -> None:
    with pytest.raises(
        ValueError,
        match="same number of queries",
    ):
        mean_reciprocal_rank(
            rankings=[
                ["fusion.txt"],
            ],
            relevant_documents_by_query=[
                {"fusion.txt"},
                {"bm25.txt"},
            ],
        )


def test_mean_reciprocal_rank_rejects_empty_input() -> None:
    with pytest.raises(
        ValueError,
        match="at least one ranking",
    ):
        mean_reciprocal_rank(
            rankings=[],
            relevant_documents_by_query=[],
        )
