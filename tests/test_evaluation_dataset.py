import json
from pathlib import Path

import pytest

from evaluation.dataset import (
    EvaluationDatasetError,
    EvaluationQuestion,
    load_evaluation_dataset,
)


def write_dataset(
    path: Path,
    data: object,
) -> None:
    path.write_text(
        json.dumps(data),
        encoding="utf-8",
    )


def test_load_evaluation_dataset(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "questions.json"

    write_dataset(
        dataset_path,
        [
            {
                "question": "What is BM25?",
                "relevant_documents": [
                    "bm25.txt",
                    "architecture.txt",
                ],
            },
            {
                "question": "How does reranking work?",
                "relevant_documents": [
                    "reranking.txt",
                ],
            },
        ],
    )

    questions = load_evaluation_dataset(dataset_path)

    assert questions == [
        EvaluationQuestion(
            question="What is BM25?",
            relevant_documents=frozenset(
                {
                    "bm25.txt",
                    "architecture.txt",
                }
            ),
        ),
        EvaluationQuestion(
            question="How does reranking work?",
            relevant_documents=frozenset(
                {
                    "reranking.txt",
                }
            ),
        ),
    ]


def test_load_evaluation_dataset_strips_whitespace(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "questions.json"

    write_dataset(
        dataset_path,
        [
            {
                "question": "  What is BM25?  ",
                "relevant_documents": [
                    "  bm25.txt  ",
                ],
            }
        ],
    )

    questions = load_evaluation_dataset(dataset_path)

    assert questions[0].question == "What is BM25?"
    assert questions[0].relevant_documents == frozenset({"bm25.txt"})


def test_load_evaluation_dataset_removes_duplicate_documents(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "questions.json"

    write_dataset(
        dataset_path,
        [
            {
                "question": "What is hybrid retrieval?",
                "relevant_documents": [
                    "fusion.txt",
                    "fusion.txt",
                ],
            }
        ],
    )

    questions = load_evaluation_dataset(dataset_path)

    assert questions[0].relevant_documents == frozenset({"fusion.txt"})


def test_load_evaluation_dataset_rejects_missing_file(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "missing.json"

    with pytest.raises(
        FileNotFoundError,
        match="evaluation dataset does not exist",
    ):
        load_evaluation_dataset(dataset_path)


def test_load_evaluation_dataset_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "questions.json"
    dataset_path.write_text(
        "{not valid json}",
        encoding="utf-8",
    )

    with pytest.raises(
        EvaluationDatasetError,
        match="invalid JSON",
    ):
        load_evaluation_dataset(dataset_path)


def test_load_evaluation_dataset_rejects_non_list_root(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "questions.json"

    write_dataset(
        dataset_path,
        {
            "question": "What is BM25?",
        },
    )

    with pytest.raises(
        EvaluationDatasetError,
        match="must contain a JSON list",
    ):
        load_evaluation_dataset(dataset_path)


def test_load_evaluation_dataset_rejects_empty_dataset(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "questions.json"

    write_dataset(
        dataset_path,
        [],
    )

    with pytest.raises(
        EvaluationDatasetError,
        match="at least one question",
    ):
        load_evaluation_dataset(dataset_path)


@pytest.mark.parametrize(
    ("entry", "expected_message"),
    [
        (
            "not an object",
            "entry 0 must be a JSON object",
        ),
        (
            {
                "question": "",
                "relevant_documents": ["bm25.txt"],
            },
            "non-empty question",
        ),
        (
            {
                "question": "What is BM25?",
                "relevant_documents": "bm25.txt",
            },
            "relevant_documents list",
        ),
        (
            {
                "question": "What is BM25?",
                "relevant_documents": [],
            },
            "at least one relevant document",
        ),
        (
            {
                "question": "What is BM25?",
                "relevant_documents": [""],
            },
            "must be a non-empty string",
        ),
    ],
)
def test_load_evaluation_dataset_rejects_invalid_entries(
    tmp_path: Path,
    entry: object,
    expected_message: str,
) -> None:
    dataset_path = tmp_path / "questions.json"

    write_dataset(
        dataset_path,
        [entry],
    )

    with pytest.raises(
        EvaluationDatasetError,
        match=expected_message,
    ):
        load_evaluation_dataset(dataset_path)
