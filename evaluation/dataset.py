from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvaluationQuestion:
    question: str
    relevant_documents: frozenset[str]


class EvaluationDatasetError(ValueError):
    """Raised when an evaluation dataset is malformed."""


def load_evaluation_dataset(path: str | Path) -> list[EvaluationQuestion]:
    dataset_path = Path(path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"evaluation dataset does not exist: {dataset_path}")

    if not dataset_path.is_file():
        raise EvaluationDatasetError(f"evaluation dataset path is not a file: {dataset_path}")

    try:
        raw_data = json.loads(dataset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationDatasetError(
            f"evaluation dataset contains invalid JSON: {dataset_path}"
        ) from exc

    if not isinstance(raw_data, list):
        raise EvaluationDatasetError("evaluation dataset must contain a JSON list")

    questions = [_parse_question(entry, index=index) for index, entry in enumerate(raw_data)]

    if not questions:
        raise EvaluationDatasetError("evaluation dataset must contain at least one question")

    return questions


def _parse_question(
    entry: Any,
    *,
    index: int,
) -> EvaluationQuestion:
    if not isinstance(entry, dict):
        raise EvaluationDatasetError(f"entry {index} must be a JSON object")

    question = entry.get("question")
    relevant_documents = entry.get("relevant_documents")

    if not isinstance(question, str) or not question.strip():
        raise EvaluationDatasetError(f"entry {index} must contain a non-empty question")

    if not isinstance(relevant_documents, list):
        raise EvaluationDatasetError(f"entry {index} must contain a relevant_documents list")

    normalized_documents: set[str] = set()

    for document_index, document in enumerate(relevant_documents):
        if not isinstance(document, str) or not document.strip():
            raise EvaluationDatasetError(
                f"entry {index} relevant document {document_index} must be a non-empty string"
            )

        normalized_documents.add(document.strip())

    if not normalized_documents:
        raise EvaluationDatasetError(f"entry {index} must contain at least one relevant document")

    return EvaluationQuestion(
        question=question.strip(),
        relevant_documents=frozenset(normalized_documents),
    )
