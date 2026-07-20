import os
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_TOP_K = 5


def get_api_url() -> str:
    return os.getenv(
        "HOMELAB_API_URL",
        DEFAULT_API_URL,
    ).rstrip("/")


def get_request_timeout() -> float:
    raw_value = os.getenv(
        "HOMELAB_REQUEST_TIMEOUT",
        str(DEFAULT_TIMEOUT_SECONDS),
    )

    try:
        timeout = float(raw_value)
    except ValueError as exc:
        raise ValueError("HOMELAB_REQUEST_TIMEOUT must be numeric.") from exc

    if timeout <= 0:
        raise ValueError("HOMELAB_REQUEST_TIMEOUT must be positive.")

    return timeout


def get_default_top_k() -> int:
    raw_value = os.getenv(
        "HOMELAB_DEFAULT_TOP_K",
        str(DEFAULT_TOP_K),
    )

    try:
        top_k = int(raw_value)
    except ValueError as exc:
        raise ValueError("HOMELAB_DEFAULT_TOP_K must be an integer.") from exc

    if not 1 <= top_k <= 20:
        raise ValueError("HOMELAB_DEFAULT_TOP_K must be between 1 and 20.")

    return top_k


class HomelabAPIError(RuntimeError):
    """Raised when the Homelab AI backend cannot complete a request."""


@dataclass(frozen=True)
class Source:
    text: str
    score: float
    document: str
    chunk_id: str


@dataclass(frozen=True)
class SearchResult:
    answer: str
    sources: list[Source]


@dataclass(frozen=True)
class UploadResult:
    document: str
    size_bytes: int
    document_count: int
    chunk_count: int
    status: str


@dataclass(frozen=True)
class EvaluationMetrics:
    question_count: int
    hit_at_1: float
    hit_at_5: float
    precision_at_5: float
    recall_at_5: float
    mean_reciprocal_rank: float


@dataclass(frozen=True)
class QuestionEvaluation:
    question: str
    relevant_documents: list[str]
    retrieved_documents: list[str]
    hit_at_1: float
    hit_at_5: float
    precision_at_5: float
    recall_at_5: float
    reciprocal_rank: float


@dataclass(frozen=True)
class EvaluationResult:
    metrics: EvaluationMetrics
    questions: list[QuestionEvaluation]
    top_k: int
    elapsed_ms: float


class HomelabAPIClient:
    def __init__(
        self,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        resolved_base_url = base_url or get_api_url()
        resolved_timeout = timeout_seconds if timeout_seconds is not None else get_request_timeout()

        if not resolved_base_url.strip():
            raise ValueError("base_url must not be empty")

        if resolved_timeout <= 0:
            raise ValueError("timeout_seconds must be positive")

        self.base_url = resolved_base_url.rstrip("/")
        self.timeout = httpx.Timeout(resolved_timeout)

    def health(self) -> dict[str, str]:
        response = self._request(
            method="GET",
            path="/health",
        )

        return {
            "status": str(
                response.get(
                    "status",
                    "unknown",
                )
            ),
            "service": str(
                response.get(
                    "service",
                    "unknown",
                )
            ),
        }

    def search(
        self,
        question: str,
        top_k: int | None = None,
    ) -> SearchResult:
        cleaned_question = question.strip()
        resolved_top_k = top_k if top_k is not None else get_default_top_k()

        if not cleaned_question:
            raise ValueError("Question cannot be empty.")

        if not 1 <= resolved_top_k <= 20:
            raise ValueError("top_k must be between 1 and 20.")

        response = self._request(
            method="POST",
            path="/search",
            json={
                "question": cleaned_question,
                "top_k": resolved_top_k,
            },
        )

        raw_sources = response.get(
            "sources",
            [],
        )

        if not isinstance(raw_sources, list):
            raise HomelabAPIError("The backend returned an invalid sources field.")

        sources = [
            Source(
                text=str(
                    source.get(
                        "text",
                        "",
                    )
                ),
                score=float(
                    source.get(
                        "score",
                        0.0,
                    )
                ),
                document=str(
                    source.get(
                        "document",
                        "unknown",
                    )
                ),
                chunk_id=str(
                    source.get(
                        "chunk_id",
                        "",
                    )
                ),
            )
            for source in raw_sources
            if isinstance(source, dict)
        ]

        return SearchResult(
            answer=str(
                response.get(
                    "answer",
                    "",
                )
            ),
            sources=sources,
        )

    def upload_document(
        self,
        filename: str,
        content: bytes,
        content_type: str | None = None,
    ) -> UploadResult:
        cleaned_filename = filename.strip()

        if not cleaned_filename:
            raise ValueError("The uploaded file must have a filename.")

        if not content:
            raise ValueError("The uploaded file is empty.")

        response = self._request(
            method="POST",
            path="/ingest",
            files={
                "file": (
                    cleaned_filename,
                    content,
                    content_type or "application/octet-stream",
                )
            },
        )

        return UploadResult(
            document=str(
                response.get(
                    "document",
                    cleaned_filename,
                )
            ),
            size_bytes=int(
                response.get(
                    "size_bytes",
                    len(content),
                )
            ),
            document_count=int(
                response.get(
                    "document_count",
                    0,
                )
            ),
            chunk_count=int(
                response.get(
                    "chunk_count",
                    0,
                )
            ),
            status=str(
                response.get(
                    "status",
                    "unknown",
                )
            ),
        )

    def evaluate(
        self,
        top_k: int = 5,
    ) -> EvaluationResult:
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20.")

        response = self._request(
            method="POST",
            path="/evaluate",
            json={
                "top_k": top_k,
            },
        )

        raw_metrics = response.get("metrics")
        raw_questions = response.get(
            "questions",
            [],
        )

        if not isinstance(raw_metrics, dict):
            raise HomelabAPIError("The backend returned invalid evaluation metrics.")

        if not isinstance(raw_questions, list):
            raise HomelabAPIError("The backend returned invalid question results.")

        metrics = EvaluationMetrics(
            question_count=int(
                raw_metrics.get(
                    "question_count",
                    0,
                )
            ),
            hit_at_1=float(
                raw_metrics.get(
                    "hit_at_1",
                    0.0,
                )
            ),
            hit_at_5=float(
                raw_metrics.get(
                    "hit_at_5",
                    0.0,
                )
            ),
            precision_at_5=float(
                raw_metrics.get(
                    "precision_at_5",
                    0.0,
                )
            ),
            recall_at_5=float(
                raw_metrics.get(
                    "recall_at_5",
                    0.0,
                )
            ),
            mean_reciprocal_rank=float(
                raw_metrics.get(
                    "mean_reciprocal_rank",
                    0.0,
                )
            ),
        )

        questions = [
            self._parse_question_evaluation(question)
            for question in raw_questions
            if isinstance(question, dict)
        ]

        return EvaluationResult(
            metrics=metrics,
            questions=questions,
            top_k=int(
                response.get(
                    "top_k",
                    top_k,
                )
            ),
            elapsed_ms=float(
                response.get(
                    "elapsed_ms",
                    0.0,
                )
            ),
        )

    @staticmethod
    def _parse_question_evaluation(
        raw_question: dict[str, Any],
    ) -> QuestionEvaluation:
        relevant_documents = raw_question.get(
            "relevant_documents",
            [],
        )
        retrieved_documents = raw_question.get(
            "retrieved_documents",
            [],
        )

        if not isinstance(
            relevant_documents,
            list,
        ):
            relevant_documents = []

        if not isinstance(
            retrieved_documents,
            list,
        ):
            retrieved_documents = []

        return QuestionEvaluation(
            question=str(
                raw_question.get(
                    "question",
                    "",
                )
            ),
            relevant_documents=[str(document) for document in relevant_documents],
            retrieved_documents=[str(document) for document in retrieved_documents],
            hit_at_1=float(
                raw_question.get(
                    "hit_at_1",
                    0.0,
                )
            ),
            hit_at_5=float(
                raw_question.get(
                    "hit_at_5",
                    0.0,
                )
            ),
            precision_at_5=float(
                raw_question.get(
                    "precision_at_5",
                    0.0,
                )
            ),
            recall_at_5=float(
                raw_question.get(
                    "recall_at_5",
                    0.0,
                )
            ),
            reciprocal_rank=float(
                raw_question.get(
                    "reciprocal_rank",
                    0.0,
                )
            ),
        )

    def _request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        files: dict[
            str,
            tuple[str, bytes, str],
        ]
        | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.request(
                    method=method,
                    url=url,
                    json=json,
                    files=files,
                )
                response.raise_for_status()

        except httpx.ConnectError as exc:
            raise HomelabAPIError(
                f"Could not connect to the Homelab AI backend at {self.base_url}."
            ) from exc

        except httpx.TimeoutException as exc:
            raise HomelabAPIError(
                "The backend request timed out while processing the request."
            ) from exc

        except httpx.HTTPStatusError as exc:
            detail = self._extract_error_detail(exc.response)

            raise HomelabAPIError(
                f"Backend request failed with status {exc.response.status_code}: {detail}"
            ) from exc

        except httpx.HTTPError as exc:
            raise HomelabAPIError(f"An HTTP error occurred: {exc}") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise HomelabAPIError(
                "The backend returned a response that was not valid JSON."
            ) from exc

        if not isinstance(payload, dict):
            raise HomelabAPIError("The backend returned an unexpected response format.")

        return payload

    @staticmethod
    def _extract_error_detail(
        response: httpx.Response,
    ) -> str:
        try:
            payload = response.json()
        except ValueError:
            return response.text or "Unknown backend error."

        if isinstance(payload, dict):
            detail = payload.get("detail")

            if detail is not None:
                return str(detail)

        return response.text or "Unknown backend error."
