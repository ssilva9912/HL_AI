from collections.abc import Callable, Sequence
from threading import Lock
from typing import Protocol, cast

from sentence_transformers import CrossEncoder

from backend.interfaces.retriever import RetrievalResult
from backend.observability import stage


class CrossEncoderModel(Protocol):
    def predict(
        self,
        sentences: list[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
    ) -> Sequence[float]: ...


ModelFactory = Callable[[str], CrossEncoderModel]


class CrossEncoderReranker:
    DEFAULT_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        batch_size: int = 32,
        model_factory: ModelFactory | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be empty")

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self._model_name = model_name
        self._batch_size = batch_size
        self._model_factory = model_factory or cast(ModelFactory, CrossEncoder)
        self._model: CrossEncoderModel | None = None
        self._load_lock = Lock()

    def rerank(
        self,
        query: str,
        results: Sequence[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("query must not be empty")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if not results:
            return []

        pairs = [
            (
                query,
                result.chunk.content,
            )
            for result in results
        ]

        with stage("retrieval.rerank"):
            scores = self._predict(pairs)

        scored_results = [
            (
                position,
                RetrievalResult(
                    chunk=result.chunk,
                    score=score,
                    retriever="cross_encoder",
                    original_score=result.score,
                ),
            )
            for position, (result, score) in enumerate(zip(results, scores, strict=True))
        ]

        ranked_results = sorted(
            scored_results,
            key=lambda item: (-item[1].score, item[0]),
        )

        return [result for _, result in ranked_results[:top_k]]

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Score each passage against the query; higher means more relevant."""

        if not query.strip():
            raise ValueError("query must not be empty")

        if not passages:
            return []

        with stage("retrieval.score"):
            return self._predict([(query, passage) for passage in passages])

    def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        raw_scores = self._get_model().predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        scores = [float(score) for score in raw_scores]

        if len(scores) != len(pairs):
            raise RuntimeError(
                "cross-encoder returned a different number of scores "
                f"than input pairs: expected {len(pairs)}, received {len(scores)}"
            )

        return scores

    def warm_up(self) -> None:
        """Load the model and run one tiny batch so the first real request is fast."""

        with stage("warmup.reranker"):
            self._get_model().predict(
                [("warm up", "warm up")],
                batch_size=1,
                show_progress_bar=False,
            )

    def _get_model(self) -> CrossEncoderModel:
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    with stage("reranker.model_load"):
                        self._model = self._model_factory(self._model_name)

        return self._model
