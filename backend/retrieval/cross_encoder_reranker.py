from collections.abc import Callable, Sequence
from typing import Protocol, cast

from sentence_transformers import CrossEncoder

from backend.interfaces.retriever import RetrievalResult


class CrossEncoderModel(Protocol):
    def predict(
        self,
        sentences: list[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
    ) -> object: ...


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

        raw_scores = self._get_model().predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )

        scores = [float(score) for score in raw_scores]  # type: ignore[union-attr]

        if len(scores) != len(results):
            raise RuntimeError(
                "cross-encoder returned a different number of scores "
                f"than input pairs: expected {len(results)}, received {len(scores)}"
            )

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

    def _get_model(self) -> CrossEncoderModel:
        if self._model is None:
            self._model = self._model_factory(self._model_name)

        return self._model
