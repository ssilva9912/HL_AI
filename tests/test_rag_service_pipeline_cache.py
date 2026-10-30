from pathlib import Path
from typing import Any, cast

import pytest

import backend.api.rag_service as rag_service_module
from backend.api.rag_service import HomelabRAGService
from backend.config import Settings
from backend.indexing.models import IndexedCorpus


class FakeCorpus:
    chunk_count = 3
    chunks: list[object] = []


class FakePipeline:
    def __init__(self, top_k: int) -> None:
        self.top_k = top_k


class EmbedderStub:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def embed_text(self, text: str) -> list[float]:
        self._calls.append("embed")
        return [0.0]


@pytest.fixture
def build_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_build(**kwargs: Any) -> FakePipeline:
        calls.append(kwargs)
        return FakePipeline(kwargs["top_k"])

    monkeypatch.setattr(rag_service_module, "build_rag_pipeline", fake_build)
    return calls


def make_service(tmp_path: Path) -> HomelabRAGService:
    return HomelabRAGService(
        settings=Settings(document_directory=tmp_path, vector_store_path=tmp_path / "index"),
        indexer_factory=lambda: cast(Any, None),
    )


def corpus() -> IndexedCorpus:
    return cast(IndexedCorpus, FakeCorpus())


def test_pipeline_is_built_once_per_corpus_and_top_k(
    tmp_path: Path,
    build_calls: list[dict[str, Any]],
) -> None:
    service = make_service(tmp_path)
    current = corpus()

    first = service._get_pipeline(current, 5)
    second = service._get_pipeline(current, 5)
    other_top_k = service._get_pipeline(current, 3)

    assert first is second
    assert other_top_k is not first
    assert len(build_calls) == 2


def test_pipeline_is_rebuilt_when_the_corpus_changes(
    tmp_path: Path,
    build_calls: list[dict[str, Any]],
) -> None:
    service = make_service(tmp_path)

    before = service._get_pipeline(corpus(), 5)
    after = service._get_pipeline(corpus(), 5)

    assert before is not after
    assert len(build_calls) == 2


def test_expensive_components_are_shared_across_pipelines(
    tmp_path: Path,
    build_calls: list[dict[str, Any]],
) -> None:
    service = make_service(tmp_path)
    current = corpus()

    service._get_pipeline(current, 5)
    service._get_pipeline(current, 3)

    first, second = build_calls
    for shared in ("bm25_retriever", "reranker", "generator", "embedding_cache", "retrieval_cache"):
        assert first[shared] is second[shared]
        assert first[shared] is not None


def test_corpus_change_clears_cached_retrievals_and_the_bm25_index(
    tmp_path: Path,
    build_calls: list[dict[str, Any]],
) -> None:
    service = make_service(tmp_path)
    service._get_pipeline(corpus(), 5)
    retrieval_cache = build_calls[0]["retrieval_cache"]
    retrieval_cache.put(("question", 5), ())

    service._get_pipeline(corpus(), 5)

    assert len(retrieval_cache) == 0
    assert build_calls[1]["bm25_retriever"] is not build_calls[0]["bm25_retriever"]


def test_warm_up_continues_when_a_step_fails(
    tmp_path: Path,
    build_calls: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = make_service(tmp_path)
    monkeypatch.setattr(service, "_get_corpus", lambda: corpus())
    calls: list[str] = []

    def failing_preload() -> None:
        calls.append("llm")
        raise ConnectionError("ollama is down")

    monkeypatch.setattr(service._reranker, "warm_up", lambda: calls.append("reranker"))
    monkeypatch.setattr(service._generator, "preload", failing_preload)
    monkeypatch.setattr(service, "_create_embedder", lambda: EmbedderStub(calls))

    service.warm_up()

    assert calls == ["reranker", "llm", "embed"]
    assert len(build_calls) == 1
