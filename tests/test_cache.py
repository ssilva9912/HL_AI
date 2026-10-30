from pathlib import Path

import pytest

from backend.cache import CachedQueryEmbedder, CachingRetriever, LRUCache
from backend.ingestion.scanner import FileMetadata
from backend.interfaces.chunker import DocumentChunk
from backend.interfaces.embedder import EmbeddedChunk
from backend.interfaces.parser import ParsedDocument
from backend.interfaces.retriever import RetrievalResult


def make_result(text: str) -> RetrievalResult:
    metadata = FileMetadata(
        path=Path("a.txt"),
        name="a.txt",
        size_bytes=len(text),
        extension=".txt",
    )
    document = ParsedDocument(
        source_path=Path("a.txt"),
        file_type="text",
        content=text,
        metadata=metadata,
    )
    chunk = DocumentChunk(
        source_document=document,
        content=text,
        chunk_index=0,
        start_char=0,
        end_char=len(text),
    )
    return RetrievalResult(chunk=chunk, score=1.0, retriever="fake")


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_lru_evicts_least_recently_used_entry() -> None:
    cache: LRUCache[str, int] = LRUCache("unit_lru", max_size=2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1

    cache.put("c", 3)

    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert len(cache) == 2


def test_lru_entries_expire_after_ttl() -> None:
    clock = FakeClock()
    cache: LRUCache[str, int] = LRUCache("unit_ttl", max_size=4, ttl_seconds=10, clock=clock)
    cache.put("a", 1)

    clock.now = 9.9
    assert cache.get("a") == 1

    clock.now = 10.1
    assert cache.get("a") is None
    assert len(cache) == 0


def test_lru_with_zero_size_stores_nothing() -> None:
    cache: LRUCache[str, int] = LRUCache("unit_off", max_size=0)

    cache.put("a", 1)

    assert cache.get("a") is None


def test_lru_clear_removes_everything() -> None:
    cache: LRUCache[str, int] = LRUCache("unit_clear", max_size=4)
    cache.put("a", 1)

    cache.clear()

    assert cache.get("a") is None


@pytest.mark.parametrize(
    ("max_size", "ttl"),
    [(-1, None), (1, 0.0), (1, -5.0)],
)
def test_lru_rejects_invalid_configuration(max_size: int, ttl: float | None) -> None:
    with pytest.raises(ValueError):
        LRUCache("unit_invalid", max_size=max_size, ttl_seconds=ttl)


class CountingEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def embed_text(self, text: str) -> list[float]:
        self.calls += 1
        return [float(len(text)), 1.0]

    def embed(self, chunk: DocumentChunk) -> EmbeddedChunk:
        return EmbeddedChunk(chunk=chunk, vector=[0.0])

    def embed_many(self, chunks: list[DocumentChunk]) -> list[EmbeddedChunk]:
        return [self.embed(chunk) for chunk in chunks]


def test_cached_embedder_embeds_each_query_once() -> None:
    inner = CountingEmbedder()
    embedder = CachedQueryEmbedder(inner, LRUCache("unit_embed", max_size=8))

    first = embedder.embed_text("hello")
    second = embedder.embed_text("hello")
    embedder.embed_text("other")

    assert first == second == [5.0, 1.0]
    assert inner.calls == 2


def test_cached_embedder_does_not_leak_mutations_into_the_cache() -> None:
    embedder = CachedQueryEmbedder(CountingEmbedder(), LRUCache("unit_embed_copy", max_size=8))

    embedder.embed_text("hello").append(99.0)

    assert embedder.embed_text("hello") == [5.0, 1.0]


class CountingRetriever:
    def __init__(self) -> None:
        self.calls = 0

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        self.calls += 1
        return [make_result(f"{query}-{top_k}")]


def test_caching_retriever_reuses_results_per_query_and_top_k() -> None:
    inner = CountingRetriever()
    retriever = CachingRetriever(inner, LRUCache("unit_retrieval", max_size=8))

    first = retriever.search("q", top_k=3)
    second = retriever.search("q", top_k=3)
    retriever.search("q", top_k=5)

    assert first == second
    assert inner.calls == 2


def test_caching_retriever_returns_independent_lists() -> None:
    retriever = CachingRetriever(
        CountingRetriever(),
        LRUCache("unit_retrieval_copy", max_size=8),
    )

    retriever.search("q").clear()

    assert len(retriever.search("q")) == 1
