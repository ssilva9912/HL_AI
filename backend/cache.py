from collections import OrderedDict
from collections.abc import Callable
from threading import Lock
from time import monotonic

from backend.interfaces.chunker import DocumentChunk
from backend.interfaces.embedder import EmbeddedChunk, EmbeddingProvider
from backend.interfaces.retriever import RetrievalResult, Retriever
from backend.observability.metrics import CACHE_EVENTS


class LRUCache[K, V]:
    """Thread-safe LRU cache with an optional time-to-live, reporting hits and misses."""

    def __init__(
        self,
        name: str,
        max_size: int,
        ttl_seconds: float | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if max_size < 0:
            raise ValueError("max_size cannot be negative")

        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")

        self._name = name
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._entries: OrderedDict[K, tuple[float, V]] = OrderedDict()
        self._lock = Lock()

    def get(self, key: K) -> V | None:
        if self._max_size == 0:
            return None

        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                stored_at, value = entry
                if self._ttl_seconds is None or self._clock() - stored_at < self._ttl_seconds:
                    self._entries.move_to_end(key)
                    CACHE_EVENTS.labels(cache=self._name, result="hit").inc()
                    return value
                del self._entries[key]

        CACHE_EVENTS.labels(cache=self._name, result="miss").inc()
        return None

    def put(self, key: K, value: V) -> None:
        if self._max_size == 0:
            return

        with self._lock:
            self._entries[key] = (self._clock(), value)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_size:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


class CachedQueryEmbedder:
    """Memoizes query embeddings; ingestion paths keep using the uncached embedder."""

    def __init__(
        self,
        embedder: EmbeddingProvider,
        cache: LRUCache[str, list[float]],
    ) -> None:
        self._embedder = embedder
        self._cache = cache

    def embed_text(self, text: str) -> list[float]:
        cached = self._cache.get(text)
        if cached is not None:
            return list(cached)

        vector = self._embedder.embed_text(text)
        self._cache.put(text, list(vector))
        return vector

    def embed(self, chunk: DocumentChunk) -> EmbeddedChunk:
        return self._embedder.embed(chunk)

    def embed_many(self, chunks: list[DocumentChunk]) -> list[EmbeddedChunk]:
        return self._embedder.embed_many(chunks)


class CachingRetriever:
    """
    Memoizes search results for a retriever over a fixed corpus.

    Create one per corpus snapshot: a new corpus means a new retriever, which is
    what invalidates cached results after documents are ingested or deleted.
    """

    def __init__(
        self,
        retriever: Retriever,
        cache: LRUCache[tuple[str, int], tuple[RetrievalResult, ...]],
    ) -> None:
        self._retriever = retriever
        self._cache = cache

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        key = (query, top_k)
        cached = self._cache.get(key)
        if cached is not None:
            return list(cached)

        results = self._retriever.search(query=query, top_k=top_k)
        self._cache.put(key, tuple(results))
        return results
