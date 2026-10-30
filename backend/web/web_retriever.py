import logging
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha1
from pathlib import Path
from urllib.parse import urlsplit

from backend.cache import LRUCache
from backend.ingestion.scanner import FileMetadata
from backend.interfaces.chunker import Chunker, DocumentChunk
from backend.interfaces.parser import ParsedDocument
from backend.interfaces.retriever import RetrievalResult
from backend.observability import stage
from backend.observability.metrics import WEB_REQUESTS
from backend.retrieval.bm25_retriever import BM25Retriever
from backend.web.safe_http import FetchedPage, FetchError, PageFetcher, UnsafeURLError
from backend.web.search import SearchError, SearchHit, SearxngClient

logger = logging.getLogger(__name__)

WEB_FILE_TYPE = "web"
_MAX_DISPLAY_NAME = 80


def is_web_chunk(chunk: DocumentChunk) -> bool:
    return chunk.source_document.file_type == WEB_FILE_TYPE


def display_name(url: str) -> str:
    parts = urlsplit(url)
    host = (parts.hostname or url).removeprefix("www.")
    name = f"{host}{parts.path}".rstrip("/")
    return name[:_MAX_DISPLAY_NAME]


def _interleave_by_page(results: list[RetrievalResult]) -> list[RetrievalResult]:
    """
    Orders results so each page's best chunk comes before any page's second chunk.

    Without this a single long page can fill every slot, which both hides other
    sources and lets one stale article decide the answer.
    """
    pages: dict[str, list[RetrievalResult]] = {}
    for result in results:
        pages.setdefault(str(result.chunk.source_document.source_path), []).append(result)

    interleaved: list[RetrievalResult] = []
    for depth in range(max((len(page) for page in pages.values()), default=0)):
        interleaved.extend(page[depth] for page in pages.values() if len(page) > depth)
    return interleaved


class WebRetriever:
    """
    Retrieves passages from the live web for a question.

    Searches, downloads the top pages in parallel, splits them into chunks and ranks
    those chunks against the question. A failing page is skipped, and a failing search
    backend yields no results rather than an error, so local documents still answer.
    Fetched pages are cached per question because search results are slow to gather.
    """

    def __init__(
        self,
        search_client: SearxngClient,
        fetcher: PageFetcher,
        chunker: Chunker,
        max_results: int = 4,
        cache: LRUCache[str, tuple[DocumentChunk, ...]] | None = None,
    ) -> None:
        if max_results <= 0:
            raise ValueError("max_results must be positive")

        self._search_client = search_client
        self._fetcher = fetcher
        self._chunker = chunker
        self._max_results = max_results
        self._cache = cache

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        chunks = self._cache.get(query) if self._cache is not None else None
        if chunks is None:
            chunks = self._gather(query)
            if chunks and self._cache is not None:
                self._cache.put(query, chunks)
        if not chunks:
            return []

        ranked = BM25Retriever(chunks).search(query=query, top_k=len(chunks))
        return [
            RetrievalResult(chunk=result.chunk, score=result.score, retriever="web")
            for result in _interleave_by_page(ranked)[:top_k]
        ]

    def _gather(self, query: str) -> tuple[DocumentChunk, ...]:
        with stage("retrieval.web_search"):
            try:
                hits = self._search_client.search(query, self._max_results)
            except SearchError:
                WEB_REQUESTS.labels(operation="search", outcome="error").inc()
                logger.warning("Web search failed; continuing without web results", exc_info=True)
                return ()
        WEB_REQUESTS.labels(operation="search", outcome="ok").inc()

        with stage("retrieval.web_fetch"), ThreadPoolExecutor(max_workers=len(hits) or 1) as pool:
            pages = list(pool.map(self._fetch, hits))

        chunks: list[DocumentChunk] = []
        for page in pages:
            if page is not None:
                chunks.extend(self._chunker.chunk(self._document(page)))
        return tuple(chunks)

    def _fetch(self, hit: SearchHit) -> FetchedPage | None:
        try:
            page = self._fetcher.fetch(hit.url)
        except UnsafeURLError:
            WEB_REQUESTS.labels(operation="fetch", outcome="blocked").inc()
            logger.warning("Blocked unsafe web result url=%s", hit.url)
            return None
        except FetchError as error:
            WEB_REQUESTS.labels(operation="fetch", outcome="error").inc()
            logger.info("Could not fetch web result url=%s reason=%s", hit.url, error)
            if hit.snippet.strip():
                return FetchedPage(url=hit.url, title=hit.title, text=hit.snippet.strip())
            return None
        WEB_REQUESTS.labels(operation="fetch", outcome="ok").inc()
        return page

    @staticmethod
    def _document(page: FetchedPage) -> ParsedDocument:
        path = Path("web") / sha1(page.url.encode("utf-8")).hexdigest()[:16]
        text = f"{page.title}\n{page.text}" if page.title else page.text
        return ParsedDocument(
            source_path=path,
            file_type=WEB_FILE_TYPE,
            url=page.url,
            content=text,
            metadata=FileMetadata(
                name=display_name(page.url),
                path=path,
                extension=".html",
                size_bytes=len(text),
            ),
        )
