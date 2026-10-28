from dataclasses import dataclass

import httpx


class SearchError(RuntimeError):
    """The search backend could not be reached or returned something unusable."""


@dataclass(frozen=True)
class SearchHit:
    title: str
    url: str
    snippet: str


class SearxngClient:
    """Queries a SearXNG instance (JSON output must be enabled in its settings)."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 10.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url must not be empty")
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        self._base_url = base_url.strip().rstrip("/")
        self._client = client or httpx.Client(timeout=timeout)

    def search(self, query: str, limit: int = 5) -> list[SearchHit]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if limit <= 0:
            raise ValueError("limit must be positive")

        try:
            response = self._client.get(
                f"{self._base_url}/search",
                params={"q": query.strip(), "format": "json"},
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as error:
            raise SearchError(f"search request failed: {error}") from error
        except ValueError as error:
            raise SearchError(
                "the search backend did not return JSON; enable the json format in SearXNG"
            ) from error

        raw_results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(raw_results, list):
            raise SearchError("the search backend returned an unexpected response")

        hits: list[SearchHit] = []
        seen: set[str] = set()
        for raw in raw_results:
            if not isinstance(raw, dict):
                continue
            url = raw.get("url")
            if not isinstance(url, str) or not url or url in seen:
                continue
            seen.add(url)
            hits.append(
                SearchHit(
                    title=str(raw.get("title") or url),
                    url=url,
                    snippet=str(raw.get("content") or ""),
                )
            )
            if len(hits) == limit:
                break
        return hits
