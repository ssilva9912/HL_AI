import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

_CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\](?!\()")


@dataclass(frozen=True)
class Citation:
    number: int
    label: str
    url: str | None = None
    passages: tuple[dict[str, Any], ...] = field(default_factory=tuple)


def safe_url(url: object) -> str | None:
    """Only plain http(s) addresses become links; anything else is shown as text."""

    if not isinstance(url, str):
        return None
    parts = urlsplit(url.strip())
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return None
    return url.strip().replace(" ", "%20").replace("(", "%28").replace(")", "%29")


def _escape_label(label: str) -> str:
    return label.replace("[", "\\[").replace("]", "\\]")


def group_citations(sources: list[dict[str, Any]]) -> list[Citation]:
    """
    Collapse retrieved passages into one numbered entry per document.

    Sources saved before citations were numbered have no number, so each passage is its own
    entry in the order shown.
    """

    grouped: dict[int, list[dict[str, Any]]] = {}
    for index, source in enumerate(sources, start=1):
        raw_number = source.get("citation")
        number = raw_number if isinstance(raw_number, int) and raw_number > 0 else index
        grouped.setdefault(number, []).append(source)

    return [
        Citation(
            number=number,
            label=str(passages[0].get("document") or "Unknown source"),
            url=next((url for p in passages if (url := safe_url(p.get("url")))), None),
            passages=tuple(passages),
        )
        for number, passages in sorted(grouped.items())
    ]


def format_citation(citation: Citation) -> str:
    """One line of the numbered source list at the bottom of an answer."""

    label = _escape_label(citation.label)
    target = f"[{label}]({citation.url})" if citation.url else label
    return f"**[{citation.number}]** {target}"


def link_citations(text: str, citations: list[Citation]) -> str:
    """Turn [n] markers in an answer into links to the web page they cite."""

    links = {citation.number: citation.url for citation in citations if citation.url}
    if not links:
        return text

    def replace(match: re.Match[str]) -> str:
        numbers = [int(part) for part in match.group(1).split(",")]
        if not all(number in links for number in numbers):
            return match.group(0)
        return "".join(f"[[{number}]]({links[number]})" for number in numbers)

    return _CITATION_PATTERN.sub(replace, text)
