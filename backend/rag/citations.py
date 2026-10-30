from collections.abc import Sequence

from backend.interfaces.retriever import RetrievalResult


def citation_numbers(results: Sequence[RetrievalResult]) -> list[int]:
    """
    Number each result by the document it came from, in order of first appearance.

    Passages from the same page or file share a number, so the prompt and the source list
    shown to the user refer to the same [1], [2], ... for the same document.
    """

    numbers: dict[str, int] = {}
    assigned: list[int] = []

    for result in results:
        document = result.chunk.source_document
        key = document.url or str(document.source_path)
        assigned.append(numbers.setdefault(key, len(numbers) + 1))

    return assigned
