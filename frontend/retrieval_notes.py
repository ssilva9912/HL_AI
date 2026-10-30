from typing import Any


def describe_correction(correction: dict[str, Any] | None, web_enabled: bool) -> str | None:
    """Explain, in one sentence, how well the documents covered the question."""

    if not correction:
        return None

    verdict = correction.get("verdict")
    web_used = bool(correction.get("web_used"))
    query = correction.get("search_query")
    searched = f" for “{query}”" if query else ""

    if verdict == "correct":
        return None

    if verdict == "ambiguous":
        if web_used:
            return f"Your documents only partly cover this, so I also searched the web{searched}."
        if web_enabled:
            return "Your documents only partly cover this, and the web search found nothing usable."
        return "Your documents only partly cover this. Tick “Search the web” to add online results."

    if verdict == "incorrect":
        if web_used:
            return f"Your documents don't cover this, so I searched the web{searched}."
        if web_enabled:
            return "Your documents don't cover this, and the web search found nothing usable."
        return "Your documents don't cover this. Tick “Search the web” to look online."

    return None


_ABSTENTION_OPENINGS = ("i don't know", "i do not know", "i dont know")


def looks_like_abstention(answer: str) -> bool:
    """Whether the model opened by saying it could not answer from what it was given."""

    opening = answer.strip().lower().lstrip("\"'*_ ").replace("’", "'")
    return opening.startswith(_ABSTENTION_OPENINGS)
