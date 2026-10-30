import logging
import re
from collections.abc import Callable, Sequence

from backend.forms.models import FieldProposal, FormField
from backend.interfaces.generator import Generator
from backend.interfaces.retriever import RetrievalResult
from backend.observability import stage

logger = logging.getLogger(__name__)

_MAX_VALUE_CHARS = 200
_MAX_CONTEXT_CHARS = 2_500
_SNIPPET_CHARS = 200
_NOT_FOUND = re.compile(
    r"^\W*(not found|n/a|none|unknown|i (do not|don't|dont) know|no information)\b",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    return re.sub(r"[\W_]+", " ", text.casefold()).strip()


class FieldExtractor:
    """
    Proposes a value for each form field using only the user's own documents.

    A value is "verified" when it appears word for word in the passages it came from, so a
    guess the model made up shows as unverified and is flagged for the user to check.
    """

    def __init__(
        self,
        retrieve: Callable[[str], Sequence[RetrievalResult]],
        generator: Generator,
    ) -> None:
        self._retrieve = retrieve
        self._generator = generator

    def propose_all(self, fields: Sequence[FormField]) -> list[FieldProposal]:
        return [self.propose(field) for field in fields]

    def propose(self, field: FormField) -> FieldProposal:
        if field.kind != "text":
            return FieldProposal(field=field)

        with stage("forms.extract"):
            passages = list(self._retrieve(field.label))
            if not passages:
                return FieldProposal(field=field)

            try:
                reply = self._generator.generate(self._prompt(field, passages))
            except Exception:
                logger.warning("Could not extract a form field value", exc_info=True)
                return FieldProposal(field=field)

        value = self._clean(reply)
        if value is None:
            return FieldProposal(field=field)

        source, snippet = self._locate(value, passages)
        return FieldProposal(
            field=field,
            value=value,
            verified=source is not None,
            source=source,
            snippet=snippet,
        )

    @staticmethod
    def _clean(reply: str) -> str | None:
        for line in reply.splitlines():
            value = line.strip().strip("\"'`").strip()
            if not value:
                continue
            if _NOT_FOUND.match(value) or len(value) > _MAX_VALUE_CHARS:
                return None
            return value
        return None

    @staticmethod
    def _locate(value: str, passages: Sequence[RetrievalResult]) -> tuple[str | None, str | None]:
        wanted = _normalize(value)
        for passage in passages:
            if wanted and wanted in _normalize(passage.chunk.content):
                text = " ".join(passage.chunk.content.split())
                index = text.casefold().find(value.casefold())
                start = max(0, index - 60) if index >= 0 else 0
                return (
                    passage.chunk.source_document.metadata.name,
                    text[start : start + _SNIPPET_CHARS],
                )
        return None, None

    @staticmethod
    def _prompt(field: FormField, passages: Sequence[RetrievalResult]) -> str:
        context = ""
        for number, passage in enumerate(passages, start=1):
            section = f"[{number}] {passage.chunk.content.strip()}\n\n"
            if len(context) + len(section) > _MAX_CONTEXT_CHARS:
                break
            context += section

        return (
            "You are filling in one field of a form for the user, using only the "
            "information below.\n"
            "- Copy the value exactly as it is written in the information.\n"
            "- Do not guess, infer, calculate or combine details from different places.\n"
            "- If the information does not clearly state it, reply NOT FOUND.\n"
            "- Reply with only the value, nothing else.\n\n"
            f"FIELD\n{field.label}\n\n"
            f"INFORMATION\n{context.strip()}\n\n"
            "VALUE\n"
        )
