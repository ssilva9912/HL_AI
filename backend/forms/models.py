from dataclasses import dataclass


class UnsupportedFormError(ValueError):
    """Raised for a file that is not a Word or PDF form."""


class InvalidFormError(ValueError):
    """Raised when a form file cannot be read."""


@dataclass(frozen=True)
class FormField:
    id: str
    label: str
    kind: str = "text"


@dataclass(frozen=True)
class FieldProposal:
    field: FormField
    value: str | None = None
    verified: bool = False
    source: str | None = None
    snippet: str | None = None
