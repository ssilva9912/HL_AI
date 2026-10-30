import re
from collections.abc import Mapping
from io import BytesIO
from typing import Any

from pypdf import PdfReader, PdfWriter
from pypdf.errors import PyPdfError

from backend.forms.models import FormField, InvalidFormError

_PUSH_BUTTON = 1 << 16
_RADIO = 1 << 15
_TRUTHY = {"1", "true", "yes", "y", "x", "on", "checked"}


def _read(data: bytes) -> PdfReader:
    try:
        return PdfReader(BytesIO(data))
    except (PyPdfError, ValueError, OSError) as exc:
        raise InvalidFormError("The PDF could not be read.") from exc


def _humanize(name: str) -> str:
    words = re.sub(r"[_\-.]+", " ", re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)).strip()
    return words[:1].upper() + words[1:].lower() if words else name


def _field_kind(field: Mapping[str, Any]) -> str | None:
    field_type = field.get("/FT")
    flags = int(field.get("/Ff", 0) or 0)

    if field_type == "/Tx":
        return "text"
    if field_type == "/Btn" and not flags & (_PUSH_BUTTON | _RADIO):
        return "checkbox"
    return None


def detect_pdf_fields(data: bytes) -> list[FormField]:
    reader = _read(data)

    try:
        raw_fields = reader.get_fields() or {}
    except (PyPdfError, ValueError, KeyError) as exc:
        raise InvalidFormError("The PDF form fields could not be read.") from exc

    fields: list[FormField] = []
    for name, field in raw_fields.items():
        kind = _field_kind(field)
        if kind is None:
            continue
        label = str(field.get("/TU") or "").strip() or _humanize(name)
        fields.append(FormField(id=name, label=label, kind=kind))

    return fields


def _checkbox_on_state(field: Mapping[str, Any]) -> str:
    states = [str(state) for state in field.get("/_States_", [])]
    return next((state for state in states if state != "/Off"), "/Yes")


def fill_pdf(data: bytes, values: Mapping[str, str]) -> bytes:
    reader = _read(data)
    writer = PdfWriter(clone_from=reader)
    fields = {field.id: field for field in detect_pdf_fields(data)}
    raw_fields = reader.get_fields() or {}

    updates: dict[str, str] = {}
    for name, value in values.items():
        field = fields.get(name)
        cleaned = value.strip()
        if field is None or not cleaned:
            continue
        if field.kind == "checkbox":
            if cleaned.lower() in _TRUTHY:
                updates[name] = _checkbox_on_state(raw_fields[name])
        else:
            updates[name] = cleaned

    if updates:
        for page in writer.pages:
            writer.update_page_form_field_values(page, updates, auto_regenerate=False)
        writer.set_need_appearances_writer(True)

    output = BytesIO()
    writer.write(output)
    return output.getvalue()
