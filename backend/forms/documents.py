from collections.abc import Mapping
from pathlib import PurePath

from backend.forms.docx_form import detect_docx_fields, fill_docx
from backend.forms.models import FormField, UnsupportedFormError
from backend.forms.pdf_form import detect_pdf_fields, fill_pdf

DOCX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
PDF_MEDIA_TYPE = "application/pdf"


def form_kind(filename: str) -> str:
    suffix = PurePath(filename).suffix.lower()
    if suffix == ".docx":
        return "docx"
    if suffix == ".pdf":
        return "pdf"
    raise UnsupportedFormError("Forms must be Word (.docx) documents or fillable PDFs.")


def media_type(filename: str) -> str:
    return DOCX_MEDIA_TYPE if form_kind(filename) == "docx" else PDF_MEDIA_TYPE


def detect_fields(filename: str, data: bytes) -> list[FormField]:
    if form_kind(filename) == "docx":
        return detect_docx_fields(data)
    return detect_pdf_fields(data)


def fill_form(filename: str, data: bytes, values: Mapping[str, str]) -> bytes:
    if form_kind(filename) == "docx":
        return fill_docx(data, values)
    return fill_pdf(data, values)


def filled_filename(filename: str) -> str:
    path = PurePath(filename)
    return f"{path.stem}-filled{path.suffix.lower()}"
