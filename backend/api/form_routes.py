import json
import logging
from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status
from pydantic import BaseModel, Field

from backend.api.dependencies import get_rag_service
from backend.api.rag_service import HomelabRAGService
from backend.config import Settings, get_settings
from backend.forms import (
    InvalidFormError,
    UnsupportedFormError,
    detect_fields,
    fill_form,
    filled_filename,
    media_type,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/forms", tags=["forms"])


class FormFieldResponse(BaseModel):
    id: str
    label: str
    kind: str = Field(description="text or checkbox.")
    value: str | None = Field(default=None, description="Suggested value from your documents.")
    verified: bool = Field(
        default=False,
        description="Whether the suggested value appears word for word in your documents.",
    )
    source: str | None = None
    snippet: str | None = None


class FormProposalResponse(BaseModel):
    filename: str
    fields: list[FormFieldResponse]
    elapsed_ms: float


def _read_upload(file: UploadFile, settings: Settings) -> tuple[str, bytes]:
    filename = file.filename or ""
    content = file.file.read(settings.max_upload_bytes + 1)

    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="The form is larger than the upload limit.",
        )
    if not content:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="The form is empty.",
        )
    return filename, content


@router.post("/propose", response_model=FormProposalResponse)
def propose_form_values(
    file: Annotated[UploadFile, File(description="A Word (.docx) or fillable PDF form.")],
    rag_service: Annotated[HomelabRAGService, Depends(get_rag_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> FormProposalResponse:
    started_at = perf_counter()
    filename, content = _read_upload(file, settings)

    try:
        fields = detect_fields(filename, content)
    except UnsupportedFormError as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)) from exc
    except InvalidFormError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc

    if not fields:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                "No fillable fields were found. Use a fillable PDF, or a Word document with "
                'lines like "Name: ____", empty table cells next to labels, or {{placeholders}}.'
            ),
        )

    try:
        proposals = rag_service.propose_form_values(fields)
    except RuntimeError as exc:
        logger.exception("Form values could not be proposed")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    elapsed_ms = (perf_counter() - started_at) * 1_000
    logger.info("Form analysed fields=%d elapsed_ms=%.2f", len(fields), elapsed_ms)

    return FormProposalResponse(
        filename=filename,
        elapsed_ms=elapsed_ms,
        fields=[
            FormFieldResponse(
                id=proposal.field.id,
                label=proposal.field.label,
                kind=proposal.field.kind,
                value=proposal.value,
                verified=proposal.verified,
                source=proposal.source,
                snippet=proposal.snippet,
            )
            for proposal in proposals
        ],
    )


@router.post("/fill")
def fill_form_document(
    file: Annotated[UploadFile, File(description="The form to fill.")],
    values: Annotated[str, Form(description='JSON object of {"field id": "value"}.')],
    settings: Annotated[Settings, Depends(get_settings)],
) -> Response:
    filename, content = _read_upload(file, settings)

    try:
        parsed = json.loads(values)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="values must be a JSON object.",
        ) from exc

    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="values must be a JSON object.",
        )
    cleaned = {str(key): str(value) for key, value in parsed.items() if value is not None}

    try:
        filled = fill_form(filename, content, cleaned)
    except UnsupportedFormError as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)) from exc
    except InvalidFormError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc

    return Response(
        content=filled,
        media_type=media_type(filename),
        headers={"Content-Disposition": f'attachment; filename="{filled_filename(filename)}"'},
    )
