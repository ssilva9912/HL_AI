from uuid import UUID

from sqlalchemy.orm import Session

from backend.database.models import IngestionPayload


def _validate_sha256(checksum: str) -> str:
    normalized_checksum = checksum.lower()

    if len(normalized_checksum) != 64:
        raise ValueError(
            "SHA-256 checksum must contain exactly 64 characters.",
        )

    try:
        int(normalized_checksum, 16)
    except ValueError as error:
        raise ValueError(
            "SHA-256 checksum must be hexadecimal.",
        ) from error

    return normalized_checksum


class IngestionPayloadRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        *,
        job_id: UUID,
        staged_path: str,
        content_type: str | None,
        size_bytes: int,
        checksum_sha256: str,
    ) -> IngestionPayload:
        normalized_staged_path = staged_path.strip()

        if not normalized_staged_path:
            raise ValueError(
                "Staged upload path cannot be empty.",
            )

        if size_bytes < 0:
            raise ValueError(
                "Staged upload size cannot be negative.",
            )

        normalized_content_type = content_type.strip() if content_type is not None else None

        payload = IngestionPayload(
            job_id=job_id,
            staged_path=normalized_staged_path,
            content_type=normalized_content_type or None,
            size_bytes=size_bytes,
            checksum_sha256=_validate_sha256(
                checksum_sha256,
            ),
        )

        self._session.add(payload)
        self._session.flush()

        return payload

    def get(
        self,
        job_id: UUID,
    ) -> IngestionPayload | None:
        return self._session.get(
            IngestionPayload,
            job_id,
        )

    def delete(
        self,
        payload: IngestionPayload,
    ) -> None:
        self._session.delete(payload)
        self._session.flush()
