from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.database.models import (
    Conversation,
    ConversationMessage,
    Document,
    DocumentStatus,
    IngestionJob,
    IngestionJobStatus,
    IngestionOperation,
    IngestionStage,
    MessageRole,
    NetworkShareFile,
    NetworkShareFileStatus,
    NetworkShareSource,
    NetworkShareStatus,
)


def _validate_sha256(checksum: str) -> str:
    normalized_checksum = checksum.lower()

    if len(normalized_checksum) != 64:
        raise ValueError("SHA-256 checksum must contain exactly 64 characters.")

    try:
        int(normalized_checksum, 16)
    except ValueError as error:
        raise ValueError("SHA-256 checksum must be hexadecimal.") from error

    return normalized_checksum


def _validate_pagination(offset: int, limit: int) -> None:
    if offset < 0:
        raise ValueError("Pagination offset cannot be negative.")

    if limit < 1 or limit > 1000:
        raise ValueError("Pagination limit must be between 1 and 1000.")


def _normalize_content_type(content_type: str | None) -> str | None:
    if content_type is None:
        return None

    normalized_content_type = content_type.strip()

    return normalized_content_type or None


class DocumentRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        *,
        filename: str,
        storage_path: str,
        content_type: str | None,
        size_bytes: int,
        checksum_sha256: str,
    ) -> Document:
        normalized_filename = filename.strip()
        normalized_storage_path = storage_path.strip()

        if not normalized_filename:
            raise ValueError("Document filename cannot be empty.")

        if not normalized_storage_path:
            raise ValueError("Document storage path cannot be empty.")

        if size_bytes < 0:
            raise ValueError("Document size cannot be negative.")

        document = Document(
            filename=normalized_filename,
            storage_path=normalized_storage_path,
            content_type=_normalize_content_type(content_type),
            size_bytes=size_bytes,
            checksum_sha256=_validate_sha256(checksum_sha256),
        )

        self._session.add(document)
        self._session.flush()

        return document

    def update_content(
        self,
        document: Document,
        *,
        filename: str,
        content_type: str | None,
        size_bytes: int,
        checksum_sha256: str,
    ) -> Document:
        normalized_filename = filename.strip()

        if not normalized_filename:
            raise ValueError("Document filename cannot be empty.")

        if size_bytes < 0:
            raise ValueError("Document size cannot be negative.")

        document.filename = normalized_filename
        document.content_type = _normalize_content_type(content_type)
        document.size_bytes = size_bytes
        document.checksum_sha256 = _validate_sha256(checksum_sha256)

        self._session.flush()

        return document

    def get(self, document_id: UUID) -> Document | None:
        return self._session.get(Document, document_id)

    def get_by_storage_path(self, storage_path: str) -> Document | None:
        statement = select(Document).where(
            Document.storage_path == storage_path,
        )
        return self._session.scalar(statement)

    def list_by_checksum(self, checksum_sha256: str) -> list[Document]:
        statement = (
            select(Document)
            .where(
                Document.checksum_sha256 == _validate_sha256(checksum_sha256),
            )
            .order_by(Document.created_at.desc(), Document.id)
        )
        return list(self._session.scalars(statement))

    def list_all(
        self,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> list[Document]:
        _validate_pagination(offset, limit)

        statement = (
            select(Document)
            .order_by(Document.created_at.desc(), Document.id)
            .offset(offset)
            .limit(limit)
        )
        return list(self._session.scalars(statement))

    def update_status(
        self,
        document: Document,
        status: DocumentStatus,
        *,
        chunk_count: int | None = None,
        error_message: str | None = None,
    ) -> Document:
        if chunk_count is not None and chunk_count < 0:
            raise ValueError("Document chunk count cannot be negative.")

        document.status = status
        document.error_message = error_message

        if chunk_count is not None:
            document.chunk_count = chunk_count

        self._session.flush()

        return document

    def delete(self, document: Document) -> None:
        self._session.delete(document)
        self._session.flush()


class NetworkShareSourceRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get_or_create(self, *, name: str, root_path: str) -> NetworkShareSource:
        normalized_name = name.strip()
        normalized_root = root_path.strip()
        if not normalized_name:
            raise ValueError("Network-share name cannot be empty.")
        if not normalized_root:
            raise ValueError("Network-share root path cannot be empty.")

        source = self._session.scalar(
            select(NetworkShareSource).where(NetworkShareSource.name == normalized_name)
        )
        if source is None:
            source = NetworkShareSource(name=normalized_name, root_path=normalized_root)
            self._session.add(source)
            self._session.flush()
        elif source.root_path != normalized_root:
            source.root_path = normalized_root
            self._session.flush()
        return source

    def update_scan_status(
        self,
        source: NetworkShareSource,
        *,
        status: NetworkShareStatus,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        error: str | None = None,
    ) -> NetworkShareSource:
        source.status = status
        source.last_error = error
        if started_at is not None:
            source.last_scan_started_at = started_at
        if completed_at is not None:
            source.last_scan_completed_at = completed_at
        self._session.flush()
        return source


class NetworkShareFileRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get_by_relative_path(
        self,
        source_id: UUID,
        relative_path: str,
    ) -> NetworkShareFile | None:
        return self._session.scalar(
            select(NetworkShareFile).where(
                NetworkShareFile.source_id == source_id,
                NetworkShareFile.relative_path == relative_path,
            )
        )

    def record_seen(
        self,
        *,
        source: NetworkShareSource,
        relative_path: str,
        checksum_sha256: str,
        size_bytes: int,
        modified_time_ns: int,
        seen_at: datetime,
    ) -> tuple[NetworkShareFile, bool, bool]:
        normalized_path = relative_path.strip().replace("\\", "/")
        if not normalized_path or normalized_path.startswith("/"):
            raise ValueError("A safe share-relative path is required.")
        if size_bytes < 0 or modified_time_ns < 0:
            raise ValueError("File size and modification time cannot be negative.")

        tracked = self.get_by_relative_path(source.id, normalized_path)
        validated_checksum = _validate_sha256(checksum_sha256)
        is_new = tracked is None
        if tracked is None:
            changed = True
            tracked = NetworkShareFile(
                source=source,
                relative_path=normalized_path,
                checksum_sha256=validated_checksum,
                size_bytes=size_bytes,
                modified_time_ns=modified_time_ns,
                status=NetworkShareFileStatus.DISCOVERED,
                last_seen_at=seen_at,
            )
            self._session.add(tracked)
        else:
            changed = (
                tracked.checksum_sha256 != validated_checksum
                or tracked.status is NetworkShareFileStatus.MISSING
            )
            tracked.checksum_sha256 = validated_checksum
            tracked.size_bytes = size_bytes
            tracked.modified_time_ns = modified_time_ns
            tracked.last_seen_at = seen_at
            tracked.last_error = None
            if changed or tracked.status is NetworkShareFileStatus.MISSING:
                tracked.status = NetworkShareFileStatus.DISCOVERED
        self._session.flush()
        return tracked, is_new, changed

    def mark_unseen_missing(
        self,
        *,
        source_id: UUID,
        seen_paths: set[str],
    ) -> int:
        tracked_files = list(
            self._session.scalars(
                select(NetworkShareFile).where(NetworkShareFile.source_id == source_id)
            )
        )
        marked = 0
        for tracked in tracked_files:
            if (
                tracked.relative_path not in seen_paths
                and tracked.status is not NetworkShareFileStatus.MISSING
            ):
                tracked.status = NetworkShareFileStatus.MISSING
                marked += 1
        self._session.flush()
        return marked


class IngestionJobRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        *,
        document_id: UUID,
        operation: IngestionOperation = IngestionOperation.INDEX,
    ) -> IngestionJob:
        job = IngestionJob(
            document_id=document_id,
            operation=operation,
        )

        self._session.add(job)
        self._session.flush()

        return job

    def get(self, job_id: UUID) -> IngestionJob | None:
        return self._session.get(IngestionJob, job_id)

    def list_for_document(
        self,
        document_id: UUID,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> list[IngestionJob]:
        _validate_pagination(offset, limit)

        statement = (
            select(IngestionJob)
            .where(IngestionJob.document_id == document_id)
            .order_by(IngestionJob.created_at.desc(), IngestionJob.id)
            .offset(offset)
            .limit(limit)
        )
        return list(self._session.scalars(statement))

    def mark_running(
        self,
        job: IngestionJob,
        *,
        total_chunks: int | None = None,
    ) -> IngestionJob:
        if total_chunks is not None and total_chunks < 0:
            raise ValueError("Total chunk count cannot be negative.")

        job.status = IngestionJobStatus.RUNNING
        job.stage = IngestionStage.PARSING
        job.attempt_count += 1
        job.total_chunks = total_chunks
        job.processed_chunks = 0
        job.error_message = None
        job.started_at = datetime.now(UTC)
        job.completed_at = None

        self._session.flush()

        return job

    def update_stage(
        self,
        job: IngestionJob,
        stage: IngestionStage,
        *,
        processed_chunks: int | None = None,
        total_chunks: int | None = None,
    ) -> IngestionJob:
        job.stage = stage

        if processed_chunks is not None or total_chunks is not None:
            self.update_progress(
                job,
                processed_chunks=(
                    processed_chunks if processed_chunks is not None else job.processed_chunks
                ),
                total_chunks=total_chunks,
            )
        else:
            self._session.flush()

        return job

    def update_progress(
        self,
        job: IngestionJob,
        *,
        processed_chunks: int,
        total_chunks: int | None = None,
    ) -> IngestionJob:
        effective_total = total_chunks
        if effective_total is None:
            effective_total = job.total_chunks

        if processed_chunks < 0:
            raise ValueError("Processed chunk count cannot be negative.")

        if effective_total is not None:
            if effective_total < 0:
                raise ValueError("Total chunk count cannot be negative.")

            if processed_chunks > effective_total:
                raise ValueError(
                    "Processed chunk count cannot exceed the total.",
                )

        job.processed_chunks = processed_chunks
        job.total_chunks = effective_total

        self._session.flush()

        return job

    def mark_succeeded(self, job: IngestionJob) -> IngestionJob:
        job.status = IngestionJobStatus.SUCCEEDED
        job.stage = IngestionStage.SUCCEEDED
        job.error_message = None
        job.completed_at = datetime.now(UTC)

        if job.total_chunks is not None:
            job.processed_chunks = job.total_chunks

        self._session.flush()

        return job

    def mark_failed(
        self,
        job: IngestionJob,
        *,
        error_message: str,
    ) -> IngestionJob:
        normalized_error = error_message.strip()

        if not normalized_error:
            raise ValueError("Failed ingestion jobs require an error message.")

        job.status = IngestionJobStatus.FAILED
        job.stage = IngestionStage.FAILED
        job.error_message = normalized_error
        job.completed_at = datetime.now(UTC)

        self._session.flush()

        return job

    def delete(self, job: IngestionJob) -> None:
        self._session.delete(job)
        self._session.flush()


class ConversationRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        *,
        title: str = "New conversation",
        owner_key: str = "local",
    ) -> Conversation:
        normalized_title = title.strip()
        normalized_owner = owner_key.strip()
        if not normalized_title:
            raise ValueError("Conversation title cannot be empty.")
        if not normalized_owner:
            raise ValueError("Conversation owner cannot be empty.")

        conversation = Conversation(
            title=normalized_title,
            owner_key=normalized_owner,
        )
        self._session.add(conversation)
        self._session.flush()
        return conversation

    def get(
        self,
        conversation_id: UUID,
        *,
        owner_key: str = "local",
    ) -> Conversation | None:
        statement = select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.owner_key == owner_key,
        )
        return self._session.scalar(statement)

    def list_all(
        self,
        *,
        owner_key: str = "local",
        offset: int = 0,
        limit: int = 100,
    ) -> list[Conversation]:
        _validate_pagination(offset, limit)
        statement = (
            select(Conversation)
            .where(Conversation.owner_key == owner_key)
            .order_by(Conversation.updated_at.desc(), Conversation.id)
            .offset(offset)
            .limit(limit)
        )
        return list(self._session.scalars(statement))

    def rename(
        self,
        conversation: Conversation,
        title: str,
    ) -> Conversation:
        normalized_title = title.strip()
        if not normalized_title:
            raise ValueError("Conversation title cannot be empty.")
        conversation.title = normalized_title
        conversation.updated_at = datetime.now(UTC)
        self._session.flush()
        return conversation

    def delete(self, conversation: Conversation) -> None:
        self._session.delete(conversation)
        self._session.flush()


class ConversationMessageRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        *,
        conversation: Conversation,
        role: MessageRole,
        content: str,
        sources: list[dict[str, object]] | None = None,
        answer_mode: str = "documents",
    ) -> ConversationMessage:
        normalized_content = content.strip()
        if not normalized_content:
            raise ValueError("Conversation message cannot be empty.")
        if role is MessageRole.USER and sources:
            raise ValueError("User messages cannot contain answer sources.")
        normalized_answer_mode = answer_mode.strip().lower()
        if normalized_answer_mode not in {"documents", "general"}:
            raise ValueError("Answer mode must be 'documents' or 'general'.")

        message = ConversationMessage(
            conversation_id=conversation.id,
            role=role,
            content=normalized_content,
            sources=sources or [],
            answer_mode=normalized_answer_mode,
            created_at=datetime.now(UTC),
        )
        conversation.updated_at = datetime.now(UTC)
        self._session.add(message)
        self._session.flush()
        return message

    def list_for_conversation(
        self,
        conversation_id: UUID,
        *,
        limit: int = 200,
    ) -> list[ConversationMessage]:
        if limit < 1 or limit > 1000:
            raise ValueError("Message limit must be between 1 and 1000.")
        statement = (
            select(ConversationMessage)
            .where(ConversationMessage.conversation_id == conversation_id)
            .order_by(ConversationMessage.created_at, ConversationMessage.id)
            .limit(limit)
        )
        return list(self._session.scalars(statement))
