from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    Uuid,
    func,
)
from sqlalchemy import (
    Enum as SqlEnum,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database.base import Base


class DocumentStatus(StrEnum):
    PENDING = "pending"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"


class IngestionOperation(StrEnum):
    INDEX = "index"
    REINDEX = "reindex"
    DELETE = "delete"


class IngestionJobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


document_status_type = SqlEnum(
    DocumentStatus,
    name="document_status",
    native_enum=False,
    create_constraint=True,
    validate_strings=True,
    values_callable=lambda enum_type: [member.value for member in enum_type],
)

ingestion_operation_type = SqlEnum(
    IngestionOperation,
    name="ingestion_operation",
    native_enum=False,
    create_constraint=True,
    validate_strings=True,
    values_callable=lambda enum_type: [member.value for member in enum_type],
)

ingestion_job_status_type = SqlEnum(
    IngestionJobStatus,
    name="ingestion_job_status",
    native_enum=False,
    create_constraint=True,
    validate_strings=True,
    values_callable=lambda enum_type: [member.value for member in enum_type],
)


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint(
            "size_bytes >= 0",
            name="size_bytes_nonnegative",
        ),
        CheckConstraint(
            "chunk_count >= 0",
            name="chunk_count_nonnegative",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )

    storage_path: Mapped[str] = mapped_column(
        String(1024),
        nullable=False,
        unique=True,
    )

    content_type: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    size_bytes: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    checksum_sha256: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )

    status: Mapped[DocumentStatus] = mapped_column(
        document_status_type,
        nullable=False,
        default=DocumentStatus.PENDING,
        server_default=DocumentStatus.PENDING.value,
    )

    chunk_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    ingestion_jobs: Mapped[list[IngestionJob]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"
    __table_args__ = (
        CheckConstraint(
            "attempt_count >= 0",
            name="attempt_count_nonnegative",
        ),
        CheckConstraint(
            "processed_chunks >= 0",
            name="processed_chunks_nonnegative",
        ),
        CheckConstraint(
            "total_chunks IS NULL OR total_chunks >= 0",
            name="total_chunks_nonnegative",
        ),
        CheckConstraint(
            "total_chunks IS NULL OR processed_chunks <= total_chunks",
            name="processed_chunks_within_total",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    document_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            "documents.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    operation: Mapped[IngestionOperation] = mapped_column(
        ingestion_operation_type,
        nullable=False,
        default=IngestionOperation.INDEX,
        server_default=IngestionOperation.INDEX.value,
    )

    status: Mapped[IngestionJobStatus] = mapped_column(
        ingestion_job_status_type,
        nullable=False,
        default=IngestionJobStatus.QUEUED,
        server_default=IngestionJobStatus.QUEUED.value,
    )

    attempt_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    processed_chunks: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    total_chunks: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    document: Mapped[Document] = relationship(
        back_populates="ingestion_jobs",
    )

    payload: Mapped[IngestionPayload | None] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        passive_deletes=True,
        uselist=False,
    )


class IngestionPayload(Base):
    __tablename__ = "ingestion_payloads"
    __table_args__ = (
        CheckConstraint(
            "size_bytes >= 0",
            name="ingestion_payload_size_bytes_nonnegative",
        ),
    )

    job_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            "ingestion_jobs.id",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )

    staged_path: Mapped[str] = mapped_column(
        String(1024),
        nullable=False,
        unique=True,
    )

    content_type: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    size_bytes: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    checksum_sha256: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    job: Mapped[IngestionJob] = relationship(
        back_populates="payload",
    )
