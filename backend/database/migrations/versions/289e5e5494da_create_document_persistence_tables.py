"""Create document persistence tables.

Revision ID: 289e5e5494da
Revises:
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "289e5e5494da"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column(
            "id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "filename",
            sa.String(length=255),
            nullable=False,
        ),
        sa.Column(
            "storage_path",
            sa.String(length=1024),
            nullable=False,
        ),
        sa.Column(
            "content_type",
            sa.String(length=255),
            nullable=True,
        ),
        sa.Column(
            "size_bytes",
            sa.BigInteger(),
            nullable=False,
        ),
        sa.Column(
            "checksum_sha256",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.Enum(
                "pending",
                "indexing",
                "ready",
                "failed",
                name="document_status",
                native_enum=False,
                create_constraint=True,
            ),
            server_default="pending",
            nullable=False,
        ),
        sa.Column(
            "chunk_count",
            sa.Integer(),
            server_default="0",
            nullable=False,
        ),
        sa.Column(
            "error_message",
            sa.Text(),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "chunk_count >= 0",
            name=op.f("ck_documents_chunk_count_nonnegative"),
        ),
        sa.CheckConstraint(
            "size_bytes >= 0",
            name=op.f("ck_documents_size_bytes_nonnegative"),
        ),
        sa.PrimaryKeyConstraint(
            "id",
            name=op.f("pk_documents"),
        ),
        sa.UniqueConstraint(
            "storage_path",
            name=op.f("uq_documents_storage_path"),
        ),
    )

    op.create_index(
        op.f("ix_documents_checksum_sha256"),
        "documents",
        ["checksum_sha256"],
        unique=False,
    )

    op.create_index(
        op.f("ix_documents_filename"),
        "documents",
        ["filename"],
        unique=False,
    )

    op.create_table(
        "ingestion_jobs",
        sa.Column(
            "id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "document_id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "operation",
            sa.Enum(
                "index",
                "reindex",
                "delete",
                name="ingestion_operation",
                native_enum=False,
                create_constraint=True,
            ),
            server_default="index",
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.Enum(
                "queued",
                "running",
                "succeeded",
                "failed",
                name="ingestion_job_status",
                native_enum=False,
                create_constraint=True,
            ),
            server_default="queued",
            nullable=False,
        ),
        sa.Column(
            "attempt_count",
            sa.Integer(),
            server_default="0",
            nullable=False,
        ),
        sa.Column(
            "processed_chunks",
            sa.Integer(),
            server_default="0",
            nullable=False,
        ),
        sa.Column(
            "total_chunks",
            sa.Integer(),
            nullable=True,
        ),
        sa.Column(
            "error_message",
            sa.Text(),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "completed_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "attempt_count >= 0",
            name=op.f("ck_ingestion_jobs_attempt_count_nonnegative"),
        ),
        sa.CheckConstraint(
            "processed_chunks >= 0",
            name=op.f("ck_ingestion_jobs_processed_chunks_nonnegative"),
        ),
        sa.CheckConstraint(
            "total_chunks IS NULL OR processed_chunks <= total_chunks",
            name=op.f("ck_ingestion_jobs_processed_chunks_within_total"),
        ),
        sa.CheckConstraint(
            "total_chunks IS NULL OR total_chunks >= 0",
            name=op.f("ck_ingestion_jobs_total_chunks_nonnegative"),
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_ingestion_jobs_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "id",
            name=op.f("pk_ingestion_jobs"),
        ),
    )

    op.create_index(
        op.f("ix_ingestion_jobs_document_id"),
        "ingestion_jobs",
        ["document_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_ingestion_jobs_document_id"),
        table_name="ingestion_jobs",
    )
    op.drop_table("ingestion_jobs")

    op.drop_index(
        op.f("ix_documents_filename"),
        table_name="documents",
    )
    op.drop_index(
        op.f("ix_documents_checksum_sha256"),
        table_name="documents",
    )
    op.drop_table("documents")
