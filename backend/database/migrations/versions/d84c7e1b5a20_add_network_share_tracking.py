"""Add read-only network-share source tracking.

Revision ID: d84c7e1b5a20
Revises: a91f4d6e2c10
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d84c7e1b5a20"
down_revision: str | Sequence[str] | None = "a91f4d6e2c10"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "network_share_sources",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("root_path", sa.String(length=1024), nullable=False),
        sa.Column(
            "enabled",
            sa.Boolean(),
            server_default=sa.true(),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.Enum(
                "never_synced",
                "syncing",
                "ready",
                "unavailable",
                "failed",
                name="network_share_status",
                native_enum=False,
                create_constraint=True,
            ),
            server_default="never_synced",
            nullable=False,
        ),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("last_scan_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_scan_completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
        sa.UniqueConstraint("root_path"),
    )

    op.create_table(
        "network_share_files",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("source_id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=True),
        sa.Column("relative_path", sa.String(length=1024), nullable=False),
        sa.Column("checksum_sha256", sa.String(length=64), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("modified_time_ns", sa.BigInteger(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "discovered",
                "queued",
                "synced",
                "missing",
                "failed",
                name="network_share_file_status",
                native_enum=False,
                create_constraint=True,
            ),
            server_default="discovered",
            nullable=False,
        ),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column(
            "last_seen_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "modified_time_ns >= 0",
            name="network_share_file_modified_time_nonnegative",
        ),
        sa.CheckConstraint(
            "size_bytes >= 0",
            name="network_share_file_size_bytes_nonnegative",
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["source_id"],
            ["network_share_sources.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "source_id",
            "relative_path",
            name="uq_network_share_file_source_path",
        ),
    )
    op.create_index(
        op.f("ix_network_share_files_checksum_sha256"),
        "network_share_files",
        ["checksum_sha256"],
        unique=False,
    )
    op.create_index(
        op.f("ix_network_share_files_document_id"),
        "network_share_files",
        ["document_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_network_share_files_last_seen_at"),
        "network_share_files",
        ["last_seen_at"],
        unique=False,
    )
    op.create_index(
        op.f("ix_network_share_files_source_id"),
        "network_share_files",
        ["source_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_network_share_files_source_id"),
        table_name="network_share_files",
    )
    op.drop_index(
        op.f("ix_network_share_files_last_seen_at"),
        table_name="network_share_files",
    )
    op.drop_index(
        op.f("ix_network_share_files_document_id"),
        table_name="network_share_files",
    )
    op.drop_index(
        op.f("ix_network_share_files_checksum_sha256"),
        table_name="network_share_files",
    )
    op.drop_table("network_share_files")
    op.drop_table("network_share_sources")
