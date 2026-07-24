"""Prevent duplicate active ingestion jobs.

Revision ID: b6e4f2a91c73
Revises: 72251d9f630f
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b6e4f2a91c73"
down_revision: str | Sequence[str] | None = "72251d9f630f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "uq_ingestion_jobs_active_document",
        "ingestion_jobs",
        ["document_id"],
        unique=True,
        postgresql_where=sa.text(
            "status IN ('queued', 'running')",
        ),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_ingestion_jobs_active_document",
        table_name="ingestion_jobs",
    )
