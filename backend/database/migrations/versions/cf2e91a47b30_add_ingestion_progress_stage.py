"""Add durable ingestion progress stage.

Revision ID: cf2e91a47b30
Revises: b6e4f2a91c73
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "cf2e91a47b30"
down_revision: str | Sequence[str] | None = "b6e4f2a91c73"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

ingestion_stage = sa.Enum(
    "queued",
    "parsing",
    "embedding",
    "indexing",
    "succeeded",
    "failed",
    name="ingestion_stage",
    native_enum=False,
    create_constraint=True,
)


def upgrade() -> None:
    op.add_column(
        "ingestion_jobs",
        sa.Column(
            "stage",
            ingestion_stage,
            nullable=False,
            server_default="queued",
        ),
    )

    op.execute(
        """
        UPDATE ingestion_jobs
        SET stage = CASE status
            WHEN 'succeeded' THEN 'succeeded'
            WHEN 'failed' THEN 'failed'
            WHEN 'running' THEN 'parsing'
            ELSE 'queued'
        END
        """,
    )


def downgrade() -> None:
    op.drop_column("ingestion_jobs", "stage")
