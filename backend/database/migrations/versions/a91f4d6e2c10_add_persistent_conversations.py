"""Add persistent conversations and messages.

Revision ID: a91f4d6e2c10
Revises: cf2e91a47b30
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a91f4d6e2c10"
down_revision: str | Sequence[str] | None = "cf2e91a47b30"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

message_role = sa.Enum(
    "user",
    "assistant",
    name="message_role",
    native_enum=False,
    create_constraint=True,
)


def upgrade() -> None:
    op.create_table(
        "conversations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "title",
            sa.String(length=255),
            server_default="New conversation",
            nullable=False,
        ),
        sa.Column(
            "owner_key",
            sa.String(length=128),
            server_default="local",
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
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_conversations_owner_key"),
        "conversations",
        ["owner_key"],
        unique=False,
    )
    op.create_index(
        op.f("ix_conversations_updated_at"),
        "conversations",
        ["updated_at"],
        unique=False,
    )
    op.create_table(
        "conversation_messages",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("role", message_role, nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "answer_mode",
            sa.String(length=32),
            server_default="documents",
            nullable=False,
        ),
        sa.Column("sources", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_conversation_messages_conversation_id"),
        "conversation_messages",
        ["conversation_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_conversation_messages_created_at"),
        "conversation_messages",
        ["created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_conversation_messages_created_at"),
        table_name="conversation_messages",
    )
    op.drop_index(
        op.f("ix_conversation_messages_conversation_id"),
        table_name="conversation_messages",
    )
    op.drop_table("conversation_messages")
    op.drop_index(
        op.f("ix_conversations_updated_at"),
        table_name="conversations",
    )
    op.drop_index(
        op.f("ix_conversations_owner_key"),
        table_name="conversations",
    )
    op.drop_table("conversations")
