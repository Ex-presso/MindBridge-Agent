"""harden memory ownership

Revision ID: 005
Revises: 004
Create Date: 2026-07-10 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None
CRISIS_DETECTOR_VERSION = 3


def _assert_safe_downgrade(bind: sa.Connection) -> None:
    """Preserve deletion and historical-crisis gates across rollbacks."""
    # Block concurrent tombstone/review/job changes across the preflight and
    # the destructive DDL. Parent tables come before their children.
    bind.execute(
        sa.text(
            "LOCK TABLE users, conversations, messages, user_api_keys, "
            "memory_jobs IN EXCLUSIVE MODE"
        )
    )
    unsafe_state = bind.execute(
        sa.text(
            f"""
            SELECT EXISTS (
                SELECT 1
                FROM users
                WHERE account_deletion_pending IS TRUE
            ) OR EXISTS (
                SELECT 1
                FROM conversations
                WHERE memory_crisis_reviewed IS NOT TRUE
                   OR memory_crisis_review_version <> {CRISIS_DETECTOR_VERSION}
            )
            """
        )
    ).scalar_one()
    if unsafe_state:
        raise RuntimeError(
            "Cannot downgrade memory ownership while an account deletion or "
            "historical crisis review is pending. Complete or explicitly "
            "recover that state first."
        )


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column(
            "account_deletion_pending",
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
    )
    op.add_column(
        "conversations",
        sa.Column(
            "memory_crisis_reviewed",
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
    )
    op.add_column(
        "conversations",
        sa.Column(
            "memory_crisis_review_version",
            sa.Integer(),
            server_default="0",
            nullable=False,
        ),
    )
    # Existing conversations remain (reviewed=false, version=0) and are excluded
    # until their complete user-message history is deterministically reviewed.
    # New rows are reviewed by construction at the current detector version.
    op.alter_column(
        "conversations",
        "memory_crisis_reviewed",
        server_default=sa.true(),
    )
    op.alter_column(
        "conversations",
        "memory_crisis_review_version",
        server_default=str(CRISIS_DETECTOR_VERSION),
    )

    op.create_unique_constraint(
        "uq_conversations_id_user_id",
        "conversations",
        ["id", "user_id"],
    )
    op.create_unique_constraint(
        "uq_messages_id_conversation_id",
        "messages",
        ["id", "conversation_id"],
    )
    op.create_unique_constraint(
        "uq_user_api_keys_id_user_id",
        "user_api_keys",
        ["id", "user_id"],
    )

    op.drop_constraint(
        "memory_jobs_conversation_id_fkey",
        "memory_jobs",
        type_="foreignkey",
    )
    op.drop_constraint(
        "memory_jobs_source_user_message_id_fkey",
        "memory_jobs",
        type_="foreignkey",
    )
    op.drop_constraint(
        "memory_jobs_source_assistant_message_id_fkey",
        "memory_jobs",
        type_="foreignkey",
    )
    op.drop_constraint(
        "memory_jobs_api_key_id_fkey",
        "memory_jobs",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "fk_memory_jobs_conversation_owner",
        "memory_jobs",
        "conversations",
        ["conversation_id", "user_id"],
        ["id", "user_id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_memory_jobs_user_message_conversation",
        "memory_jobs",
        "messages",
        ["source_user_message_id", "conversation_id"],
        ["id", "conversation_id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_memory_jobs_assistant_message_conversation",
        "memory_jobs",
        "messages",
        ["source_assistant_message_id", "conversation_id"],
        ["id", "conversation_id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_memory_jobs_api_key_owner",
        "memory_jobs",
        "user_api_keys",
        ["api_key_id", "user_id"],
        ["id", "user_id"],
        ondelete="CASCADE",
    )
    op.create_index(
        "ix_memory_jobs_user_message_conversation",
        "memory_jobs",
        ["source_user_message_id", "conversation_id"],
        unique=False,
    )
    op.create_index(
        "ix_memory_jobs_assistant_message_conversation",
        "memory_jobs",
        ["source_assistant_message_id", "conversation_id"],
        unique=False,
    )
    op.create_index(
        "ix_memory_jobs_api_key_user",
        "memory_jobs",
        ["api_key_id", "user_id"],
        unique=False,
    )


def downgrade() -> None:
    _assert_safe_downgrade(op.get_bind())

    op.drop_index(
        "ix_memory_jobs_api_key_user",
        table_name="memory_jobs",
    )
    op.drop_index(
        "ix_memory_jobs_assistant_message_conversation",
        table_name="memory_jobs",
    )
    op.drop_index(
        "ix_memory_jobs_user_message_conversation",
        table_name="memory_jobs",
    )
    op.drop_constraint(
        "fk_memory_jobs_api_key_owner",
        "memory_jobs",
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_memory_jobs_assistant_message_conversation",
        "memory_jobs",
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_memory_jobs_user_message_conversation",
        "memory_jobs",
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_memory_jobs_conversation_owner",
        "memory_jobs",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "memory_jobs_api_key_id_fkey",
        "memory_jobs",
        "user_api_keys",
        ["api_key_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "memory_jobs_source_assistant_message_id_fkey",
        "memory_jobs",
        "messages",
        ["source_assistant_message_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "memory_jobs_source_user_message_id_fkey",
        "memory_jobs",
        "messages",
        ["source_user_message_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "memory_jobs_conversation_id_fkey",
        "memory_jobs",
        "conversations",
        ["conversation_id"],
        ["id"],
        ondelete="CASCADE",
    )

    op.drop_constraint(
        "uq_user_api_keys_id_user_id",
        "user_api_keys",
        type_="unique",
    )
    op.drop_constraint(
        "uq_messages_id_conversation_id",
        "messages",
        type_="unique",
    )
    op.drop_constraint(
        "uq_conversations_id_user_id",
        "conversations",
        type_="unique",
    )
    op.drop_column("conversations", "memory_crisis_review_version")
    op.drop_column("conversations", "memory_crisis_reviewed")
    op.drop_column("users", "account_deletion_pending")
