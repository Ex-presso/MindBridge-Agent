"""replace memory jobs with an outbox

Background memory work now travels through RabbitMQ. The database only needs
an outbox: rows waiting to be published. Leases, attempts and statuses were
the job queue's bookkeeping and move to the broker.

Revision ID: 006
Revises: 005
Create Date: 2026-11-05 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "outbox_events",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    # Carry unfinished jobs over so a queued extraction or privacy deletion
    # is published instead of being dropped with the old table.
    op.execute(
        """
        INSERT INTO outbox_events (id, event_type, payload, created_at)
        SELECT
            id,
            CASE operation
                WHEN 'delete_episode' THEN 'memory.delete_conversation'
                ELSE 'memory.extract'
            END,
            jsonb_strip_nulls(jsonb_build_object(
                'operation',
                CASE operation
                    WHEN 'delete_episode' THEN 'delete_conversation'
                    ELSE 'extract'
                END,
                'user_id', user_id::text,
                'conversation_id', conversation_id::text,
                'consent_version', consent_version,
                'data_epoch', data_epoch,
                'message_id', source_user_message_id::text,
                'api_key_id', api_key_id::text,
                'provider', provider,
                'model', model
            )),
            created_at
        FROM memory_jobs
        WHERE status IN ('pending', 'processing', 'retry')
        """
    )
    op.drop_table("memory_jobs")
    op.drop_column("conversations", "memory_revision")


def downgrade() -> None:
    raise RuntimeError(
        "Migration 006 cannot be reversed: the job history in memory_jobs "
        "was dropped. Restore a backup taken before the upgrade instead."
    )
