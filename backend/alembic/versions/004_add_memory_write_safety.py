"""add memory write safety

Revision ID: 004
Revises: 003
Create Date: 2026-07-10 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

API_KEY_UNIQUE_CONSTRAINT = "uq_user_api_keys_user_id_provider"
MEMORY_JOB_OPERATIONS = ("extract_episode", "delete_episode")
MEMORY_JOB_STATUSES = (
    "pending",
    "processing",
    "retry",
    "succeeded",
    "superseded",
    "canceled",
    "filtered",
    "blocked",
)


def _has_user_provider_uniqueness(inspector: sa.Inspector) -> bool:
    expected_columns = {"user_id", "provider"}
    for constraint in inspector.get_unique_constraints("user_api_keys"):
        if set(constraint.get("column_names") or ()) == expected_columns:
            return True
    return any(
        index.get("unique")
        and set(index.get("column_names") or ()) == expected_columns
        and not (index.get("dialect_options") or {}).get("postgresql_where")
        for index in inspector.get_indexes("user_api_keys")
    )


def _ensure_api_key_uniqueness(bind: sa.Connection) -> None:
    """Add the upsert invariant without choosing between duplicate secrets."""
    inspector = sa.inspect(bind)
    if not inspector.has_table("user_api_keys"):
        raise RuntimeError(
            "Cannot add memory write safety: user_api_keys table is missing. "
            "Restore it or repair the Alembic revision before retrying."
        )
    if _has_user_provider_uniqueness(inspector):
        return

    duplicate = bind.execute(
        sa.text(
            """
            SELECT user_id, provider, COUNT(*) AS duplicate_count
            FROM user_api_keys
            GROUP BY user_id, provider
            HAVING COUNT(*) > 1
            LIMIT 1
            """
        )
    ).mappings().first()
    if duplicate is not None:
        raise RuntimeError(
            "Cannot add unique user_api_keys(user_id, provider) constraint: "
            f"found {duplicate['duplicate_count']} records for user "
            f"{duplicate['user_id']} and provider {duplicate['provider']!r}. "
            "Resolve the duplicate API keys manually; no key was deleted."
        )

    op.create_unique_constraint(
        API_KEY_UNIQUE_CONSTRAINT,
        "user_api_keys",
        ["user_id", "provider"],
    )


def _assert_safe_downgrade(bind: sa.Connection) -> None:
    """Refuse to erase live invalidation, crisis, revision, or outbox state."""
    # Prevent a clear/job enqueue from committing after this preflight and
    # before the destructive DDL. The order mirrors runtime parent→child locks.
    bind.execute(
        sa.text(
            "LOCK TABLE users, conversations, user_api_keys, memory_jobs "
            "IN EXCLUSIVE MODE"
        )
    )
    unsafe_state = bind.execute(
        sa.text(
            """
            SELECT EXISTS (
                SELECT 1
                FROM users
                WHERE memory_consent_version <> 0 OR memory_data_epoch <> 0
            ) OR EXISTS (
                SELECT 1
                FROM conversations
                WHERE memory_revision <> 0 OR memory_crisis_seen IS TRUE
            ) OR EXISTS (
                SELECT 1 FROM memory_jobs
            )
            """
        )
    ).scalar_one()
    if unsafe_state:
        raise RuntimeError(
            "Cannot downgrade memory write safety while non-default memory "
            "versions, crisis tombstones, revisions, or jobs exist. Clear or "
            "migrate that state explicitly before retrying."
        )


def upgrade() -> None:
    bind = op.get_bind()
    _ensure_api_key_uniqueness(bind)

    op.add_column(
        "users",
        sa.Column(
            "memory_consent_version",
            sa.BigInteger(),
            server_default="0",
            nullable=False,
        ),
    )
    op.add_column(
        "users",
        sa.Column(
            "memory_data_epoch",
            sa.BigInteger(),
            server_default="0",
            nullable=False,
        ),
    )
    op.add_column(
        "conversations",
        sa.Column(
            "memory_revision",
            sa.BigInteger(),
            server_default="0",
            nullable=False,
        ),
    )
    op.add_column(
        "conversations",
        sa.Column(
            "memory_crisis_seen",
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
    )

    op.create_table(
        "memory_jobs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("operation", sa.String(32), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("target_revision", sa.BigInteger(), nullable=False),
        sa.Column("consent_version", sa.BigInteger(), nullable=False),
        sa.Column("data_epoch", sa.BigInteger(), nullable=False),
        sa.Column("source_user_message_id", sa.Uuid(), nullable=True),
        sa.Column("source_assistant_message_id", sa.Uuid(), nullable=True),
        sa.Column("api_key_id", sa.Uuid(), nullable=True),
        sa.Column("provider", sa.String(50), nullable=True),
        sa.Column("model", sa.String(100), nullable=True),
        sa.Column("base_url", sa.String(500), nullable=True),
        sa.Column(
            "status",
            sa.String(32),
            server_default="pending",
            nullable=False,
        ),
        sa.Column("priority", sa.Integer(), server_default="100", nullable=False),
        sa.Column(
            "available_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("lease_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column("error_code", sa.String(64), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("dedupe_key", sa.String(255), nullable=False),
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
            f"operation IN {MEMORY_JOB_OPERATIONS}",
            name="ck_memory_jobs_operation",
        ),
        sa.CheckConstraint(
            f"status IN {MEMORY_JOB_STATUSES}",
            name="ck_memory_jobs_status",
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["source_user_message_id"],
            ["messages.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["source_assistant_message_id"],
            ["messages.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["api_key_id"],
            ["user_api_keys.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("dedupe_key", name="uq_memory_jobs_dedupe_key"),
    )
    op.create_index(
        "ix_memory_jobs_status_available_at",
        "memory_jobs",
        ["status", "available_at"],
        unique=False,
    )
    op.create_index(
        "ix_memory_jobs_conversation_revision",
        "memory_jobs",
        ["conversation_id", "target_revision"],
        unique=False,
    )
    op.create_index(
        "ix_memory_jobs_user_status",
        "memory_jobs",
        ["user_id", "status"],
        unique=False,
    )


def downgrade() -> None:
    bind = op.get_bind()
    _assert_safe_downgrade(bind)

    op.drop_index("ix_memory_jobs_user_status", table_name="memory_jobs")
    op.drop_index(
        "ix_memory_jobs_conversation_revision",
        table_name="memory_jobs",
    )
    op.drop_index("ix_memory_jobs_status_available_at", table_name="memory_jobs")
    op.drop_table("memory_jobs")
    op.drop_column("conversations", "memory_crisis_seen")
    op.drop_column("conversations", "memory_revision")
    op.drop_column("users", "memory_data_epoch")
    op.drop_column("users", "memory_consent_version")

    # Intentionally retain user/provider uniqueness. Revision 004 may have
    # adopted an equivalent pre-existing constraint and cannot prove ownership
    # during downgrade; deleting it could weaken a legacy schema.
