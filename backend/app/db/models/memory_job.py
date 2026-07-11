import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.engine import Base

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


class MemoryJob(Base):
    __tablename__ = "memory_jobs"
    __table_args__ = (
        CheckConstraint(
            f"operation IN {MEMORY_JOB_OPERATIONS}",
            name="ck_memory_jobs_operation",
        ),
        CheckConstraint(
            f"status IN {MEMORY_JOB_STATUSES}",
            name="ck_memory_jobs_status",
        ),
        UniqueConstraint("dedupe_key", name="uq_memory_jobs_dedupe_key"),
        ForeignKeyConstraint(
            ["conversation_id", "user_id"],
            ["conversations.id", "conversations.user_id"],
            name="fk_memory_jobs_conversation_owner",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["source_user_message_id", "conversation_id"],
            ["messages.id", "messages.conversation_id"],
            name="fk_memory_jobs_user_message_conversation",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["source_assistant_message_id", "conversation_id"],
            ["messages.id", "messages.conversation_id"],
            name="fk_memory_jobs_assistant_message_conversation",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["api_key_id", "user_id"],
            ["user_api_keys.id", "user_api_keys.user_id"],
            name="fk_memory_jobs_api_key_owner",
            ondelete="CASCADE",
        ),
        Index("ix_memory_jobs_status_available_at", "status", "available_at"),
        Index(
            "ix_memory_jobs_conversation_revision",
            "conversation_id",
            "target_revision",
        ),
        Index("ix_memory_jobs_user_status", "user_id", "status"),
        Index(
            "ix_memory_jobs_user_message_conversation",
            "source_user_message_id",
            "conversation_id",
        ),
        Index(
            "ix_memory_jobs_assistant_message_conversation",
            "source_assistant_message_id",
            "conversation_id",
        ),
        Index(
            "ix_memory_jobs_api_key_user",
            "api_key_id",
            "user_id",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    operation: Mapped[str] = mapped_column(String(32), nullable=False)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        nullable=False,
    )
    target_revision: Mapped[int] = mapped_column(BigInteger, nullable=False)
    consent_version: Mapped[int] = mapped_column(BigInteger, nullable=False)
    data_epoch: Mapped[int] = mapped_column(BigInteger, nullable=False)
    source_user_message_id: Mapped[uuid.UUID | None] = mapped_column(
        nullable=True,
    )
    source_assistant_message_id: Mapped[uuid.UUID | None] = mapped_column(
        nullable=True,
    )
    api_key_id: Mapped[uuid.UUID | None] = mapped_column(
        nullable=True,
    )
    provider: Mapped[str | None] = mapped_column(String(50), nullable=True)
    model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    base_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    status: Mapped[str] = mapped_column(
        String(32),
        default="pending",
        server_default="pending",
        nullable=False,
    )
    priority: Mapped[int] = mapped_column(
        Integer,
        default=100,
        server_default="100",
        nullable=False,
    )
    available_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
    )
    lease_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    attempts: Mapped[int] = mapped_column(
        Integer,
        default=0,
        server_default="0",
        nullable=False,
    )
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dedupe_key: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
