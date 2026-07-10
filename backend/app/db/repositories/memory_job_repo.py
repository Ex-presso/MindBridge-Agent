import uuid
from datetime import datetime, timezone

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.memory_job import MemoryJob

UNFINISHED_STATUSES = frozenset({"pending", "processing", "retry"})


async def enqueue_delete_episode(
    db: AsyncSession,
    *,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID,
    target_revision: int,
    consent_version: int,
    data_epoch: int,
) -> MemoryJob:
    """Durably request physical removal after a sticky crisis transition."""
    job = MemoryJob(
        operation="delete_episode",
        user_id=user_id,
        conversation_id=conversation_id,
        target_revision=target_revision,
        consent_version=consent_version,
        data_epoch=data_epoch,
        status="pending",
        priority=0,
        dedupe_key=f"delete_episode:crisis:{user_id}:{conversation_id}",
    )
    db.add(job)
    await db.flush()
    return job


async def cancel_unfinished_for_user(
    db: AsyncSession,
    user_id: uuid.UUID,
) -> int:
    """Cancel queued or leased jobs before clearing a user's memory data."""
    result = await db.execute(
        update(MemoryJob)
        .where(
            MemoryJob.user_id == user_id,
            MemoryJob.status.in_(UNFINISHED_STATUSES),
        )
        .values(
            status="canceled",
            lease_until=None,
            updated_at=datetime.now(timezone.utc),
        )
    )
    return int(result.rowcount or 0)
