import uuid
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.outbox import OutboxEvent


async def add(db: AsyncSession, event_type: str, payload: dict[str, Any]) -> OutboxEvent:
    """Record an event in the caller's transaction."""
    event = OutboxEvent(event_type=event_type, payload=payload)
    db.add(event)
    await db.flush()
    return event


async def lock_oldest(db: AsyncSession, limit: int) -> list[OutboxEvent]:
    """Lock the oldest waiting events for this transaction.

    SKIP LOCKED lets a second relay instance take different rows instead of
    waiting on these or publishing them again.
    """
    result = await db.execute(
        select(OutboxEvent)
        .order_by(OutboxEvent.created_at.asc())
        .limit(limit)
        .with_for_update(skip_locked=True)
    )
    return list(result.scalars().all())


async def delete_many(db: AsyncSession, event_ids: list[uuid.UUID]) -> None:
    if event_ids:
        await db.execute(delete(OutboxEvent).where(OutboxEvent.id.in_(event_ids)))
