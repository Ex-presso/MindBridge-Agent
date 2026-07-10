import uuid
from datetime import datetime, timezone

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.conversation import Conversation


async def create(db: AsyncSession, *, user_id: uuid.UUID, title: str = "New conversation", model: str | None = None, provider: str | None = None) -> Conversation:
    conv = Conversation(user_id=user_id, title=title, model=model, provider=provider)
    db.add(conv)
    await db.flush()
    await db.refresh(conv)
    return conv


async def get_by_id(db: AsyncSession, conv_id: uuid.UUID) -> Conversation | None:
    result = await db.execute(select(Conversation).where(Conversation.id == conv_id))
    return result.scalar_one_or_none()


async def get_owned_for_update(
    db: AsyncSession,
    conv_id: uuid.UUID,
    user_id: uuid.UUID,
) -> Conversation | None:
    """Lock an owned conversation until the current transaction finishes.

    Chat runs and deletion both use this lock so a completed deletion cannot be
    followed by a still-running graph recreating the conversation checkpoint.
    """
    result = await db.execute(
        select(Conversation)
        .where(Conversation.id == conv_id, Conversation.user_id == user_id)
        .with_for_update()
    )
    return result.scalar_one_or_none()


async def list_by_user(db: AsyncSession, user_id: uuid.UUID, *, limit: int = 50) -> list[Conversation]:
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.updated_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def update_title(db: AsyncSession, conv_id: uuid.UUID, title: str) -> None:
    await db.execute(
        update(Conversation)
        .where(Conversation.id == conv_id)
        .values(title=title, updated_at=datetime.now(timezone.utc))
    )


async def touch(db: AsyncSession, conv_id: uuid.UUID, model: str | None = None, provider: str | None = None) -> None:
    values: dict = {"updated_at": datetime.now(timezone.utc)}
    if model:
        values["model"] = model
    if provider:
        values["provider"] = provider
    await db.execute(update(Conversation).where(Conversation.id == conv_id).values(**values))


async def delete_by_id(db: AsyncSession, conv_id: uuid.UUID) -> bool:
    result = await db.execute(delete(Conversation).where(Conversation.id == conv_id))
    return result.rowcount > 0
