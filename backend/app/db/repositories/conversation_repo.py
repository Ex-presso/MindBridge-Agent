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


async def list_owned_for_update(
    db: AsyncSession,
    user_id: uuid.UUID,
) -> list[Conversation]:
    """Lock all user conversations in stable order for account cleanup."""
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.id)
        .with_for_update()
    )
    return list(result.scalars().all())


async def filter_memory_eligible_ids(
    db: AsyncSession,
    user_id: uuid.UUID,
    conversation_ids: tuple[uuid.UUID, ...],
) -> set[str]:
    """Return owned conversations that have never crossed the crisis gate."""
    if not conversation_ids:
        return set()
    result = await db.execute(
        select(Conversation.id).where(
            Conversation.user_id == user_id,
            Conversation.id.in_(conversation_ids),
            Conversation.memory_crisis_seen.is_(False),
        )
    )
    return {str(conversation_id) for conversation_id in result.scalars().all()}


async def mark_memory_crisis_seen(
    db: AsyncSession,
    conv_id: uuid.UUID,
    user_id: uuid.UUID,
) -> bool:
    """Set the sticky crisis tombstone and report whether it transitioned."""
    result = await db.execute(
        update(Conversation)
        .where(
            Conversation.id == conv_id,
            Conversation.user_id == user_id,
            Conversation.memory_crisis_seen.is_(False),
        )
        .values(memory_crisis_seen=True)
        .returning(Conversation.id)
    )
    return result.scalar_one_or_none() is not None


async def increment_memory_revision(
    db: AsyncSession,
    conv_id: uuid.UUID,
) -> int:
    """Advance the completed-assistant-turn revision in the current transaction."""
    result = await db.execute(
        update(Conversation)
        .where(Conversation.id == conv_id)
        .values(memory_revision=Conversation.memory_revision + 1)
        .returning(Conversation.memory_revision)
    )
    revision = result.scalar_one_or_none()
    if revision is None:
        raise LookupError("Conversation no longer exists.")
    return int(revision)


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
