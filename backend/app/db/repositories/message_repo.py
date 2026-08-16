import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.message import Message


async def create(db: AsyncSession, *, conversation_id: uuid.UUID, role: str, content: str, model_used: str | None = None, provider: str | None = None, tokens_used: int | None = None) -> Message:
    msg = Message(conversation_id=conversation_id, role=role, content=content, model_used=model_used, provider=provider, tokens_used=tokens_used)
    db.add(msg)
    await db.flush()
    await db.refresh(msg)
    return msg


async def list_by_conversation(db: AsyncSession, conv_id: uuid.UUID) -> list[Message]:
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conv_id)
        .order_by(Message.created_at.asc())
    )
    return list(result.scalars().all())


async def list_by_ids_for_conversation(
    db: AsyncSession,
    conv_id: uuid.UUID,
    message_ids: tuple[uuid.UUID, ...],
) -> list[Message]:
    """Reload exact relational evidence inside one conversation boundary."""
    if not message_ids:
        return []
    result = await db.execute(
        select(Message)
        .where(
            Message.conversation_id == conv_id,
            Message.id.in_(message_ids),
        )
        .order_by(Message.created_at.asc())
    )
    return list(result.scalars().all())
