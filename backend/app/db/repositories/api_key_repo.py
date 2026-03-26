import uuid

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.api_key import UserApiKey


async def get_by_provider(db: AsyncSession, user_id: uuid.UUID, provider: str) -> UserApiKey | None:
    result = await db.execute(
        select(UserApiKey).where(UserApiKey.user_id == user_id, UserApiKey.provider == provider)
    )
    return result.scalar_one_or_none()


async def list_by_user(db: AsyncSession, user_id: uuid.UUID) -> list[UserApiKey]:
    result = await db.execute(
        select(UserApiKey).where(UserApiKey.user_id == user_id).order_by(UserApiKey.provider)
    )
    return list(result.scalars().all())


async def upsert(
    db: AsyncSession,
    *,
    user_id: uuid.UUID,
    provider: str,
    api_key_encrypted: str,
    base_url: str | None = None,
    model_id: str | None = None,
    display_name: str | None = None,
) -> UserApiKey:
    existing = await get_by_provider(db, user_id, provider)
    if existing:
        existing.api_key_encrypted = api_key_encrypted
        existing.base_url = base_url
        existing.model_id = model_id
        existing.display_name = display_name
        await db.flush()
        await db.refresh(existing)
        return existing
    record = UserApiKey(
        user_id=user_id,
        provider=provider,
        api_key_encrypted=api_key_encrypted,
        base_url=base_url,
        model_id=model_id,
        display_name=display_name,
    )
    db.add(record)
    await db.flush()
    await db.refresh(record)
    return record


async def delete_by_provider(db: AsyncSession, user_id: uuid.UUID, provider: str) -> bool:
    result = await db.execute(
        delete(UserApiKey).where(UserApiKey.user_id == user_id, UserApiKey.provider == provider)
    )
    return result.rowcount > 0
