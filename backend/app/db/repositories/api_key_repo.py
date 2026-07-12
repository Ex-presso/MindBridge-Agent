import uuid
from datetime import datetime, timezone

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.api_key import UserApiKey


async def get_by_provider(db: AsyncSession, user_id: uuid.UUID, provider: str) -> UserApiKey | None:
    result = await db.execute(
        select(UserApiKey).where(UserApiKey.user_id == user_id, UserApiKey.provider == provider)
    )
    return result.scalar_one_or_none()


async def get_owned_by_id(
    db: AsyncSession,
    user_id: uuid.UUID,
    api_key_id: uuid.UUID,
) -> UserApiKey | None:
    result = await db.execute(
        select(UserApiKey).where(
            UserApiKey.id == api_key_id,
            UserApiKey.user_id == user_id,
        )
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
    statement = (
        insert(UserApiKey)
        .values(
            user_id=user_id,
            provider=provider,
            api_key_encrypted=api_key_encrypted,
            base_url=base_url,
            model_id=model_id,
            display_name=display_name,
        )
        .on_conflict_do_update(
            index_elements=[UserApiKey.user_id, UserApiKey.provider],
            set_={
                "api_key_encrypted": api_key_encrypted,
                "base_url": base_url,
                "model_id": model_id,
                "display_name": display_name,
                "updated_at": datetime.now(timezone.utc),
            },
        )
        .returning(UserApiKey)
        .execution_options(populate_existing=True)
    )
    result = await db.execute(statement)
    return result.scalar_one()


async def delete_by_provider(db: AsyncSession, user_id: uuid.UUID, provider: str) -> bool:
    result = await db.execute(
        delete(UserApiKey).where(UserApiKey.user_id == user_id, UserApiKey.provider == provider)
    )
    return result.rowcount > 0
