import uuid
from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models.user import User


@dataclass(frozen=True, slots=True)
class MemoryAccessSnapshot:
    """Fresh, non-locking memory gate read for one graph invocation."""

    enabled: bool
    consent_version: int
    data_epoch: int
    account_deletion_pending: bool = False


async def get_by_email(db: AsyncSession, email: str) -> User | None:
    result = await db.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()


async def get_by_id(db: AsyncSession, user_id: uuid.UUID) -> User | None:
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


async def lock_for_account_deletion(
    db: AsyncSession,
    user_id: uuid.UUID,
) -> bool:
    """Take a strong User lock that blocks new child-FK KEY SHARE locks."""
    result = await db.execute(
        select(User.id).where(User.id == user_id).with_for_update()
    )
    return result.scalar_one_or_none() is not None


async def get_memory_enabled(db: AsyncSession, user_id: uuid.UUID) -> bool:
    """Read current consent as a scalar, bypassing any stale ORM identity state."""
    result = await db.execute(
        select(User.memory_enabled).where(User.id == user_id)
    )
    return bool(result.scalar_one_or_none())


async def get_memory_access_snapshot(
    db: AsyncSession,
    user_id: uuid.UUID,
) -> MemoryAccessSnapshot | None:
    """Read consent and both writer epochs in one READ COMMITTED statement."""
    result = await db.execute(
        select(
            User.memory_enabled,
            User.memory_consent_version,
            User.memory_data_epoch,
            User.account_deletion_pending,
        ).where(User.id == user_id)
    )
    row = result.one_or_none()
    if row is None:
        return None
    return MemoryAccessSnapshot(
        enabled=bool(row[0]),
        consent_version=int(row[1]),
        data_epoch=int(row[2]),
        account_deletion_pending=bool(row[3]),
    )


async def get_memory_access_for_chat(
    db: AsyncSession,
    user_id: uuid.UUID,
) -> MemoryAccessSnapshot | None:
    """Lock User in KEY SHARE before any Conversation lock or FK insert."""
    result = await db.execute(
        select(
            User.memory_enabled,
            User.memory_consent_version,
            User.memory_data_epoch,
            User.account_deletion_pending,
        )
        .where(User.id == user_id)
        .with_for_update(read=True, key_share=True)
    )
    row = result.one_or_none()
    if row is None:
        return None
    return MemoryAccessSnapshot(
        enabled=bool(row[0]),
        consent_version=int(row[1]),
        data_epoch=int(row[2]),
        account_deletion_pending=bool(row[3]),
    )


async def get_memory_access_for_update(
    db: AsyncSession,
    user_id: uuid.UUID,
) -> MemoryAccessSnapshot | None:
    """Strong-lock User before a runtime credential mutation."""
    result = await db.execute(
        select(
            User.memory_enabled,
            User.memory_consent_version,
            User.memory_data_epoch,
            User.account_deletion_pending,
        )
        .where(User.id == user_id)
        .with_for_update()
    )
    row = result.one_or_none()
    if row is None:
        return None
    return MemoryAccessSnapshot(
        enabled=bool(row[0]),
        consent_version=int(row[1]),
        data_epoch=int(row[2]),
        account_deletion_pending=bool(row[3]),
    )


async def create(db: AsyncSession, *, email: str, hashed_password: str, display_name: str | None = None) -> User:
    user = User(email=email, hashed_password=hashed_password, display_name=display_name)
    db.add(user)
    await db.flush()
    await db.refresh(user)
    return user


async def email_exists(db: AsyncSession, email: str) -> bool:
    result = await db.execute(select(User.id).where(User.email == email))
    return result.scalar_one_or_none() is not None


async def delete_by_id(db: AsyncSession, user_id: uuid.UUID) -> bool:
    result = await db.execute(delete(User).where(User.id == user_id))
    return result.rowcount > 0
