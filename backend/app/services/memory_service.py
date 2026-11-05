"""Consent, inspection and deletion of durable user memory."""

from dataclasses import dataclass
from typing import Any
import uuid

from sqlalchemy import case, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.memory import MemoryClient, MemoryRecord, get_memory_client
from app.db.models.user import User
from app.db.repositories import user_repo


class MemoryServiceError(RuntimeError):
    """Base error for memory service operations."""


class MemoryStoreError(MemoryServiceError):
    """The memory store could not be reached or failed an operation."""


class MemoryUserNotFoundError(MemoryServiceError):
    """The target user no longer exists."""


@dataclass(frozen=True, slots=True)
class MemoryGateState:
    """Database-authoritative consent and invalidation versions."""

    enabled: bool
    consent_version: int
    data_epoch: int
    account_deletion_pending: bool = False


async def _memory() -> MemoryClient:
    try:
        return await get_memory_client()
    except Exception as exc:
        raise MemoryStoreError("Memory store is unavailable.") from exc


async def list_memories(
    user_id: uuid.UUID,
    *,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[MemoryRecord], bool]:
    """Return one page of everything stored for the user, newest first.

    Inspection does not consult consent: what is stored stays visible after
    memory is switched off.
    """
    client = await _memory()
    try:
        records = await client.list_for_user(str(user_id), limit=offset + limit + 1)
    except Exception as exc:
        raise MemoryStoreError("Failed to list memories.") from exc
    return records[offset : offset + limit], len(records) > offset + limit


async def delete_memory(user_id: uuid.UUID, memory_id: str) -> bool:
    """Delete one memory if it belongs to the user."""
    client = await _memory()
    try:
        if await client.get_owned(str(user_id), memory_id) is None:
            return False
        await client.delete([memory_id])
    except Exception as exc:
        raise MemoryStoreError("Failed to delete the memory.") from exc
    return True


async def delete_conversation_memories(user_id: uuid.UUID, conversation_id: uuid.UUID) -> None:
    client = await _memory()
    try:
        await client.delete_for_conversation(str(user_id), str(conversation_id))
    except Exception as exc:
        raise MemoryStoreError("Failed to delete conversation memories.") from exc


async def delete_user_memories(user_id: uuid.UUID) -> int:
    client = await _memory()
    try:
        return await client.delete_for_user(str(user_id))
    except Exception as exc:
        raise MemoryStoreError("Failed to delete memories.") from exc


async def _write_memory_consent(
    db: AsyncSession,
    user_id: uuid.UUID,
    *,
    enabled: bool,
    bump_data_epoch: bool = False,
    account_deletion_pending: bool | None = None,
    require_account_active: bool = False,
) -> MemoryGateState:
    """Force a consent write and hold the row lock until transaction end.

    The authenticated ``User`` is already present in this session's identity
    map. A SELECT FOR UPDATE would lock the database row but can still return
    that stale Python object, making an apparently redundant assignment skip
    the UPDATE. UPDATE ... RETURNING is both the lock and an unconditional
    database write, so fail-closed transitions cannot be lost.
    """
    values: dict[str, Any] = {
        "memory_enabled": enabled,
        "memory_consent_version": case(
            (
                User.memory_enabled.is_distinct_from(enabled),
                User.memory_consent_version + 1,
            ),
            else_=User.memory_consent_version,
        ),
    }
    if bump_data_epoch:
        values["memory_data_epoch"] = User.memory_data_epoch + 1
    if account_deletion_pending is not None:
        values["account_deletion_pending"] = account_deletion_pending

    statement = update(User).where(User.id == user_id)
    if require_account_active:
        statement = statement.where(User.account_deletion_pending.is_(False))
    result = await db.execute(
        statement
        .values(**values)
        .returning(
            User.memory_enabled,
            User.memory_consent_version,
            User.memory_data_epoch,
            User.account_deletion_pending,
        )
    )
    row = result.one_or_none()
    if row is None:
        raise MemoryUserNotFoundError("User not found.")
    return MemoryGateState(
        enabled=bool(row[0]),
        consent_version=int(row[1]),
        data_epoch=int(row[2]),
        account_deletion_pending=bool(row[3]),
    )


async def set_memory_consent(
    db: AsyncSession,
    user_id: uuid.UUID,
    *,
    enabled: bool,
) -> bool:
    """Atomically change only the user's consent flag."""
    stored = await _write_memory_consent(
        db,
        user_id,
        enabled=enabled,
        require_account_active=True,
    )
    await db.commit()
    return stored.enabled


async def invalidate_memory_for_deletion(
    db: AsyncSession,
    user_id: uuid.UUID,
) -> MemoryGateState:
    """Disable and advance the clear epoch in one commit.

    Jobs already queued carry the old epoch, so the worker rejects them.
    """
    state = await _write_memory_consent(
        db,
        user_id,
        enabled=False,
        bump_data_epoch=True,
    )
    await db.commit()
    return state


async def begin_account_deletion(
    db: AsyncSession,
    user_id: uuid.UUID,
) -> MemoryGateState:
    """Strong-lock User, commit tombstone/epoch, and block new chat FK work."""
    if not await user_repo.lock_for_account_deletion(db, user_id):
        raise MemoryUserNotFoundError("User not found.")
    state = await _write_memory_consent(
        db,
        user_id,
        enabled=False,
        bump_data_epoch=True,
        account_deletion_pending=True,
    )
    await db.commit()
    return state


async def lock_account_deletion(
    db: AsyncSession,
    user_id: uuid.UUID,
) -> MemoryGateState:
    """Reacquire the strong tombstone lock before child resources are locked."""
    if not await user_repo.lock_for_account_deletion(db, user_id):
        raise MemoryUserNotFoundError("User not found.")
    state = await _write_memory_consent(
        db,
        user_id,
        enabled=False,
        account_deletion_pending=True,
    )
    return state


async def clear_memory(db: AsyncSession, user_id: uuid.UUID) -> int:
    """Disable consent, then delete every memory owned by the user.

    Consent and the new data epoch are committed before the store is touched,
    so a store outage cannot leave reads or writes enabled, and a memory from
    the old epoch can no longer be recalled even before it is deleted. A second
    row lock serializes deletion against a concurrent consent update.
    """
    await invalidate_memory_for_deletion(db, user_id)

    try:
        # Re-assert fail-closed consent with an unconditional write in case a
        # concurrent PATCH completed in the small window after the first commit.
        # This UPDATE also holds the user row lock across the deletion.
        await _write_memory_consent(db, user_id, enabled=False)
        deleted = await delete_user_memories(user_id)
    except MemoryStoreError:
        # Persist the second fail-closed write before releasing the row lock.
        await db.commit()
        raise
    except Exception:
        await db.rollback()
        # The first phase remains committed. Best-effort reassertion covers a
        # consent PATCH that may have completed in the phase boundary.
        try:
            await _write_memory_consent(db, user_id, enabled=False)
            await db.commit()
        except Exception:
            await db.rollback()
        raise

    await db.commit()
    return deleted
