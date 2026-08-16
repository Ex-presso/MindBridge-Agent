"""Privacy controls and transparent access for durable user memory."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias
import uuid

from langgraph.store.base import PutOp
from sqlalchemy import case, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.user import User
from app.db.repositories import memory_job_repo, user_repo


MemoryCategory: TypeAlias = Literal["semantic", "episodes"]
MEMORY_CATEGORIES = frozenset({"semantic", "episodes"})
MEMORY_ROOT = "memory"
_CLEAR_BATCH_SIZE = 100


class MemoryServiceError(RuntimeError):
    """Base error for memory service operations."""


class MemoryStoreError(MemoryServiceError):
    """The durable Store is unavailable or returned unsafe data."""


class MemoryUserNotFoundError(MemoryServiceError):
    """The target user no longer exists."""


@dataclass(frozen=True, slots=True)
class MemoryGateState:
    """Database-authoritative consent and invalidation versions."""

    enabled: bool
    consent_version: int
    data_epoch: int
    account_deletion_pending: bool = False


def memory_namespace(
    user_id: uuid.UUID | str,
    category: MemoryCategory,
) -> tuple[str, str, str]:
    """Build the only namespace shape used for durable MindBridge memory."""
    if category not in MEMORY_CATEGORIES:
        raise ValueError(f"Unsupported memory category: {category!r}")
    return (MEMORY_ROOT, str(user_id), category)


def memory_user_prefix(user_id: uuid.UUID | str) -> tuple[str, str]:
    """Return the prefix owning every durable memory item for one user."""
    return (MEMORY_ROOT, str(user_id))


def episode_namespace(user_id: uuid.UUID | str) -> tuple[str, str, str]:
    """Return the canonical namespace for conversation episodes."""
    return memory_namespace(user_id, "episodes")


def _validate_returned_namespaces(
    items: Sequence[Any],
    *,
    user_id: uuid.UUID | str,
    category: MemoryCategory | None,
) -> None:
    """Reject Store results outside the exact user/category namespace shape."""
    user_prefix = memory_user_prefix(user_id)
    expected = memory_namespace(user_id, category) if category is not None else None

    for item in items:
        namespace = tuple(getattr(item, "namespace", ()))
        if expected is not None:
            valid = namespace == expected
        else:
            valid = (
                len(namespace) == 3
                and namespace[:2] == user_prefix
                and isinstance(namespace[2], str)
                and bool(namespace[2])
            )
        if not valid:
            raise MemoryStoreError(
                "Memory Store returned an item outside the requested namespace."
            )


async def list_memory_items(
    store: Any,
    user_id: uuid.UUID | str,
    *,
    category: MemoryCategory | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[Any], bool]:
    """Return a transparent, user-scoped page without consulting consent."""
    if store is None:
        raise MemoryStoreError("Memory Store is unavailable.")
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if offset < 0:
        raise ValueError("offset must not be negative")
    if category is not None and category not in MEMORY_CATEGORIES:
        raise ValueError(f"Unsupported memory category: {category!r}")

    prefix = (
        memory_namespace(user_id, category)
        if category is not None
        else memory_user_prefix(user_id)
    )
    try:
        results = await store.asearch(
            prefix,
            limit=limit + 1,
            offset=offset,
            refresh_ttl=False,
        )
    except MemoryStoreError:
        raise
    except Exception as exc:
        raise MemoryStoreError("Failed to list durable memory.") from exc

    _validate_returned_namespaces(results, user_id=user_id, category=category)
    return list(results[:limit]), len(results) > limit


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
    """Disable, advance the clear epoch, cancel jobs, and commit atomically."""
    state = await _write_memory_consent(
        db,
        user_id,
        enabled=False,
        bump_data_epoch=True,
    )
    await memory_job_repo.cancel_unfinished_for_user(db, user_id)
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
    await memory_job_repo.cancel_unfinished_for_user(db, user_id)
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


async def delete_user_memory_items(
    store: Any,
    user_id: uuid.UUID | str,
) -> int:
    """Delete every Store item under one validated user prefix."""
    if store is None:
        raise MemoryStoreError("Memory Store is unavailable.")

    deleted = 0
    prefix = memory_user_prefix(user_id)
    previous_batch: tuple[tuple[tuple[str, ...], str], ...] | None = None
    while True:
        try:
            items = await store.asearch(
                prefix,
                limit=_CLEAR_BATCH_SIZE,
                offset=0,
                refresh_ttl=False,
            )
        except Exception as exc:
            raise MemoryStoreError("Failed to search durable memory.") from exc

        if not items:
            break

        _validate_returned_namespaces(items, user_id=user_id, category=None)
        batch_identity = tuple(
            (tuple(item.namespace), str(item.key)) for item in items
        )
        if batch_identity == previous_batch:
            raise MemoryStoreError("Memory Store deletion made no progress.")
        previous_batch = batch_identity

        operations = [
            PutOp(
                namespace=tuple(item.namespace),
                key=str(item.key),
                value=None,
            )
            for item in items
        ]
        try:
            await store.abatch(operations)
        except Exception as exc:
            raise MemoryStoreError("Failed to delete durable memory.") from exc
        deleted += len(operations)

    return deleted


async def clear_memory(
    db: AsyncSession,
    store: Any,
    user_id: uuid.UUID,
) -> int:
    """Disable consent, then delete every Store item owned by the user.

    Consent is committed before Store access, so a partial Store outage cannot
    leave future reads or writes enabled. A second row lock serializes deletion
    against a concurrent consent update while batches are being removed.
    """
    await invalidate_memory_for_deletion(db, user_id)

    try:
        # Re-assert fail-closed consent with an unconditional write in case a
        # concurrent PATCH completed in the small window after the first commit.
        # This UPDATE also holds the user row lock across Store deletion.
        await _write_memory_consent(db, user_id, enabled=False)
        await memory_job_repo.cancel_unfinished_for_user(db, user_id)
        deleted = await delete_user_memory_items(store, user_id)
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
