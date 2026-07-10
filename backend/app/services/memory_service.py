"""Privacy controls and transparent access for durable user memory."""

from collections.abc import Sequence
from typing import Any, Literal, TypeAlias
import uuid

from langgraph.store.base import PutOp
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.user import User


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
) -> bool:
    """Force a consent write and hold the row lock until transaction end.

    The authenticated ``User`` is already present in this session's identity
    map. A SELECT FOR UPDATE would lock the database row but can still return
    that stale Python object, making an apparently redundant assignment skip
    the UPDATE. UPDATE ... RETURNING is both the lock and an unconditional
    database write, so fail-closed transitions cannot be lost.
    """
    result = await db.execute(
        update(User)
        .where(User.id == user_id)
        .values(memory_enabled=enabled)
        .returning(User.memory_enabled)
    )
    stored_value = result.scalar_one_or_none()
    if stored_value is None:
        raise MemoryUserNotFoundError("User not found.")
    return bool(stored_value)


async def set_memory_consent(
    db: AsyncSession,
    user_id: uuid.UUID,
    *,
    enabled: bool,
) -> bool:
    """Atomically change only the user's consent flag."""
    stored_value = await _write_memory_consent(db, user_id, enabled=enabled)
    await db.commit()
    return stored_value


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
    await _write_memory_consent(db, user_id, enabled=False)
    await db.commit()

    deleted = 0
    prefix = memory_user_prefix(user_id)
    previous_batch: tuple[tuple[tuple[str, ...], str], ...] | None = None

    try:
        # Re-assert fail-closed consent with an unconditional write in case a
        # concurrent PATCH completed in the small window after the first commit.
        # This UPDATE also holds the user row lock across Store deletion.
        await _write_memory_consent(db, user_id, enabled=False)
        if store is None:
            raise MemoryStoreError("Memory Store is unavailable.")
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
    except MemoryStoreError:
        # Persist the second fail-closed write before releasing the row lock.
        await db.commit()
        raise
    except Exception:
        await db.rollback()
        raise

    await db.commit()
    return deleted
