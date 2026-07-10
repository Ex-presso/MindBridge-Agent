"""Retry-safe account deletion across relational and LangGraph storage."""

import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories import conversation_repo, user_repo
from app.services import memory_service


class AccountDeletionError(RuntimeError):
    """Account deletion could not complete without leaving retryable state."""


class AccountUserNotFoundError(AccountDeletionError):
    """The target user disappeared before cleanup completed."""


async def delete_account(
    db: AsyncSession,
    user_id: uuid.UUID,
    *,
    store: Any,
    checkpointer: Any,
) -> int:
    """Delete all user data, retaining a disabled row on external failure.

    Phase one commits the consent/version tombstone. Phase two locks User first,
    then every Conversation in stable UUID order, clears Store and checkpoints,
    and only then removes the relational User (whose FKs cascade). Every
    external operation is idempotent, so a partial failure is safe to retry.
    """
    try:
        await memory_service.invalidate_memory_for_deletion(db, user_id)
    except memory_service.MemoryUserNotFoundError as exc:
        raise AccountUserNotFoundError("User not found.") from exc
    except Exception as exc:
        await db.rollback()
        raise AccountDeletionError("Account invalidation failed.") from exc

    try:
        # Fixed lock order: User → Conversations. Chat never takes a User row
        # lock after its Conversation lock, so it cannot form the reverse edge.
        await memory_service.lock_memory_disabled(db, user_id)
        conversations = await conversation_repo.list_owned_for_update(db, user_id)

        if store is None or checkpointer is None:
            raise AccountDeletionError("Account storage cleanup is unavailable.")

        await memory_service.delete_user_memory_items(store, user_id)
        for conversation in conversations:
            try:
                await checkpointer.adelete_thread(str(conversation.id))
            except Exception as exc:
                raise AccountDeletionError(
                    "Conversation checkpoint cleanup failed."
                ) from exc

        deleted = await user_repo.delete_by_id(db, user_id)
        if not deleted:
            raise AccountUserNotFoundError("User not found.")
        await db.commit()
        return len(conversations)
    except memory_service.MemoryUserNotFoundError as exc:
        await db.rollback()
        raise AccountUserNotFoundError("User not found.") from exc
    except memory_service.MemoryStoreError as exc:
        await db.rollback()
        raise AccountDeletionError("Account memory cleanup failed.") from exc
    except Exception:
        await db.rollback()
        raise
