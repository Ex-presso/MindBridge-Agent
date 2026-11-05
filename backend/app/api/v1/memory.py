"""Authenticated privacy controls for durable cross-conversation memory."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth.deps import get_current_user
from app.db.engine import get_db
from app.db.models.user import User
from app.schemas.memory import (
    MemoryConsentResponse,
    MemoryConsentUpdate,
    MemoryDeleteResponse,
    MemoryItemResponse,
    MemoryListResponse,
)
from app.services import memory_service
from config.settings import settings


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/memory", tags=["Memory"])


def _store_unavailable(exc: memory_service.MemoryStoreError) -> HTTPException:
    logger.warning("Memory store operation failed: %s", exc)
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Memory store unavailable.",
    )


@router.get("", response_model=MemoryListResponse)
async def list_memory(
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    user: User = Depends(get_current_user),
):
    """List stored memory even when consent or the global switch is off."""
    try:
        records, has_more = await memory_service.list_memories(
            user.id,
            limit=limit,
            offset=offset,
        )
    except memory_service.MemoryStoreError as exc:
        raise _store_unavailable(exc) from exc

    return MemoryListResponse(
        memory_enabled=user.memory_enabled,
        system_enabled=settings.MEMORY_ENABLED,
        items=[
            MemoryItemResponse(
                id=record.id,
                text=record.text,
                conversation_id=record.conversation_id,
                created_at=record.created_at,
            )
            for record in records
        ],
        limit=limit,
        offset=offset,
        has_more=has_more,
    )


@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory_item(
    memory_id: str,
    user: User = Depends(get_current_user),
):
    """Delete one of the user's own memories."""
    try:
        deleted = await memory_service.delete_memory(user.id, memory_id)
    except memory_service.MemoryStoreError as exc:
        raise _store_unavailable(exc) from exc
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found.")


@router.patch("", response_model=MemoryConsentResponse)
async def update_memory_consent(
    body: MemoryConsentUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Change per-user consent without reading or writing the Store."""
    try:
        enabled = await memory_service.set_memory_consent(
            db,
            user.id,
            enabled=body.memory_enabled,
        )
    except memory_service.MemoryUserNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found.",
        ) from exc
    return MemoryConsentResponse(
        memory_enabled=enabled,
        system_enabled=settings.MEMORY_ENABLED,
    )


@router.delete("", response_model=MemoryDeleteResponse)
async def delete_memory(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Disable memory and permanently remove every memory stored for the user."""
    try:
        deleted = await memory_service.clear_memory(db, user.id)
    except memory_service.MemoryStoreError as exc:
        raise _store_unavailable(exc) from exc
    except memory_service.MemoryUserNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found.",
        ) from exc

    return MemoryDeleteResponse(
        system_enabled=settings.MEMORY_ENABLED,
        deleted_items=deleted,
    )
