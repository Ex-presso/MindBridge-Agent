"""Authenticated privacy controls for durable cross-conversation memory."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth.deps import get_current_user
from app.db.engine import get_db
from app.db.models.user import User
from app.schemas.memory import (
    MemoryCategory,
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


def _get_store(request: Request):
    store = getattr(request.app.state, "store", None)
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memory store unavailable.",
        )
    return store


def _store_unavailable(exc: memory_service.MemoryStoreError) -> HTTPException:
    logger.warning("Memory Store operation failed: %s", exc)
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Memory store unavailable.",
    )


def _item_response(item) -> MemoryItemResponse:
    return MemoryItemResponse(
        namespace=list(item.namespace),
        key=str(item.key),
        value=item.value,
        created_at=item.created_at,
        updated_at=item.updated_at,
        score=getattr(item, "score", None),
    )


@router.get("", response_model=MemoryListResponse)
async def list_memory(
    request: Request,
    category: MemoryCategory | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    user: User = Depends(get_current_user),
):
    """List stored memory even when consent or the global switch is off."""
    store = _get_store(request)
    try:
        items, has_more = await memory_service.list_memory_items(
            store,
            user.id,
            category=category,
            limit=limit,
            offset=offset,
        )
    except memory_service.MemoryStoreError as exc:
        raise _store_unavailable(exc) from exc

    return MemoryListResponse(
        memory_enabled=user.memory_enabled,
        system_enabled=settings.MEMORY_ENABLED,
        items=[_item_response(item) for item in items],
        limit=limit,
        offset=offset,
        has_more=has_more,
    )


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
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Disable memory and permanently remove every Store item for the user."""
    # Pass an unavailable Store through to the service so consent is committed
    # off before the fail-closed 503 response is produced.
    store = getattr(request.app.state, "store", None)
    try:
        deleted = await memory_service.clear_memory(db, store, user.id)
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
