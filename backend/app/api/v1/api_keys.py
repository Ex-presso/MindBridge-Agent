"""User API key management endpoints."""
import asyncio
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth.deps import get_current_user
from app.core.auth.encryption import decrypt_value, encrypt_value, mask_key
from app.db.engine import get_db
from app.db.models.user import User
from app.db.repositories import api_key_repo, user_repo
from app.core.llm.provider import PROVIDER_MODELS
from app.schemas.api_key import ApiKeyResponse, ApiKeySaveRequest, ProviderModelsResponse, VALID_PROVIDERS

router = APIRouter(prefix="/api-keys", tags=["API Keys"])
_CREDENTIAL_LOCK_TIMEOUT_SECONDS = 5.0


async def _lock_active_credential_owner(
    db: AsyncSession,
    user_id: uuid.UUID,
) -> None:
    """Serialize credential mutation with every outbound user LLM call."""
    try:
        access = await asyncio.wait_for(
            user_repo.get_memory_access_for_update(db, user_id),
            timeout=_CREDENTIAL_LOCK_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Credential mutation is temporarily unavailable.",
        ) from None
    if access is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found.",
        )
    if access.account_deletion_pending:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Account deletion is in progress.",
        )


@router.get("", response_model=list[ApiKeyResponse])
async def list_api_keys(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List user's configured API keys (masked)."""
    records = await api_key_repo.list_by_user(db, user.id)
    return [
        ApiKeyResponse(
            provider=r.provider,
            api_key_masked=mask_key(decrypt_value(r.api_key_encrypted)),
            base_url=r.base_url,
            model_id=r.model_id,
            display_name=r.display_name,
        )
        for r in records
    ]


@router.put("", response_model=ApiKeyResponse)
async def save_api_key(
    body: ApiKeySaveRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Save (create or update) an API key for a provider."""
    if body.provider not in VALID_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Invalid provider. Must be one of: {', '.join(sorted(VALID_PROVIDERS))}")

    if body.provider in ("openai_compatible", "anthropic_compatible") and not body.base_url:
        raise HTTPException(status_code=400, detail="base_url is required for compatible endpoints.")

    # Imported lazily to avoid coupling router import order. User FOR UPDATE
    # waits for active chat/title KEY SHARE barriers. The final eviction catches
    # title work scheduled while this request was waiting for that lock.
    from app.api.v1.chat import evict_user_runtime

    await evict_user_runtime(user.id)
    try:
        await _lock_active_credential_owner(db, user.id)
        encrypted = encrypt_value(body.api_key)
        record = await api_key_repo.upsert(
            db,
            user_id=user.id,
            provider=body.provider,
            api_key_encrypted=encrypted,
            base_url=body.base_url,
            model_id=body.model_id,
            display_name=body.display_name,
        )
        await db.commit()
    finally:
        await evict_user_runtime(user.id)
    return ApiKeyResponse(
        provider=record.provider,
        api_key_masked=mask_key(body.api_key),
        base_url=record.base_url,
        model_id=record.model_id,
        display_name=record.display_name,
    )


@router.delete("/{provider}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_api_key(
    provider: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Remove an API key for a provider."""
    from app.api.v1.chat import evict_user_runtime

    await evict_user_runtime(user.id)
    try:
        await _lock_active_credential_owner(db, user.id)
        deleted = await api_key_repo.delete_by_provider(db, user.id, provider)
        if not deleted:
            raise HTTPException(status_code=404, detail="API key not found for this provider.")
        await db.commit()
    finally:
        await evict_user_runtime(user.id)


@router.get("/models", response_model=list[ProviderModelsResponse])
async def get_available_models(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return available models based on the user's configured API keys."""
    records = await api_key_repo.list_by_user(db, user.id)
    configured_providers = {r.provider for r in records}
    # Build a map of custom models for compatible endpoints
    compatible_models = {}
    for r in records:
        if r.provider in ("openai_compatible", "anthropic_compatible") and r.model_id:
            compatible_models[r.provider] = [{"id": r.model_id, "name": r.display_name or r.model_id}]

    result = []
    for provider, models in PROVIDER_MODELS.items():
        is_configured = provider in configured_providers
        if provider in compatible_models:
            models = compatible_models[provider]
        result.append(ProviderModelsResponse(
            provider=provider,
            configured=is_configured,
            models=models if is_configured else [],
        ))
    return result
