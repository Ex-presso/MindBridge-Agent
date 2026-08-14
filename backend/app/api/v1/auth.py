import asyncio
from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth.deps import get_current_user
from app.core.rate_limit import limiter
from app.db.engine import get_db
from app.db.models.user import User
from app.core.auth.password import verify_password
from app.schemas.auth import (
    DeleteAccountRequest,
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from app.services import account_service, auth_service
from config.settings import settings

router = APIRouter(prefix="/auth", tags=["Auth"])
_ACCOUNT_DELETION_TIMEOUT_SECONDS = 30.0

COOKIE_NAME = "refresh_token"
COOKIE_OPTS = {
    "httponly": True,
    "samesite": "lax",
    "secure": settings.COOKIE_SECURE,  # True in production behind HTTPS
    "max_age": settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400,
}


def _set_refresh_cookie(response: Response, token: str) -> None:
    response.set_cookie(COOKIE_NAME, token, **COOKIE_OPTS)


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit(settings.AUTH_RATE_LIMIT)
async def register(request: Request, body: RegisterRequest, response: Response, db: AsyncSession = Depends(get_db)):
    access, refresh = await auth_service.register(db, email=body.email, password=body.password, display_name=body.display_name)
    # The access token is usable as soon as this response reaches the client.
    # Commit first so an immediate authenticated request can see the new user.
    await db.commit()
    _set_refresh_cookie(response, refresh)
    return TokenResponse(access_token=access)


@router.post("/login", response_model=TokenResponse)
@limiter.limit(settings.AUTH_RATE_LIMIT)
async def login(request: Request, body: LoginRequest, response: Response, db: AsyncSession = Depends(get_db)):
    access, refresh = await auth_service.login(db, email=body.email, password=body.password)
    _set_refresh_cookie(response, refresh)
    return TokenResponse(access_token=access)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    response: Response,
    db: AsyncSession = Depends(get_db),
    refresh_token: str | None = Cookie(default=None, alias=COOKIE_NAME),
):
    if not refresh_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No refresh token.")
    access, new_refresh = await auth_service.refresh(db, refresh_token=refresh_token)
    _set_refresh_cookie(response, new_refresh)
    return TokenResponse(access_token=access)


@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie(COOKIE_NAME)
    return {"message": "Logged out."}


@router.get("/me", response_model=UserResponse)
async def me(user: User = Depends(get_current_user)):
    return UserResponse(id=str(user.id), email=user.email, display_name=user.display_name)


@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
@limiter.limit(settings.AUTH_RATE_LIMIT)
async def delete_me(
    request: Request,
    body: DeleteAccountRequest,
    response: Response,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Permanently delete the authenticated account after password confirmation."""
    if not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid credentials.",
        )

    # Imported lazily to avoid the auth/chat router import cycle.
    from app.api.v1.chat import evict_user_runtime

    await evict_user_runtime(user.id)
    try:
        try:
            async with asyncio.timeout(_ACCOUNT_DELETION_TIMEOUT_SECONDS):
                await account_service.delete_account(
                    db,
                    user.id,
                    store=getattr(request.app.state, "store", None),
                    checkpointer=getattr(request.app.state, "checkpointer", None),
                )
        except account_service.AccountUserNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found.",
            ) from exc
        except account_service.AccountDeletionError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Account deletion is temporarily unavailable.",
            ) from exc
        except TimeoutError:
            await db.rollback()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Account deletion is temporarily unavailable.",
            ) from None
    finally:
        # Catch title work scheduled by an in-flight chat while deletion waited
        # on that conversation's row lock.
        await evict_user_runtime(user.id)

    response.delete_cookie(COOKIE_NAME)
    response.status_code = status.HTTP_204_NO_CONTENT
