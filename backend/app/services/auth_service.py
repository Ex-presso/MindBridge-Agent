import re
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException, status

from app.core.auth.jwt import create_access_token, create_refresh_token, decode_token
from app.core.auth.password import hash_password, verify_password
from app.db.repositories import user_repo
from jose import JWTError


_PASSWORD_RE = re.compile(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$')


async def register(db: AsyncSession, *, email: str, password: str, display_name: str | None) -> tuple[str, str]:
    """Returns (access_token, refresh_token). Raises 400 if email taken or password weak."""
    if not _PASSWORD_RE.match(password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters and contain uppercase, lowercase, and a digit.",
        )
    if await user_repo.email_exists(db, email):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Registration failed.")
    hashed = hash_password(password)
    user = await user_repo.create(db, email=email, hashed_password=hashed, display_name=display_name)
    return create_access_token(user.id), create_refresh_token(user.id)


async def login(db: AsyncSession, *, email: str, password: str) -> tuple[str, str]:
    """Returns (access_token, refresh_token). Generic error to avoid enumeration."""
    user = await user_repo.get_by_email(db, email)
    if user is None or not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials.")
    return create_access_token(user.id), create_refresh_token(user.id)


async def refresh(db: AsyncSession, *, refresh_token: str) -> tuple[str, str]:
    """Validates refresh token, returns new (access_token, refresh_token)."""
    try:
        payload = decode_token(refresh_token)
        if payload.get("type") != "refresh":
            raise JWTError("Wrong token type")
        user_id = uuid.UUID(payload["sub"])
    except (JWTError, ValueError, KeyError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token.")

    user = await user_repo.get_by_id(db, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token.")

    return create_access_token(user.id), create_refresh_token(user.id)
