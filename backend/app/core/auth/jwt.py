import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from jose import jwt

from config.settings import settings

ALGORITHM = "HS256"


def create_access_token(subject: str | uuid.UUID) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": str(subject), "exp": expire, "type": "access"}
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(subject: str | uuid.UUID) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    payload = {"sub": str(subject), "exp": expire, "type": "refresh", "jti": str(uuid.uuid4())}
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict[str, Any]:
    """Raises JWTError on invalid/expired token."""
    return jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
