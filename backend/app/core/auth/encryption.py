"""Fernet symmetric encryption for storing API keys."""
import base64
import hashlib

from cryptography.fernet import Fernet

from config.settings import settings


def _get_cipher() -> Fernet:
    # Prefer a dedicated encryption key; fall back to SECRET_KEY so existing
    # deployments keep working. Separating them lets the JWT key rotate without
    # invalidating stored API keys.
    secret = settings.ENCRYPTION_KEY or settings.SECRET_KEY
    key = hashlib.sha256(secret.encode()).digest()
    return Fernet(base64.urlsafe_b64encode(key))


def encrypt_value(plain: str) -> str:
    return _get_cipher().encrypt(plain.encode()).decode()


def decrypt_value(encrypted: str) -> str:
    return _get_cipher().decrypt(encrypted.encode()).decode()


def mask_key(key: str) -> str:
    """Show only last 4 chars for display."""
    if len(key) <= 8:
        return "••••" + key[-4:] if len(key) > 4 else "••••••••"
    return "••••••••" + key[-4:]
