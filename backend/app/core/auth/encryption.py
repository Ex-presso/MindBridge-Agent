"""Fernet symmetric encryption for storing API keys."""
import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken

from config.settings import settings


def _get_cipher() -> Fernet:
    key = hashlib.sha256(settings.SECRET_KEY.encode()).digest()
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
