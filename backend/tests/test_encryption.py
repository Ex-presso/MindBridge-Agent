"""Fernet API-key encryption roundtrip + masking."""
import pytest
from cryptography.fernet import InvalidToken

from app.core.auth import encryption
from app.core.auth.encryption import decrypt_value, encrypt_value, mask_key
from config.settings import settings


def test_encrypt_decrypt_roundtrip():
    plain = "sk-proj-abc123def456"
    assert decrypt_value(encrypt_value(plain)) == plain


def test_encrypt_is_not_plaintext():
    plain = "sk-secret"
    assert plain not in encrypt_value(plain)


def test_encrypt_nondeterministic():
    # Fernet embeds a random IV + timestamp, so ciphertexts differ each call.
    assert encrypt_value("same") != encrypt_value("same")


def test_mask_key_shows_only_last_four():
    masked = mask_key("sk-proj-abcdEFGH")
    assert masked.endswith("EFGH")
    assert "abcd" not in masked


def test_mask_short_key_fully_hidden():
    assert mask_key("ab") == "••••••••"


def test_dedicated_encryption_key_separates_from_secret(monkeypatch):
    # Ciphertext made with the SECRET_KEY fallback...
    token = encrypt_value("secret-value")
    # ...cannot be decrypted once a distinct ENCRYPTION_KEY is set,
    # proving the two keys are independent (JWT key can rotate safely).
    monkeypatch.setattr(settings, "ENCRYPTION_KEY", "a-different-dedicated-encryption-key")
    with pytest.raises(InvalidToken):
        decrypt_value(token)
    # Roundtrip still works under the new key.
    assert decrypt_value(encrypt_value("y")) == "y"
    assert encryption._get_cipher() is not None
