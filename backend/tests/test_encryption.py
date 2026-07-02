"""Fernet API-key encryption roundtrip + masking."""
from app.core.auth.encryption import decrypt_value, encrypt_value, mask_key


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
