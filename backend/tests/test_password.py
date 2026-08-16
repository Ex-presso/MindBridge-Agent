"""Password hashing + strength policy."""
import pytest

from app.core.auth.password import hash_password, verify_password
from app.services.auth_service import _PASSWORD_RE


def test_hash_verify_roundtrip():
    hashed = hash_password("Str0ngPass")
    assert verify_password("Str0ngPass", hashed)
    assert not verify_password("wrong", hashed)


def test_hash_is_salted():
    # bcrypt salts each hash, so the same password hashes differently.
    assert hash_password("Str0ngPass") != hash_password("Str0ngPass")


@pytest.mark.parametrize("pw", ["Abcdef12", "LongEnough9X"])
def test_password_policy_accepts_valid(pw):
    assert _PASSWORD_RE.match(pw)


@pytest.mark.parametrize(
    "pw",
    ["short1A", "alllowercase1", "ALLUPPERCASE1", "NoDigitsHere"],
)
def test_password_policy_rejects_invalid(pw):
    assert _PASSWORD_RE.match(pw) is None
