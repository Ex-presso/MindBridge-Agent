"""JWT access/refresh token creation and decoding."""
import uuid

import pytest
from jose import JWTError

from app.core.auth.jwt import create_access_token, create_refresh_token, decode_token


def test_access_token_roundtrip():
    uid = uuid.uuid4()
    payload = decode_token(create_access_token(uid))
    assert payload["sub"] == str(uid)
    assert payload["type"] == "access"


def test_refresh_token_has_type_and_jti():
    payload = decode_token(create_refresh_token(uuid.uuid4()))
    assert payload["type"] == "refresh"
    assert "jti" in payload


def test_tampered_token_rejected():
    token = create_access_token(uuid.uuid4())
    with pytest.raises(JWTError):
        decode_token(token + "tampered")
