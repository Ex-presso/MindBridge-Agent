"""Request schema validation."""

import pytest
from pydantic import ValidationError

from app.schemas.conversation import SessionChatRequest


def test_defaults():
    req = SessionChatRequest(message="hi")
    assert req.stream is True
    assert req.conversation_id is None
    assert req.model == "deepseek-v4-flash"
    assert req.provider == "openai_compatible"


def test_message_length_cap():
    with pytest.raises(ValidationError):
        SessionChatRequest(message="x" * 4001)
