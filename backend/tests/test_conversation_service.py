"""Conversation deletion keeps relational and LangGraph state aligned."""

import asyncio
from types import SimpleNamespace
import uuid
from unittest.mock import AsyncMock

import pytest

from app.services import conversation_service


def test_delete_conversation_passes_raw_thread_id(monkeypatch):
    conversation_id = uuid.uuid4()
    user_id = uuid.uuid4()
    db = AsyncMock()
    checkpointer = SimpleNamespace(adelete_thread=AsyncMock())
    get_by_id = AsyncMock(return_value=SimpleNamespace(user_id=user_id))
    delete_by_id = AsyncMock(return_value=True)
    monkeypatch.setattr(conversation_service.conversation_repo, "get_by_id", get_by_id)
    monkeypatch.setattr(conversation_service.conversation_repo, "delete_by_id", delete_by_id)

    deleted = asyncio.run(
        conversation_service.delete_conversation(
            db,
            conversation_id,
            user_id,
            checkpointer=checkpointer,
        )
    )

    assert deleted is True
    checkpointer.adelete_thread.assert_awaited_once_with(str(conversation_id))
    delete_by_id.assert_awaited_once_with(db, conversation_id)


def test_delete_conversation_does_not_hide_checkpoint_failure(monkeypatch):
    conversation_id = uuid.uuid4()
    user_id = uuid.uuid4()
    db = AsyncMock()
    checkpointer = SimpleNamespace(
        adelete_thread=AsyncMock(side_effect=RuntimeError("checkpoint unavailable"))
    )
    get_by_id = AsyncMock(return_value=SimpleNamespace(user_id=user_id))
    delete_by_id = AsyncMock(return_value=True)
    monkeypatch.setattr(conversation_service.conversation_repo, "get_by_id", get_by_id)
    monkeypatch.setattr(conversation_service.conversation_repo, "delete_by_id", delete_by_id)

    with pytest.raises(RuntimeError, match="checkpoint unavailable"):
        asyncio.run(
            conversation_service.delete_conversation(
                db,
                conversation_id,
                user_id,
                checkpointer=checkpointer,
            )
        )

    delete_by_id.assert_not_awaited()
