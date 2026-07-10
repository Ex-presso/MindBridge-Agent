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
    store = SimpleNamespace(adelete=AsyncMock())
    get_owned_for_update = AsyncMock(return_value=SimpleNamespace(user_id=user_id))
    delete_by_id = AsyncMock(return_value=True)
    monkeypatch.setattr(
        conversation_service.conversation_repo,
        "get_owned_for_update",
        get_owned_for_update,
    )
    monkeypatch.setattr(conversation_service.conversation_repo, "delete_by_id", delete_by_id)

    deleted = asyncio.run(
        conversation_service.delete_conversation(
            db,
            conversation_id,
            user_id,
            checkpointer=checkpointer,
            store=store,
        )
    )

    assert deleted is True
    store.adelete.assert_awaited_once_with(
        conversation_service.episode_namespace(user_id),
        str(conversation_id),
    )
    checkpointer.adelete_thread.assert_awaited_once_with(str(conversation_id))
    delete_by_id.assert_awaited_once_with(db, conversation_id)


def test_delete_conversation_does_not_hide_checkpoint_failure(monkeypatch):
    conversation_id = uuid.uuid4()
    user_id = uuid.uuid4()
    db = AsyncMock()
    store = SimpleNamespace(adelete=AsyncMock())
    checkpointer = SimpleNamespace(
        adelete_thread=AsyncMock(side_effect=RuntimeError("checkpoint unavailable"))
    )
    get_owned_for_update = AsyncMock(return_value=SimpleNamespace(user_id=user_id))
    delete_by_id = AsyncMock(return_value=True)
    monkeypatch.setattr(
        conversation_service.conversation_repo,
        "get_owned_for_update",
        get_owned_for_update,
    )
    monkeypatch.setattr(conversation_service.conversation_repo, "delete_by_id", delete_by_id)

    with pytest.raises(RuntimeError, match="checkpoint unavailable"):
        asyncio.run(
            conversation_service.delete_conversation(
                db,
                conversation_id,
                user_id,
                checkpointer=checkpointer,
                store=store,
            )
        )

    delete_by_id.assert_not_awaited()


def test_delete_conversation_stops_when_episode_cleanup_fails(monkeypatch):
    conversation_id = uuid.uuid4()
    user_id = uuid.uuid4()
    db = AsyncMock()
    store = SimpleNamespace(
        adelete=AsyncMock(side_effect=RuntimeError("store unavailable"))
    )
    checkpointer = SimpleNamespace(adelete_thread=AsyncMock())
    get_owned_for_update = AsyncMock(return_value=SimpleNamespace(user_id=user_id))
    delete_by_id = AsyncMock(return_value=True)
    monkeypatch.setattr(
        conversation_service.conversation_repo,
        "get_owned_for_update",
        get_owned_for_update,
    )
    monkeypatch.setattr(conversation_service.conversation_repo, "delete_by_id", delete_by_id)

    with pytest.raises(RuntimeError, match="store unavailable"):
        asyncio.run(
            conversation_service.delete_conversation(
                db,
                conversation_id,
                user_id,
                checkpointer=checkpointer,
                store=store,
            )
        )

    checkpointer.adelete_thread.assert_not_awaited()
    delete_by_id.assert_not_awaited()


def test_delete_conversation_not_found_touches_no_external_state(monkeypatch):
    conversation_id = uuid.uuid4()
    user_id = uuid.uuid4()
    db = AsyncMock()
    store = SimpleNamespace(adelete=AsyncMock())
    checkpointer = SimpleNamespace(adelete_thread=AsyncMock())
    get_owned_for_update = AsyncMock(return_value=None)
    delete_by_id = AsyncMock(return_value=True)
    monkeypatch.setattr(
        conversation_service.conversation_repo,
        "get_owned_for_update",
        get_owned_for_update,
    )
    monkeypatch.setattr(conversation_service.conversation_repo, "delete_by_id", delete_by_id)

    deleted = asyncio.run(
        conversation_service.delete_conversation(
            db,
            conversation_id,
            user_id,
            checkpointer=checkpointer,
            store=store,
        )
    )

    assert deleted is False
    store.adelete.assert_not_awaited()
    checkpointer.adelete_thread.assert_not_awaited()
    delete_by_id.assert_not_awaited()
