import asyncio
import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from app.api.v1 import chat as chat_api
from app.schemas.conversation import SessionChatRequest


class _Transaction:
    def __init__(
        self,
        events: list[str],
        exit_error: Exception | None = None,
    ) -> None:
        self.events = events
        self.exit_error = exit_error

    async def __aenter__(self):
        self.events.append("transaction_enter")
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> bool:
        outcome = "commit" if exc_type is None else f"rollback:{exc_type.__name__}"
        self.events.append(f"transaction_exit:{outcome}")
        if exc_type is None and self.exit_error is not None:
            raise self.exit_error
        return False


class _Session:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.transaction_exit_error: Exception | None = None

    async def __aenter__(self):
        self.events.append("session_enter")
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> bool:
        self.events.append("session_exit")
        return False

    def begin(self) -> _Transaction:
        return _Transaction(self.events, self.transaction_exit_error)


def _sse_payload(frame: bytes):
    data = frame.decode().removeprefix("data: ").removesuffix("\n\n")
    return data if data == "[DONE]" else json.loads(data)


def _arrange_existing_conversation(monkeypatch, *, stream: bool):
    events: list[str] = []
    user_id = uuid.uuid4()
    conversation_id = uuid.uuid4()
    conversation = SimpleNamespace(
        id=conversation_id,
        user_id=user_id,
        memory_revision=0,
    )
    # Simulate an authenticated ORM object loaded before a PATCH disabled
    # memory. The generation path must ignore this stale value.
    user = SimpleNamespace(
        id=user_id,
        memory_enabled=True,
        memory_consent_version=3,
        memory_data_epoch=4,
    )
    request_db = AsyncMock()

    async def commit_request():
        events.append("request_commit")

    request_db.commit.side_effect = commit_request
    generation_session = _Session(events)
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                checkpointer=object(),
                store=object(),
                db_session=lambda: generation_session,
            )
        )
    )
    body = SessionChatRequest(
        message="hello",
        conversation_id=str(conversation_id),
        provider="openai",
        model="test-model",
        stream=stream,
    )

    key_record = SimpleNamespace(api_key_encrypted="encrypted", base_url=None)
    monkeypatch.setattr(
        "app.db.repositories.api_key_repo.get_by_provider",
        AsyncMock(return_value=key_record),
    )
    monkeypatch.setattr(
        "app.core.auth.encryption.decrypt_value",
        lambda _: "decrypted",
    )
    monkeypatch.setattr(chat_api, "_get_agent", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        chat_api.user_repo,
        "get_memory_access_snapshot",
        AsyncMock(
            return_value=chat_api.user_repo.MemoryAccessSnapshot(
                enabled=False,
                consent_version=3,
                data_epoch=4,
            )
        ),
    )
    async def lock_conversation(db, locked_conversation_id, locked_user_id):
        lock_scope = "initial" if db is request_db else "generation"
        events.append(f"conversation_lock:{lock_scope}")
        return conversation

    monkeypatch.setattr(
        chat_api.conversation_repo,
        "get_owned_for_update",
        lock_conversation,
    )

    async def create_message(db, **kwargs):
        events.append(f"message_create:{kwargs['role']}")
        return SimpleNamespace()

    async def touch_conversation(db, conversation_id, **kwargs):
        events.append("conversation_touch")

    monkeypatch.setattr(chat_api.message_repo, "create", create_message)
    monkeypatch.setattr(chat_api.conversation_repo, "touch", touch_conversation)

    async def mark_crisis(db, conversation_id, user_id):
        events.append("memory_crisis_seen")
        return True

    monkeypatch.setattr(
        chat_api.conversation_repo,
        "mark_memory_crisis_seen",
        mark_crisis,
    )

    async def enqueue_delete_episode(db, **kwargs):
        events.append("delete_episode_enqueued")
        return SimpleNamespace()

    monkeypatch.setattr(
        chat_api.memory_job_repo,
        "enqueue_delete_episode",
        enqueue_delete_episode,
    )

    async def increment_revision(db, conversation_id):
        events.append("memory_revision_increment")
        return 1

    monkeypatch.setattr(
        chat_api.conversation_repo,
        "increment_memory_revision",
        increment_revision,
    )

    return SimpleNamespace(
        body=body,
        conversation=conversation,
        events=events,
        request=request,
        request_db=request_db,
        user=user,
    )


def test_crisis_tombstone_commits_with_user_message(monkeypatch):
    async def scenario():
        arranged = _arrange_existing_conversation(monkeypatch, stream=False)
        arranged.body.message = "I want to end my life."

        async def run_graph(*args, **kwargs):
            return "Please contact emergency support now."

        monkeypatch.setattr(chat_api.chat_service, "run_chat_session", run_graph)

        response = await chat_api.session_chat(
            arranged.request,
            arranged.body,
            arranged.user,
            arranged.request_db,
        )

        assert response["message"]["content"].startswith("Please contact")
        assert arranged.events.index("message_create:user") < arranged.events.index(
            "memory_crisis_seen"
        )
        assert arranged.events.index("memory_crisis_seen") < arranged.events.index(
            "delete_episode_enqueued"
        )
        assert arranged.events.index("delete_episode_enqueued") < arranged.events.index(
            "request_commit"
        )

    asyncio.run(scenario())


def test_stream_rechecks_locked_conversation_before_invoking_graph(monkeypatch):
    async def scenario():
        arranged = _arrange_existing_conversation(monkeypatch, stream=True)

        async def lock_conversation(db, conversation_id, user_id):
            if db is arranged.request_db:
                arranged.events.append("conversation_lock:initial")
                return arranged.conversation
            arranged.events.append("conversation_lock:generation_missing")
            return None

        async def must_not_stream(*args, **kwargs):
            raise AssertionError("graph must not run after conversation deletion")
            yield  # pragma: no cover - makes this an async generator

        monkeypatch.setattr(
            chat_api.conversation_repo,
            "get_owned_for_update",
            lock_conversation,
        )
        monkeypatch.setattr(
            chat_api.chat_service,
            "stream_chat_session",
            must_not_stream,
        )

        response = await chat_api.session_chat(
            arranged.request,
            arranged.body,
            arranged.user,
            arranged.request_db,
        )
        frames = [_sse_payload(frame) async for frame in response.body_iterator]

        assert frames == [
            {"error": "Conversation no longer exists."},
            "[DONE]",
        ]
        assert arranged.events == [
            "conversation_lock:initial",
            "message_create:user",
            "conversation_touch",
            "request_commit",
            "session_enter",
            "transaction_enter",
            "conversation_lock:generation_missing",
            "transaction_exit:commit",
            "session_exit",
        ]

    asyncio.run(scenario())


def test_nonstream_rechecks_locked_conversation_before_invoking_graph(monkeypatch):
    async def scenario():
        arranged = _arrange_existing_conversation(monkeypatch, stream=False)
        run_graph = AsyncMock()
        async def lock_conversation(db, conversation_id, user_id):
            if db is arranged.request_db:
                arranged.events.append("conversation_lock:initial")
                return arranged.conversation
            arranged.events.append("conversation_lock:generation_missing")
            return None

        monkeypatch.setattr(
            chat_api.conversation_repo,
            "get_owned_for_update",
            lock_conversation,
        )
        monkeypatch.setattr(chat_api.chat_service, "run_chat_session", run_graph)

        with pytest.raises(HTTPException) as exc_info:
            await chat_api.session_chat(
                arranged.request,
                arranged.body,
                arranged.user,
                arranged.request_db,
            )

        assert exc_info.value.status_code == 409
        assert exc_info.value.detail == "Conversation no longer exists."
        run_graph.assert_not_awaited()
        assert "message_create:assistant" not in arranged.events
        assert arranged.events.index("conversation_lock:initial") < arranged.events.index(
            "message_create:user"
        )
        assert arranged.events.index("request_commit") < arranged.events.index(
            "conversation_lock:generation_missing"
        )

    asyncio.run(scenario())


def test_stream_persists_reply_and_releases_lock_before_done(monkeypatch):
    async def scenario():
        arranged = _arrange_existing_conversation(monkeypatch, stream=True)

        async def lock_conversation(db, conversation_id, user_id):
            lock_scope = "initial" if db is arranged.request_db else "generation"
            arranged.events.append(f"conversation_lock:{lock_scope}")
            return arranged.conversation

        async def stream_graph(
            agent,
            message,
            thread_id,
            *,
            user_id,
            memory_enabled,
            memory_data_epoch,
            episode_guard,
            usage_sink,
        ):
            arranged.events.append("graph_started")
            assert user_id == str(arranged.user.id)
            assert memory_enabled is False
            assert memory_data_epoch == 4
            assert callable(episode_guard)
            usage_sink["total"] = 7
            yield "hello\n"
            yield "world"
            arranged.events.append("graph_completed")

        monkeypatch.setattr(
            chat_api.conversation_repo,
            "get_owned_for_update",
            lock_conversation,
        )
        monkeypatch.setattr(
            chat_api.chat_service,
            "stream_chat_session",
            stream_graph,
        )

        response = await chat_api.session_chat(
            arranged.request,
            arranged.body,
            arranged.user,
            arranged.request_db,
        )
        frames = []
        async for frame in response.body_iterator:
            payload = _sse_payload(frame)
            frames.append(payload)
            if payload == "[DONE]":
                arranged.events.append("done_observed")

        assert frames == [
            {"delta": "hello\n"},
            {"delta": "world"},
            "[DONE]",
        ]
        assert arranged.events.index("conversation_lock:initial") < arranged.events.index(
            "message_create:user"
        )
        assert arranged.events.index("message_create:user") < arranged.events.index(
            "request_commit"
        )
        assert arranged.events.index("request_commit") < arranged.events.index(
            "conversation_lock:generation"
        )
        assert arranged.events.index("conversation_lock:generation") < arranged.events.index(
            "graph_started"
        )
        assert arranged.events.index("graph_completed") < arranged.events.index(
            "message_create:assistant"
        )
        assert arranged.events.index("message_create:assistant") < arranged.events.index(
            "transaction_exit:commit"
        )
        assert arranged.events.index("transaction_exit:commit") < arranged.events.index(
            "done_observed"
        )

    asyncio.run(scenario())


def test_nonstream_rereads_latest_consent_and_passes_run_context(monkeypatch):
    async def scenario():
        arranged = _arrange_existing_conversation(monkeypatch, stream=False)
        consent_read = AsyncMock(
            return_value=chat_api.user_repo.MemoryAccessSnapshot(
                enabled=False,
                consent_version=5,
                data_epoch=6,
            )
        )
        monkeypatch.setattr(
            chat_api.user_repo,
            "get_memory_access_snapshot",
            consent_read,
        )

        async def run_graph(
            agent,
            message,
            thread_id,
            *,
            user_id,
            memory_enabled,
            memory_data_epoch,
            episode_guard,
            usage_sink,
        ):
            arranged.events.append("graph_started")
            assert thread_id == str(arranged.conversation.id)
            assert user_id == str(arranged.user.id)
            assert memory_enabled is False
            assert memory_data_epoch == 6
            assert callable(episode_guard)
            assert arranged.user.memory_enabled is True
            return "complete reply"

        monkeypatch.setattr(chat_api.chat_service, "run_chat_session", run_graph)

        response = await chat_api.session_chat(
            arranged.request,
            arranged.body,
            arranged.user,
            arranged.request_db,
        )

        assert response["message"]["content"] == "complete reply"
        consent_read.assert_awaited_once()
        read_session, read_user_id = consent_read.await_args.args
        assert read_session is arranged.request.app.state.db_session()
        assert read_user_id == arranged.user.id
        assert arranged.events.index("conversation_lock:generation") < arranged.events.index(
            "graph_started"
        )
        assert arranged.events.index("graph_started") < arranged.events.index(
            "message_create:assistant"
        )

    asyncio.run(scenario())


def test_failed_partial_stream_is_not_persisted(monkeypatch):
    async def scenario():
        arranged = _arrange_existing_conversation(monkeypatch, stream=True)
        monkeypatch.setattr(
            chat_api.conversation_repo,
            "get_owned_for_update",
            AsyncMock(return_value=arranged.conversation),
        )

        async def failing_stream(*args, **kwargs):
            yield "partial"
            raise RuntimeError("model unavailable")

        monkeypatch.setattr(
            chat_api.chat_service,
            "stream_chat_session",
            failing_stream,
        )

        response = await chat_api.session_chat(
            arranged.request,
            arranged.body,
            arranged.user,
            arranged.request_db,
        )
        frames = [_sse_payload(frame) async for frame in response.body_iterator]

        assert frames == [
            {"delta": "partial"},
            {"error": "Generation failed."},
            "[DONE]",
        ]
        assert arranged.events.count("message_create:user") == 1
        assert "message_create:assistant" not in arranged.events
        assert "transaction_exit:rollback:RuntimeError" in arranged.events

    asyncio.run(scenario())


@pytest.mark.parametrize("failure_stage", ["assistant_save", "transaction_commit"])
def test_stream_persistence_failure_emits_error_and_done(monkeypatch, failure_stage):
    async def scenario():
        arranged = _arrange_existing_conversation(monkeypatch, stream=True)
        monkeypatch.setattr(
            chat_api.conversation_repo,
            "get_owned_for_update",
            AsyncMock(return_value=arranged.conversation),
        )

        async def stream_graph(*args, **kwargs):
            yield "generated reply"

        monkeypatch.setattr(
            chat_api.chat_service,
            "stream_chat_session",
            stream_graph,
        )

        if failure_stage == "assistant_save":
            async def fail_assistant_save(db, **kwargs):
                arranged.events.append(f"message_create:{kwargs['role']}")
                if kwargs["role"] == "assistant":
                    raise RuntimeError("assistant save failed")
                return SimpleNamespace()

            monkeypatch.setattr(
                chat_api.message_repo,
                "create",
                fail_assistant_save,
            )
        else:
            arranged.request.app.state.db_session().transaction_exit_error = (
                RuntimeError("commit failed")
            )

        response = await chat_api.session_chat(
            arranged.request,
            arranged.body,
            arranged.user,
            arranged.request_db,
        )
        frames = [_sse_payload(frame) async for frame in response.body_iterator]

        assert frames == [
            {"delta": "generated reply"},
            {"error": "Generation failed."},
            "[DONE]",
        ]
        assert "message_create:assistant" in arranged.events
        if failure_stage == "assistant_save":
            assert "transaction_exit:rollback:RuntimeError" in arranged.events
        else:
            assert "transaction_exit:commit" in arranged.events

    asyncio.run(scenario())


def test_title_update_failure_does_not_change_successful_stream(monkeypatch):
    async def scenario():
        arranged = _arrange_existing_conversation(monkeypatch, stream=True)
        arranged.body.conversation_id = None
        monkeypatch.setattr(
            chat_api.conversation_service,
            "create_conversation",
            AsyncMock(return_value=arranged.conversation),
        )
        monkeypatch.setattr(
            chat_api.conversation_repo,
            "get_owned_for_update",
            AsyncMock(return_value=arranged.conversation),
        )
        monkeypatch.setattr(
            chat_api.conversation_service,
            "generate_title",
            AsyncMock(return_value="A title"),
        )
        monkeypatch.setattr(
            chat_api.conversation_repo,
            "update_title",
            AsyncMock(side_effect=RuntimeError("title write failed")),
        )

        async def stream_graph(*args, **kwargs):
            yield "complete reply"

        monkeypatch.setattr(
            chat_api.chat_service,
            "stream_chat_session",
            stream_graph,
        )

        response = await chat_api.session_chat(
            arranged.request,
            arranged.body,
            arranged.user,
            arranged.request_db,
        )
        frames = [_sse_payload(frame) async for frame in response.body_iterator]
        if chat_api._BACKGROUND_TASKS:
            await asyncio.gather(*tuple(chat_api._BACKGROUND_TASKS))

        assert frames == [{"delta": "complete reply"}, "[DONE]"]
        assert "message_create:assistant" in arranged.events

    asyncio.run(scenario())


def test_title_update_does_not_delay_done(monkeypatch):
    async def scenario():
        arranged = _arrange_existing_conversation(monkeypatch, stream=True)
        arranged.body.conversation_id = None
        monkeypatch.setattr(
            chat_api.conversation_service,
            "create_conversation",
            AsyncMock(return_value=arranged.conversation),
        )
        monkeypatch.setattr(
            chat_api.conversation_repo,
            "get_owned_for_update",
            AsyncMock(return_value=arranged.conversation),
        )

        title_started = asyncio.Event()
        release_title = asyncio.Event()

        async def blocked_title_update(*args, **kwargs):
            title_started.set()
            await release_title.wait()

        async def stream_graph(*args, **kwargs):
            yield "complete reply"

        monkeypatch.setattr(
            chat_api,
            "_save_generated_title_if_present",
            blocked_title_update,
        )
        monkeypatch.setattr(
            chat_api.chat_service,
            "stream_chat_session",
            stream_graph,
        )

        response = await chat_api.session_chat(
            arranged.request,
            arranged.body,
            arranged.user,
            arranged.request_db,
        )

        async def collect_frames():
            return [_sse_payload(frame) async for frame in response.body_iterator]

        frames = await asyncio.wait_for(collect_frames(), timeout=0.5)
        assert frames == [{"delta": "complete reply"}, "[DONE]"]
        await asyncio.sleep(0)
        assert title_started.is_set()

        release_title.set()
        if chat_api._BACKGROUND_TASKS:
            await asyncio.gather(*tuple(chat_api._BACKGROUND_TASKS))

    asyncio.run(scenario())


def test_client_disconnect_exits_generation_transaction(monkeypatch):
    async def scenario():
        arranged = _arrange_existing_conversation(monkeypatch, stream=True)
        monkeypatch.setattr(
            chat_api.conversation_repo,
            "get_owned_for_update",
            AsyncMock(return_value=arranged.conversation),
        )

        async def endless_stream(*args, **kwargs):
            yield "first"
            yield "second"

        monkeypatch.setattr(
            chat_api.chat_service,
            "stream_chat_session",
            endless_stream,
        )

        response = await chat_api.session_chat(
            arranged.request,
            arranged.body,
            arranged.user,
            arranged.request_db,
        )
        first_frame = await anext(response.body_iterator)
        await response.body_iterator.aclose()

        assert _sse_payload(first_frame) == {"delta": "first"}
        assert "transaction_exit:rollback:GeneratorExit" in arranged.events
        assert "session_exit" in arranged.events
        assert "message_create:assistant" not in arranged.events

    asyncio.run(scenario())
