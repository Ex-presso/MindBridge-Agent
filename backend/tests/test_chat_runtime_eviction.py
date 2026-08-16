import asyncio
import uuid
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock

import pytest

from app.api.v1 import chat as chat_api


class _Transaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> bool:
        return False


class _Session:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> bool:
        return False

    def begin(self) -> _Transaction:
        return _Transaction()


def test_evict_user_runtime_only_drops_and_drains_target_user(monkeypatch):
    async def scenario() -> None:
        original_cache = OrderedDict(chat_api._AGENT_CACHE)
        original_tasks = set(chat_api._BACKGROUND_TASKS)
        original_task_users = dict(chat_api._BACKGROUND_TASK_USERS)
        created_tasks: list[asyncio.Task[None]] = []

        target_user_id = uuid.uuid4()
        other_user_id = uuid.uuid4()
        started = {
            str(target_user_id): asyncio.Event(),
            str(other_user_id): asyncio.Event(),
        }
        released = {
            str(target_user_id): asyncio.Event(),
            str(other_user_id): asyncio.Event(),
        }
        finalized = {
            str(target_user_id): asyncio.Event(),
            str(other_user_id): asyncio.Event(),
        }

        async def blocked_title_update(
            request,
            conversation_id,
            user_id,
            first_user_message,
            provider,
            model,
        ) -> None:
            owner_id = str(user_id)
            started[owner_id].set()
            try:
                await released[owner_id].wait()
            finally:
                finalized[owner_id].set()

        monkeypatch.setattr(
            chat_api,
            "_save_generated_title_if_present",
            blocked_title_update,
        )

        target_agent_one = object()
        target_agent_two = object()
        other_agent = object()

        try:
            chat_api._AGENT_CACHE.clear()
            chat_api._BACKGROUND_TASKS.clear()
            chat_api._BACKGROUND_TASK_USERS.clear()
            chat_api._AGENT_CACHE.update(
                {
                    (str(target_user_id), "openai"): target_agent_one,
                    (str(other_user_id), "anthropic"): other_agent,
                    (str(target_user_id), "gemini"): target_agent_two,
                }
            )

            chat_api._schedule_title_update(
                object(),
                uuid.uuid4(),
                target_user_id,
                "target message",
                "openai",
                "test-model",
            )
            chat_api._schedule_title_update(
                object(),
                uuid.uuid4(),
                other_user_id,
                "other message",
                "anthropic",
                "test-model",
            )
            created_tasks.extend(chat_api._BACKGROUND_TASKS)

            await asyncio.wait_for(
                asyncio.gather(
                    started[str(target_user_id)].wait(),
                    started[str(other_user_id)].wait(),
                ),
                timeout=0.5,
            )
            target_task = next(
                task
                for task, owner_id in chat_api._BACKGROUND_TASK_USERS.items()
                if owner_id == str(target_user_id)
            )
            other_task = next(
                task
                for task, owner_id in chat_api._BACKGROUND_TASK_USERS.items()
                if owner_id == str(other_user_id)
            )

            await chat_api.evict_user_runtime(target_user_id)

            assert list(chat_api._AGENT_CACHE.items()) == [
                ((str(other_user_id), "anthropic"), other_agent)
            ]
            assert target_task.done()
            assert target_task.cancelled()
            assert finalized[str(target_user_id)].is_set()
            assert target_task not in chat_api._BACKGROUND_TASKS
            assert target_task not in chat_api._BACKGROUND_TASK_USERS

            assert not other_task.done()
            assert other_task in chat_api._BACKGROUND_TASKS
            assert chat_api._BACKGROUND_TASK_USERS[other_task] == str(other_user_id)
            assert not finalized[str(other_user_id)].is_set()

            released[str(other_user_id)].set()
            await asyncio.wait_for(other_task, timeout=0.5)
            await asyncio.sleep(0)
            assert finalized[str(other_user_id)].is_set()
            assert other_task not in chat_api._BACKGROUND_TASKS
            assert other_task not in chat_api._BACKGROUND_TASK_USERS
        finally:
            for task in created_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*created_tasks, return_exceptions=True)
            chat_api._AGENT_CACHE.clear()
            chat_api._AGENT_CACHE.update(original_cache)
            chat_api._BACKGROUND_TASKS.clear()
            chat_api._BACKGROUND_TASKS.update(original_tasks)
            chat_api._BACKGROUND_TASK_USERS.clear()
            chat_api._BACKGROUND_TASK_USERS.update(original_task_users)

    asyncio.run(scenario())


def test_evict_user_runtime_is_bounded_when_task_suppresses_cancellation(
    monkeypatch,
):
    async def scenario() -> None:
        original_tasks = set(chat_api._BACKGROUND_TASKS)
        original_task_users = dict(chat_api._BACKGROUND_TASK_USERS)
        user_id = uuid.uuid4()
        started = asyncio.Event()
        release = asyncio.Event()
        cancellation_seen = asyncio.Event()

        async def stubborn_title() -> None:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellation_seen.set()
                await release.wait()

        task = asyncio.create_task(stubborn_title())
        chat_api._BACKGROUND_TASKS.add(task)
        chat_api._BACKGROUND_TASK_USERS[task] = str(user_id)
        monkeypatch.setattr(
            chat_api,
            "_RUNTIME_EVICTION_TIMEOUT_SECONDS",
            0.001,
        )

        try:
            await asyncio.wait_for(started.wait(), timeout=0.1)
            await asyncio.wait_for(
                chat_api.evict_user_runtime(user_id),
                timeout=0.1,
            )
            assert cancellation_seen.is_set()
            assert not task.done()
        finally:
            release.set()
            await asyncio.wait_for(task, timeout=0.1)
            chat_api._BACKGROUND_TASKS.discard(task)
            chat_api._BACKGROUND_TASK_USERS.pop(task, None)
            chat_api._BACKGROUND_TASKS.clear()
            chat_api._BACKGROUND_TASKS.update(original_tasks)
            chat_api._BACKGROUND_TASK_USERS.clear()
            chat_api._BACKGROUND_TASK_USERS.update(original_task_users)

    asyncio.run(scenario())


def test_real_agent_cache_key_is_user_scoped_and_target_evictable(monkeypatch):
    async def scenario() -> None:
        original_cache = OrderedDict(chat_api._AGENT_CACHE)
        user_one = uuid.uuid4()
        user_two = uuid.uuid4()
        created = []

        class StubAgent:
            def __init__(self, llm, *, checkpointer, store):
                self.llm = llm
                self.checkpointer = checkpointer
                self.store = store
                created.append(self)

        monkeypatch.setattr(
            "app.core.llm.provider.get_llm",
            lambda provider, **kwargs: (provider, kwargs),
        )
        monkeypatch.setattr("app.core.agent.agent.Agent", StubAgent)
        checkpointer = object()
        store = object()

        try:
            chat_api._AGENT_CACHE.clear()
            first = chat_api._get_agent(
                "openai",
                "test-model",
                None,
                "same-key",
                checkpointer,
                store,
                user_id=str(user_one),
            )
            first_again = chat_api._get_agent(
                "openai",
                "test-model",
                None,
                "same-key",
                checkpointer,
                store,
                user_id=str(user_one),
            )
            second_user = chat_api._get_agent(
                "openai",
                "test-model",
                None,
                "same-key",
                checkpointer,
                store,
                user_id=str(user_two),
            )

            assert first_again is first
            assert second_user is not first
            assert len(created) == 2

            await chat_api.evict_user_runtime(user_one)

            assert all(
                cache_key[0] == str(user_two)
                for cache_key in chat_api._AGENT_CACHE
            )
            assert list(chat_api._AGENT_CACHE.values()) == [second_user]
        finally:
            chat_api._AGENT_CACHE.clear()
            chat_api._AGENT_CACHE.update(original_cache)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "access",
    [
        None,
        SimpleNamespace(account_deletion_pending=True),
    ],
    ids=["deleted", "deletion-pending"],
)
def test_title_generation_stops_when_account_is_unavailable(monkeypatch, access):
    async def scenario() -> None:
        user_id = uuid.uuid4()
        get_access = AsyncMock(return_value=access)
        get_key = AsyncMock()
        lock_conversation = AsyncMock()
        generate_title = AsyncMock(return_value="Generated title")
        update_title = AsyncMock()
        monkeypatch.setattr(
            chat_api.user_repo,
            "get_memory_access_for_chat",
            get_access,
        )
        monkeypatch.setattr(
            chat_api.api_key_repo,
            "get_by_provider",
            get_key,
        )
        monkeypatch.setattr(
            chat_api.conversation_repo,
            "get_owned_for_update",
            lock_conversation,
        )
        monkeypatch.setattr(
            chat_api.conversation_service,
            "generate_title",
            generate_title,
        )
        monkeypatch.setattr(
            chat_api.conversation_repo,
            "update_title",
            update_title,
        )
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(db_session=lambda: _Session())
            )
        )

        await chat_api._save_generated_title_if_present(
            request,
            uuid.uuid4(),
            user_id,
            "first message",
            "openai",
            "test-model",
        )

        get_access.assert_awaited_once_with(ANY, user_id)
        get_key.assert_not_awaited()
        lock_conversation.assert_not_awaited()
        generate_title.assert_not_awaited()
        update_title.assert_not_awaited()

    asyncio.run(scenario())


def test_title_generation_stops_when_current_api_key_is_missing(monkeypatch):
    async def scenario() -> None:
        user_id = uuid.uuid4()
        monkeypatch.setattr(
            chat_api.user_repo,
            "get_memory_access_for_chat",
            AsyncMock(
                return_value=SimpleNamespace(account_deletion_pending=False)
            ),
        )
        monkeypatch.setattr(
            chat_api.api_key_repo,
            "get_by_provider",
            AsyncMock(return_value=None),
        )
        lock_conversation = AsyncMock()
        generate_title = AsyncMock()
        monkeypatch.setattr(
            chat_api.conversation_repo,
            "get_owned_for_update",
            lock_conversation,
        )
        monkeypatch.setattr(
            chat_api.conversation_service,
            "generate_title",
            generate_title,
        )
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(db_session=lambda: _Session())
            )
        )

        await chat_api._save_generated_title_if_present(
            request,
            uuid.uuid4(),
            user_id,
            "first message",
            "openai",
            "test-model",
        )

        lock_conversation.assert_not_awaited()
        generate_title.assert_not_awaited()

    asyncio.run(scenario())


def test_title_generation_reloads_current_api_key_under_user_barrier(monkeypatch):
    async def scenario() -> None:
        user_id = uuid.uuid4()
        conversation_id = uuid.uuid4()
        current_key = SimpleNamespace(
            api_key_encrypted="current-ciphertext",
            base_url="https://current.example/v1",
        )
        monkeypatch.setattr(
            chat_api.user_repo,
            "get_memory_access_for_chat",
            AsyncMock(
                return_value=SimpleNamespace(account_deletion_pending=False)
            ),
        )
        monkeypatch.setattr(
            chat_api.api_key_repo,
            "get_by_provider",
            AsyncMock(return_value=current_key),
        )
        monkeypatch.setattr(
            chat_api.conversation_repo,
            "get_owned_for_update",
            AsyncMock(return_value=SimpleNamespace(id=conversation_id)),
        )
        monkeypatch.setattr(
            "app.core.auth.encryption.decrypt_value",
            lambda ciphertext: f"decrypted:{ciphertext}",
        )
        agent_calls = []

        def get_agent(*args, **kwargs):
            agent_calls.append((args, kwargs))
            return object()

        monkeypatch.setattr(chat_api, "_get_agent", get_agent)
        generate_title = AsyncMock(return_value="Current title")
        update_title = AsyncMock()
        monkeypatch.setattr(
            chat_api.conversation_service,
            "generate_title",
            generate_title,
        )
        monkeypatch.setattr(
            chat_api.conversation_repo,
            "update_title",
            update_title,
        )
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    db_session=lambda: _Session(),
                    checkpointer=object(),
                    store=object(),
                )
            )
        )

        await chat_api._save_generated_title_if_present(
            request,
            conversation_id,
            user_id,
            "first message",
            "openai",
            "test-model",
        )

        assert len(agent_calls) == 1
        args, kwargs = agent_calls[0]
        assert args[:4] == (
            "openai",
            "test-model",
            "https://current.example/v1",
            "decrypted:current-ciphertext",
        )
        assert kwargs["user_id"] == str(user_id)
        generate_title.assert_awaited_once()
        update_title.assert_awaited_once_with(
            ANY,
            conversation_id,
            "Current title",
        )

    asyncio.run(scenario())
