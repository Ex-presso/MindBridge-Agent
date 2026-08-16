"""Account deletion keeps external cleanup retryable and fail closed."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
import uuid

import pytest
from fastapi import HTTPException, Response

from app.api.v1 import auth as auth_api
from app.api.v1 import chat as chat_api
from app.schemas.auth import DeleteAccountRequest
from app.services import account_service


def test_delete_account_uses_user_then_conversation_lock_order(monkeypatch):
    async def scenario():
        events = []
        user_id = uuid.uuid4()
        conversations = [
            SimpleNamespace(id=uuid.UUID(int=1)),
            SimpleNamespace(id=uuid.UUID(int=2)),
        ]
        db = AsyncMock()
        store = object()
        checkpointer = AsyncMock()

        async def invalidate(_db, _user_id):
            events.append("invalidate_commit")

        async def lock_user(_db, _user_id):
            events.append("lock_user")

        async def lock_conversations(_db, _user_id):
            events.append("lock_conversations")
            return conversations

        async def cancel_jobs(_db, _user_id):
            events.append("cancel_jobs")
            return 0

        async def clear_store(_store, _user_id):
            events.append("clear_store")
            return 3

        async def delete_thread(thread_id):
            events.append(f"delete_checkpoint:{thread_id}")

        async def delete_user(_db, _user_id):
            events.append("delete_user")
            return True

        async def commit():
            events.append("relational_commit")

        monkeypatch.setattr(
            account_service.memory_service,
            "begin_account_deletion",
            invalidate,
        )
        monkeypatch.setattr(
            account_service.memory_service,
            "lock_account_deletion",
            lock_user,
        )
        monkeypatch.setattr(
            account_service.conversation_repo,
            "list_owned_for_update",
            lock_conversations,
        )
        monkeypatch.setattr(
            account_service.memory_job_repo,
            "cancel_unfinished_for_user",
            cancel_jobs,
        )
        monkeypatch.setattr(
            account_service.memory_service,
            "delete_user_memory_items",
            clear_store,
        )
        monkeypatch.setattr(
            account_service.user_repo,
            "delete_by_id",
            delete_user,
        )
        checkpointer.adelete_thread.side_effect = delete_thread
        db.commit.side_effect = commit

        deleted_conversations = await account_service.delete_account(
            db,
            user_id,
            store=store,
            checkpointer=checkpointer,
        )

        assert deleted_conversations == 2
        assert events == [
            "invalidate_commit",
            "lock_user",
            "lock_conversations",
            "cancel_jobs",
            "clear_store",
            f"delete_checkpoint:{conversations[0].id}",
            f"delete_checkpoint:{conversations[1].id}",
            "delete_user",
            "relational_commit",
        ]
        db.rollback.assert_not_awaited()

    asyncio.run(scenario())


def test_checkpoint_failure_keeps_relational_user_for_retry(monkeypatch):
    async def scenario():
        user_id = uuid.uuid4()
        db = AsyncMock()
        checkpointer = AsyncMock()
        checkpointer.adelete_thread.side_effect = RuntimeError("checkpoint down")
        delete_user = AsyncMock(return_value=True)

        monkeypatch.setattr(
            account_service.memory_service,
            "begin_account_deletion",
            AsyncMock(),
        )
        monkeypatch.setattr(
            account_service.memory_service,
            "lock_account_deletion",
            AsyncMock(),
        )
        monkeypatch.setattr(
            account_service.conversation_repo,
            "list_owned_for_update",
            AsyncMock(return_value=[SimpleNamespace(id=uuid.uuid4())]),
        )
        monkeypatch.setattr(
            account_service.memory_job_repo,
            "cancel_unfinished_for_user",
            AsyncMock(return_value=0),
        )
        monkeypatch.setattr(
            account_service.memory_service,
            "delete_user_memory_items",
            AsyncMock(return_value=1),
        )
        monkeypatch.setattr(account_service.user_repo, "delete_by_id", delete_user)

        with pytest.raises(account_service.AccountDeletionError, match="checkpoint"):
            await account_service.delete_account(
                db,
                user_id,
                store=object(),
                checkpointer=checkpointer,
            )

        delete_user.assert_not_awaited()
        db.rollback.assert_awaited_once()
        db.commit.assert_not_awaited()

    asyncio.run(scenario())


def test_missing_external_runtime_fails_after_durable_invalidation(monkeypatch):
    async def scenario():
        user_id = uuid.uuid4()
        db = AsyncMock()
        invalidate = AsyncMock()
        delete_user = AsyncMock()
        monkeypatch.setattr(
            account_service.memory_service,
            "begin_account_deletion",
            invalidate,
        )
        monkeypatch.setattr(
            account_service.memory_service,
            "lock_account_deletion",
            AsyncMock(),
        )
        monkeypatch.setattr(
            account_service.conversation_repo,
            "list_owned_for_update",
            AsyncMock(return_value=[]),
        )
        monkeypatch.setattr(
            account_service.memory_job_repo,
            "cancel_unfinished_for_user",
            AsyncMock(return_value=0),
        )
        monkeypatch.setattr(account_service.user_repo, "delete_by_id", delete_user)

        with pytest.raises(account_service.AccountDeletionError, match="unavailable"):
            await account_service.delete_account(
                db,
                user_id,
                store=None,
                checkpointer=None,
            )

        invalidate.assert_awaited_once_with(db, user_id)
        delete_user.assert_not_awaited()
        db.rollback.assert_awaited_once()

    asyncio.run(scenario())


def test_delete_account_api_requires_password_and_clears_cookie(monkeypatch):
    async def scenario():
        user = SimpleNamespace(
            id=uuid.uuid4(),
            hashed_password="hash",
        )
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(store=object(), checkpointer=object())
            )
        )
        db = AsyncMock()
        delete_account = AsyncMock(return_value=0)
        evict_runtime = AsyncMock()
        monkeypatch.setattr(auth_api, "verify_password", lambda *_: True)
        monkeypatch.setattr(
            auth_api.account_service,
            "delete_account",
            delete_account,
        )
        monkeypatch.setattr(chat_api, "evict_user_runtime", evict_runtime)
        response = Response()

        endpoint = auth_api.delete_me.__wrapped__
        await endpoint(
            request=request,
            body=DeleteAccountRequest(password="correct password"),
            response=response,
            user=user,
            db=db,
        )

        assert response.status_code == 204
        assert "refresh_token=" in response.headers["set-cookie"]
        delete_account.assert_awaited_once()
        assert evict_runtime.await_count == 2
        assert all(
            call.args == (user.id,)
            for call in evict_runtime.await_args_list
        )

        monkeypatch.setattr(auth_api, "verify_password", lambda *_: False)
        with pytest.raises(HTTPException) as exc_info:
            await endpoint(
                request=request,
                body=DeleteAccountRequest(password="wrong password"),
                response=Response(),
                user=user,
                db=db,
            )
        assert exc_info.value.status_code == 403
        assert evict_runtime.await_count == 2

    asyncio.run(scenario())


def test_delete_account_api_purges_runtime_again_after_retryable_failure(
    monkeypatch,
):
    async def scenario():
        user = SimpleNamespace(id=uuid.uuid4(), hashed_password="hash")
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(store=object(), checkpointer=object())
            )
        )
        evict_runtime = AsyncMock()
        monkeypatch.setattr(auth_api, "verify_password", lambda *_: True)
        monkeypatch.setattr(chat_api, "evict_user_runtime", evict_runtime)
        monkeypatch.setattr(
            auth_api.account_service,
            "delete_account",
            AsyncMock(
                side_effect=account_service.AccountDeletionError(
                    "checkpoint unavailable"
                )
            ),
        )

        with pytest.raises(HTTPException) as exc_info:
            await auth_api.delete_me.__wrapped__(
                request=request,
                body=DeleteAccountRequest(password="correct password"),
                response=Response(),
                user=user,
                db=AsyncMock(),
            )

        assert exc_info.value.status_code == 503
        assert evict_runtime.await_count == 2
        assert all(
            call.args == (user.id,)
            for call in evict_runtime.await_args_list
        )

    asyncio.run(scenario())


def test_delete_account_api_times_out_safely_and_remains_retryable(monkeypatch):
    async def scenario():
        user = SimpleNamespace(id=uuid.uuid4(), hashed_password="hash")
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(store=object(), checkpointer=object())
            )
        )
        db = AsyncMock()
        evict_runtime = AsyncMock()

        async def never_finishes(*args, **kwargs):
            await asyncio.Event().wait()

        monkeypatch.setattr(auth_api, "verify_password", lambda *_: True)
        monkeypatch.setattr(chat_api, "evict_user_runtime", evict_runtime)
        monkeypatch.setattr(
            auth_api.account_service,
            "delete_account",
            never_finishes,
        )
        monkeypatch.setattr(
            auth_api,
            "_ACCOUNT_DELETION_TIMEOUT_SECONDS",
            0.001,
        )

        with pytest.raises(HTTPException) as exc_info:
            await auth_api.delete_me.__wrapped__(
                request=request,
                body=DeleteAccountRequest(password="correct password"),
                response=Response(),
                user=user,
                db=db,
            )

        assert exc_info.value.status_code == 503
        db.rollback.assert_awaited_once()
        assert evict_runtime.await_count == 2

    asyncio.run(scenario())
