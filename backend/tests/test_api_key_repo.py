"""Atomic API-key upsert and encryption-boundary regressions."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy.dialects import postgresql

from app.api.v1 import api_keys as api_keys_api
from app.api.v1 import chat as chat_api
from app.core.auth.encryption import decrypt_value, encrypt_value
from app.db.repositories import api_key_repo
from app.schemas.api_key import ApiKeySaveRequest


class _ScalarResult:
    def __init__(self, value):
        self.value = value
        self.scalar_one_calls = 0

    def scalar_one(self):
        self.scalar_one_calls += 1
        return self.value


class _RecordingDb:
    def __init__(self, result):
        self.result = result
        self.statements = []

    async def execute(self, statement):
        self.statements.append(statement)
        return self.result


def test_upsert_infers_unique_columns_returning_and_keeps_plaintext_out():
    user_id = uuid.uuid4()
    plaintext = "sk-live-super-secret"
    ciphertext = encrypt_value(plaintext)
    record = SimpleNamespace(id=uuid.uuid4(), api_key_encrypted=ciphertext)
    result = _ScalarResult(record)
    db = _RecordingDb(result)

    returned = asyncio.run(
        api_key_repo.upsert(
            db,
            user_id=user_id,
            provider="openai",
            api_key_encrypted=ciphertext,
            model_id="gpt-4.1",
            display_name="Primary",
        )
    )

    assert returned is record
    assert result.scalar_one_calls == 1
    assert len(db.statements) == 1

    statement = db.statements[0]
    compiled = statement.compile(dialect=postgresql.dialect())
    sql = " ".join(str(compiled).split())
    assert "ON CONFLICT (user_id, provider) DO UPDATE SET" in sql
    assert "ON CONFLICT ON CONSTRAINT" not in sql
    assert "RETURNING user_api_keys.id" in sql
    assert statement.get_execution_options()["populate_existing"] is True

    assert compiled.params["user_id"] == user_id
    assert compiled.params["api_key_encrypted"] == ciphertext
    assert compiled.params["param_1"] == ciphertext
    assert decrypt_value(ciphertext) == plaintext
    assert all(plaintext not in str(value) for value in compiled.params.values())


def test_save_api_key_encrypts_before_calling_repository(monkeypatch):
    user_id = uuid.uuid4()
    plaintext = "sk-api-plain-secret"
    ciphertext = "encrypted-at-api-boundary"
    record = SimpleNamespace(
        provider="openai",
        base_url=None,
        model_id="gpt-4.1",
        display_name="Primary",
    )
    upsert = AsyncMock(return_value=record)
    lock_owner = AsyncMock()
    evict_runtime = AsyncMock()
    db = AsyncMock()
    monkeypatch.setattr(api_keys_api, "encrypt_value", lambda value: ciphertext)
    monkeypatch.setattr(api_keys_api.api_key_repo, "upsert", upsert)
    monkeypatch.setattr(
        api_keys_api,
        "_lock_active_credential_owner",
        lock_owner,
    )
    monkeypatch.setattr(chat_api, "evict_user_runtime", evict_runtime)

    response = asyncio.run(
        api_keys_api.save_api_key(
            body=ApiKeySaveRequest(
                provider="openai",
                api_key=plaintext,
                model_id="gpt-4.1",
                display_name="Primary",
            ),
            user=SimpleNamespace(id=user_id),
            db=db,
        )
    )

    assert response.provider == "openai"
    assert response.api_key_masked.endswith(plaintext[-4:])
    upsert.assert_awaited_once()
    kwargs = upsert.await_args.kwargs
    assert kwargs["user_id"] == user_id
    assert kwargs["api_key_encrypted"] == ciphertext
    assert plaintext not in kwargs.values()
    lock_owner.assert_awaited_once_with(db, user_id)
    db.commit.assert_awaited_once()
    assert evict_runtime.await_count == 2


def test_delete_api_key_serializes_and_purges_runtime_before_return(monkeypatch):
    async def scenario():
        events: list[str] = []
        user_id = uuid.uuid4()
        db = AsyncMock()

        async def evict(user):
            assert user == user_id
            events.append("evict")

        async def lock_owner(locked_db, locked_user):
            assert locked_db is db
            assert locked_user == user_id
            events.append("lock_user")

        async def delete_key(locked_db, locked_user, provider):
            assert locked_db is db
            assert locked_user == user_id
            assert provider == "openai"
            events.append("delete_key")
            return True

        async def commit():
            events.append("commit")

        monkeypatch.setattr(chat_api, "evict_user_runtime", evict)
        monkeypatch.setattr(
            api_keys_api,
            "_lock_active_credential_owner",
            lock_owner,
        )
        monkeypatch.setattr(
            api_keys_api.api_key_repo,
            "delete_by_provider",
            delete_key,
        )
        db.commit.side_effect = commit

        await api_keys_api.delete_api_key(
            provider="openai",
            user=SimpleNamespace(id=user_id),
            db=db,
        )

        assert events == [
            "evict",
            "lock_user",
            "delete_key",
            "commit",
            "evict",
        ]

    asyncio.run(scenario())


def test_credential_mutation_rejects_account_deletion_pending(monkeypatch):
    user_id = uuid.uuid4()
    db = object()
    monkeypatch.setattr(
        api_keys_api.user_repo,
        "get_memory_access_for_update",
        AsyncMock(
            return_value=SimpleNamespace(account_deletion_pending=True)
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(api_keys_api._lock_active_credential_owner(db, user_id))

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Account deletion is in progress."


def test_credential_mutation_lock_wait_has_a_retryable_deadline(monkeypatch):
    async def scenario():
        user_id = uuid.uuid4()
        db = AsyncMock()

        async def never_acquires_lock(*args, **kwargs):
            await asyncio.Event().wait()

        monkeypatch.setattr(
            api_keys_api.user_repo,
            "get_memory_access_for_update",
            never_acquires_lock,
        )
        monkeypatch.setattr(
            api_keys_api,
            "_CREDENTIAL_LOCK_TIMEOUT_SECONDS",
            0.001,
        )

        with pytest.raises(HTTPException) as exc_info:
            await api_keys_api._lock_active_credential_owner(db, user_id)

        assert exc_info.value.status_code == 503
        db.rollback.assert_awaited_once()

    asyncio.run(scenario())
