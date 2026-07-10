"""Durable memory privacy service invariants."""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy.dialects import postgresql

from app.api.v1 import memory as memory_api
from app.services import memory_service


def _item(user_id, category, key):
    now = datetime.now(timezone.utc)
    return SimpleNamespace(
        namespace=("memory", str(user_id), category),
        key=key,
        value={"content": key},
        created_at=now,
        updated_at=now,
        score=None,
    )


class _Result:
    def __init__(self, value=None, *, rowcount=0):
        self._value = value
        self.rowcount = rowcount

    def one_or_none(self):
        return self._value


class _Db:
    def __init__(self, user, events=None):
        self.user = user
        if user is not None:
            if not hasattr(user, "memory_consent_version"):
                user.memory_consent_version = 0
            if not hasattr(user, "memory_data_epoch"):
                user.memory_data_epoch = 0
        self.events = events if events is not None else []
        self.execute_count = 0
        self.commit_count = 0
        self.rollback_count = 0
        self.statements = []

    async def execute(self, statement):
        self.execute_count += 1
        self.statements.append(statement)
        if statement.table.name == "memory_jobs":
            self.events.append("jobs_cancel")
            return _Result(rowcount=0)
        self.events.append("consent_write")
        if self.user is None:
            return _Result(None)
        compiled = statement.compile()
        enabled = compiled.params["memory_enabled"]
        if self.user.memory_enabled != enabled:
            self.user.memory_consent_version += 1
        self.user.memory_enabled = enabled
        updated_columns = {
            getattr(column, "key", str(column)) for column in statement._values
        }
        if "memory_data_epoch" in updated_columns:
            self.user.memory_data_epoch += 1
        return _Result(
            (
                self.user.memory_enabled,
                self.user.memory_consent_version,
                self.user.memory_data_epoch,
            ),
            rowcount=1,
        )

    async def commit(self):
        self.commit_count += 1
        self.events.append("commit")

    async def rollback(self):
        self.rollback_count += 1
        self.events.append("rollback")


class _Store:
    def __init__(self, items=None, events=None):
        self.items = list(items or [])
        self.events = events if events is not None else []
        self.search_calls = []
        self.batches = []

    async def asearch(self, prefix, *, limit, offset, refresh_ttl):
        self.events.append("search")
        self.search_calls.append((prefix, limit, offset, refresh_ttl))
        matching = [item for item in self.items if item.namespace[:len(prefix)] == prefix]
        return matching[offset:offset + limit]

    async def abatch(self, operations):
        operations = list(operations)
        self.events.append("delete")
        self.batches.append(operations)
        deleted = {(op.namespace, op.key) for op in operations}
        self.items = [
            item for item in self.items
            if (item.namespace, item.key) not in deleted
        ]
        return [None] * len(operations)


def test_memory_namespace_has_exact_user_scoped_shape():
    user_id = uuid.uuid4()

    assert memory_service.memory_namespace(user_id, "semantic") == (
        "memory",
        str(user_id),
        "semantic",
    )
    with pytest.raises(ValueError, match="Unsupported memory category"):
        memory_service.memory_namespace(user_id, "profile")


def test_list_memory_items_is_paginated_and_category_scoped():
    user_id = uuid.uuid4()
    store = _Store(
        [
            _item(user_id, "semantic", "one"),
            _item(user_id, "semantic", "two"),
            _item(user_id, "semantic", "three"),
        ]
    )

    items, has_more = asyncio.run(
        memory_service.list_memory_items(
            store,
            user_id,
            category="semantic",
            limit=2,
            offset=0,
        )
    )

    assert [item.key for item in items] == ["one", "two"]
    assert has_more is True
    assert store.search_calls == [
        (("memory", str(user_id), "semantic"), 3, 0, False)
    ]


def test_list_memory_items_rejects_cross_user_store_results():
    user_id = uuid.uuid4()
    other_user_id = uuid.uuid4()

    class UnsafeStore:
        async def asearch(self, _prefix, *, limit, offset, refresh_ttl):
            return [_item(other_user_id, "semantic", "not-owned")]

    with pytest.raises(memory_service.MemoryStoreError, match="outside"):
        asyncio.run(memory_service.list_memory_items(UnsafeStore(), user_id))


def test_set_memory_consent_uses_forced_update_returning():
    user_id = uuid.uuid4()
    user = SimpleNamespace(id=user_id, memory_enabled=False)
    db = _Db(user)

    enabled = asyncio.run(
        memory_service.set_memory_consent(db, user_id, enabled=True)
    )

    assert enabled is True
    assert user.memory_enabled is True
    assert db.execute_count == 1
    assert db.commit_count == 1
    sql = str(db.statements[0].compile(dialect=postgresql.dialect()))
    assert "UPDATE users SET memory_enabled" in sql
    assert "users.memory_consent_version" in sql
    assert "users.memory_data_epoch" in sql
    assert user.memory_consent_version == 1


def test_clear_memory_disables_before_store_access_and_batch_deletes():
    user_id = uuid.uuid4()
    events = []
    user = SimpleNamespace(id=user_id, memory_enabled=True)
    db = _Db(user, events)
    store = _Store(
        [
            _item(user_id, "semantic", "fact"),
            _item(user_id, "episodes", "episode"),
        ],
        events,
    )

    deleted = asyncio.run(memory_service.clear_memory(db, store, user_id))

    assert deleted == 2
    assert user.memory_enabled is False
    assert events[:5] == [
        "consent_write",
        "jobs_cancel",
        "commit",
        "consent_write",
        "jobs_cancel",
    ]
    assert events[5:] == ["search", "delete", "search", "commit"]
    assert db.execute_count == 4
    assert user.memory_consent_version == 1
    assert user.memory_data_epoch == 1
    assert all(operation.value is None for operation in store.batches[0])
    assert store.search_calls == [
        (("memory", str(user_id)), 100, 0, False),
        (("memory", str(user_id)), 100, 0, False),
    ]


def test_clear_memory_is_idempotent():
    user_id = uuid.uuid4()
    user = SimpleNamespace(id=user_id, memory_enabled=False)
    db = _Db(user)
    store = _Store()

    first = asyncio.run(memory_service.clear_memory(db, store, user_id))
    second = asyncio.run(memory_service.clear_memory(db, store, user_id))

    assert first == second == 0
    assert store.batches == []
    assert user.memory_enabled is False
    assert user.memory_consent_version == 0
    assert user.memory_data_epoch == 2


def test_clear_memory_restarts_at_zero_across_batches_and_isolates_users():
    user_id = uuid.uuid4()
    other_user_id = uuid.uuid4()
    user = SimpleNamespace(id=user_id, memory_enabled=True)
    db = _Db(user)
    owned_items = [
        _item(
            user_id,
            "semantic" if index % 2 == 0 else "episodes",
            f"owned-{index}",
        )
        for index in range(205)
    ]
    other_items = [
        _item(other_user_id, "semantic", f"other-{index}")
        for index in range(3)
    ]
    store = _Store([*owned_items, *other_items])

    deleted = asyncio.run(memory_service.clear_memory(db, store, user_id))

    assert deleted == 205
    assert [len(batch) for batch in store.batches] == [100, 100, 5]
    assert all(call[2] == 0 for call in store.search_calls)
    assert {(item.namespace, item.key) for item in store.items} == {
        (item.namespace, item.key) for item in other_items
    }


def test_clear_memory_keeps_consent_disabled_when_store_fails():
    user_id = uuid.uuid4()
    user = SimpleNamespace(id=user_id, memory_enabled=True)
    db = _Db(user)

    class FailingStore:
        async def asearch(self, _prefix, *, limit, offset, refresh_ttl):
            raise RuntimeError("database unavailable")

    with pytest.raises(memory_service.MemoryStoreError, match="search"):
        asyncio.run(memory_service.clear_memory(db, FailingStore(), user_id))

    assert user.memory_enabled is False
    assert db.commit_count == 2
    assert db.rollback_count == 0


def test_clear_memory_without_store_still_reasserts_disabled_consent():
    user_id = uuid.uuid4()
    user = SimpleNamespace(id=user_id, memory_enabled=True)

    class ConcurrentConsentDb(_Db):
        async def commit(self):
            await super().commit()
            if self.commit_count == 1:
                # Simulate an enabling PATCH completing before the second lock.
                self.user.memory_enabled = True

    db = ConcurrentConsentDb(user)

    with pytest.raises(memory_service.MemoryStoreError, match="unavailable"):
        asyncio.run(memory_service.clear_memory(db, None, user_id))

    assert db.execute_count == 4
    assert db.commit_count == 2
    assert user.memory_enabled is False


def test_delete_api_maps_missing_store_to_503_after_disabling_consent():
    user_id = uuid.uuid4()
    user = SimpleNamespace(id=user_id, memory_enabled=True)
    db = _Db(user)
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace())
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            memory_api.delete_memory(
                request=request,
                user=user,
                db=db,
            )
        )

    assert exc_info.value.status_code == 503
    assert user.memory_enabled is False


def test_list_api_maps_store_failure_to_503():
    user_id = uuid.uuid4()
    user = SimpleNamespace(id=user_id, memory_enabled=False)

    class FailingStore:
        async def asearch(self, _prefix, **_kwargs):
            raise RuntimeError("database unavailable")

    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(store=FailingStore()))
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            memory_api.list_memory(
                request=request,
                category=None,
                limit=50,
                offset=0,
                user=user,
            )
        )

    assert exc_info.value.status_code == 503
