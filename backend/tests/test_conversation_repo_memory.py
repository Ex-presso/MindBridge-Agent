"""Relational episode guards and memory revision repository contracts."""

import asyncio
from types import SimpleNamespace
import uuid

from sqlalchemy.dialects import postgresql

from app.db.repositories import conversation_repo


class _Scalars:
    def __init__(self, values):
        self.values = values

    def all(self):
        return self.values


class _Result:
    def __init__(self, *, values=(), scalar=None):
        self.values = values
        self.scalar = scalar

    def scalars(self):
        return _Scalars(self.values)

    def scalar_one_or_none(self):
        return self.scalar


class _Db:
    def __init__(self, result):
        self.result = result
        self.statements = []

    async def execute(self, statement):
        self.statements.append(statement)
        return self.result


def _sql(statement):
    return str(statement.compile(dialect=postgresql.dialect()))


def test_account_cleanup_locks_conversations_in_stable_order():
    user_id = uuid.uuid4()
    conversations = [SimpleNamespace(id=uuid.UUID(int=1))]
    db = _Db(_Result(values=conversations))

    result = asyncio.run(
        conversation_repo.list_owned_for_update(db, user_id)
    )

    assert result == conversations
    sql = _sql(db.statements[0])
    assert "WHERE conversations.user_id" in sql
    assert "ORDER BY conversations.id" in sql
    assert "FOR UPDATE" in sql


def test_episode_guard_requires_owner_existing_ids_and_no_crisis():
    user_id = uuid.uuid4()
    allowed_id = uuid.uuid4()
    rejected_id = uuid.uuid4()
    db = _Db(_Result(values=[allowed_id]))

    allowed = asyncio.run(
        conversation_repo.filter_memory_eligible_ids(
            db,
            user_id,
            (allowed_id, rejected_id),
        )
    )

    assert allowed == {str(allowed_id)}
    sql = _sql(db.statements[0])
    assert "conversations.user_id" in sql
    assert "conversations.id IN" in sql
    assert "conversations.memory_crisis_seen IS false" in sql


def test_crisis_tombstone_is_one_way_and_reports_transition():
    db = _Db(_Result(scalar=uuid.uuid4()))

    transitioned = asyncio.run(
        conversation_repo.mark_memory_crisis_seen(
            db,
            uuid.uuid4(),
            uuid.uuid4(),
        )
    )

    assert transitioned is True
    sql = _sql(db.statements[0])
    assert "memory_crisis_seen IS false" in sql
    assert "memory_crisis_seen=" in sql
    assert "RETURNING conversations.id" in sql


def test_memory_revision_uses_atomic_increment_returning():
    db = _Db(_Result(scalar=8))

    revision = asyncio.run(
        conversation_repo.increment_memory_revision(db, uuid.uuid4())
    )

    assert revision == 8
    sql = _sql(db.statements[0])
    assert "memory_revision=(conversations.memory_revision +" in sql
    assert "RETURNING conversations.memory_revision" in sql
