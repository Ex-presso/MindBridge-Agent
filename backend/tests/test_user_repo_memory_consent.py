"""User consent snapshots and row-lock modes stay explicit."""

import asyncio
import uuid

from sqlalchemy.dialects import postgresql

from app.db.repositories import user_repo


class _ScalarResult:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value


class _RecordingDb:
    def __init__(self, value):
        self.value = value
        self.statements = []

    async def execute(self, statement):
        self.statements.append(statement)
        return _ScalarResult(self.value)


class _RowResult:
    def __init__(self, value):
        self.value = value

    def one_or_none(self):
        return self.value


class _RecordingRowDb(_RecordingDb):
    async def execute(self, statement):
        self.statements.append(statement)
        return _RowResult(self.value)


def test_get_memory_enabled_reads_only_scalar_consent_without_for_update():
    user_id = uuid.uuid4()
    db = _RecordingDb(False)

    enabled = asyncio.run(user_repo.get_memory_enabled(db, user_id))

    assert enabled is False
    assert len(db.statements) == 1
    sql = str(db.statements[0].compile())
    assert "SELECT users.memory_enabled" in sql
    assert "FOR UPDATE" not in sql
    assert db.statements[0].compile().params["id_1"] == user_id


def test_get_memory_access_snapshot_reads_all_gates_atomically_without_lock():
    user_id = uuid.uuid4()
    db = _RecordingRowDb((True, 7, 11, False))

    snapshot = asyncio.run(user_repo.get_memory_access_snapshot(db, user_id))

    assert snapshot == user_repo.MemoryAccessSnapshot(
        enabled=True,
        consent_version=7,
        data_epoch=11,
    )
    assert len(db.statements) == 1
    sql = str(db.statements[0].compile())
    assert "users.memory_enabled" in sql
    assert "users.memory_consent_version" in sql
    assert "users.memory_data_epoch" in sql
    assert "users.account_deletion_pending" in sql
    assert "FOR UPDATE" not in sql


def test_chat_memory_gate_takes_key_share_before_conversation_work():
    user_id = uuid.uuid4()
    db = _RecordingRowDb((True, 7, 11, False))

    snapshot = asyncio.run(user_repo.get_memory_access_for_chat(db, user_id))

    assert snapshot == user_repo.MemoryAccessSnapshot(True, 7, 11, False)
    sql = str(db.statements[0].compile(dialect=postgresql.dialect()))
    assert "FOR KEY SHARE" in sql


def test_account_deletion_gate_takes_strong_user_lock():
    user_id = uuid.uuid4()
    db = _RecordingDb(user_id)

    locked = asyncio.run(user_repo.lock_for_account_deletion(db, user_id))

    assert locked is True
    sql = str(db.statements[0].compile(dialect=postgresql.dialect()))
    assert "FOR UPDATE" in sql
    assert "KEY SHARE" not in sql


def test_credential_mutation_gate_returns_snapshot_under_strong_lock():
    user_id = uuid.uuid4()
    db = _RecordingRowDb((False, 8, 12, False))

    snapshot = asyncio.run(user_repo.get_memory_access_for_update(db, user_id))

    assert snapshot == user_repo.MemoryAccessSnapshot(False, 8, 12, False)
    sql = str(db.statements[0].compile(dialect=postgresql.dialect()))
    assert "FOR UPDATE" in sql
    assert "KEY SHARE" not in sql
