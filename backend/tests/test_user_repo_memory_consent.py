"""Consent reads use a fresh scalar query without taking a user-row lock."""

import asyncio
import uuid

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
