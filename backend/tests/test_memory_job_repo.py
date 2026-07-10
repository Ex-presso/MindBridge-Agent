"""Durable memory outbox repository contracts."""

import asyncio
import uuid

from sqlalchemy.dialects import postgresql

from app.db.repositories import memory_job_repo


class _Result:
    rowcount = 3


class _Db:
    def __init__(self):
        self.statement = None
        self.added = []
        self.flush_count = 0

    async def execute(self, statement):
        self.statement = statement
        return _Result()

    def add(self, value):
        self.added.append(value)

    async def flush(self):
        self.flush_count += 1


def test_cancel_unfinished_jobs_clears_lease_and_uses_terminal_status():
    db = _Db()
    user_id = uuid.uuid4()

    canceled = asyncio.run(
        memory_job_repo.cancel_unfinished_for_user(db, user_id)
    )

    assert canceled == 3
    compiled = db.statement.compile(dialect=postgresql.dialect())
    sql = str(compiled)
    assert "UPDATE memory_jobs" in sql
    assert "memory_jobs.user_id" in sql
    assert "memory_jobs.status IN" in sql
    assert compiled.params["status"] == "canceled"
    assert compiled.params["lease_until"] is None
    assert memory_job_repo.UNFINISHED_STATUSES == {
        "pending",
        "processing",
        "retry",
    }


def test_crisis_delete_job_has_stable_dedupe_key_and_high_priority():
    db = _Db()
    user_id = uuid.uuid4()
    conversation_id = uuid.uuid4()

    job = asyncio.run(
        memory_job_repo.enqueue_delete_episode(
            db,
            user_id=user_id,
            conversation_id=conversation_id,
            target_revision=4,
            consent_version=2,
            data_epoch=3,
        )
    )

    assert db.added == [job]
    assert db.flush_count == 1
    assert job.operation == "delete_episode"
    assert job.priority == 0
    assert job.dedupe_key == (
        f"delete_episode:crisis:{user_id}:{conversation_id}"
    )
