"""Durable memory outbox repository contracts."""

import asyncio
from datetime import datetime, timezone
import uuid

from sqlalchemy.dialects import postgresql

from app.db.models.memory_job import MemoryJob
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


class _ClaimResult:
    def __init__(self, job):
        self.job = job

    def scalar_one_or_none(self):
        return self.job


class _ClaimDb(_Db):
    def __init__(self, job):
        super().__init__()
        self.job = job

    async def execute(self, statement):
        self.statement = statement
        return _ClaimResult(self.job)


def test_memory_job_ownership_foreign_keys_have_matching_child_indexes():
    table = MemoryJob.__table__
    foreign_keys = {
        constraint.name: tuple(column.name for column in constraint.columns)
        for constraint in table.foreign_key_constraints
    }
    indexes = {
        index.name: tuple(column.name for column in index.columns)
        for index in table.indexes
    }

    assert foreign_keys["fk_memory_jobs_conversation_owner"] == (
        "conversation_id",
        "user_id",
    )
    assert foreign_keys["fk_memory_jobs_user_message_conversation"] == (
        "source_user_message_id",
        "conversation_id",
    )
    assert foreign_keys["fk_memory_jobs_assistant_message_conversation"] == (
        "source_assistant_message_id",
        "conversation_id",
    )
    assert foreign_keys["fk_memory_jobs_api_key_owner"] == (
        "api_key_id",
        "user_id",
    )
    assert indexes["ix_memory_jobs_conversation_revision"][:1] == (
        "conversation_id",
    )
    assert indexes["ix_memory_jobs_user_message_conversation"] == foreign_keys[
        "fk_memory_jobs_user_message_conversation"
    ]
    assert indexes[
        "ix_memory_jobs_assistant_message_conversation"
    ] == foreign_keys["fk_memory_jobs_assistant_message_conversation"]
    assert indexes["ix_memory_jobs_api_key_user"] == foreign_keys[
        "fk_memory_jobs_api_key_owner"
    ]


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

    # Privacy deletion keeps retrying after the normal extraction limit.
    job.status = "processing"
    job.attempts = 3
    job.lease_until = datetime.now(timezone.utc)
    retry_db = _ClaimDb(job)
    asyncio.run(
        memory_job_repo.retry_claim(
            retry_db,
            job_id=job.id,
            lease_until=job.lease_until,
            max_attempts=3,
            error_code="store_delete_failed",
        )
    )
    assert job.status == "retry"
    assert job.attempts == 4


def test_claim_next_uses_skip_locked_and_sets_a_bounded_lease():
    job = MemoryJob(
        operation="extract_episode",
        user_id=uuid.uuid4(),
        conversation_id=uuid.uuid4(),
        target_revision=1,
        consent_version=1,
        data_epoch=0,
        status="pending",
        attempts=0,
        priority=100,
        available_at=datetime.now(timezone.utc),
        dedupe_key=f"test:{uuid.uuid4()}",
    )
    db = _ClaimDb(job)

    claimed = asyncio.run(
        memory_job_repo.claim_next(
            db,
            lease_seconds=90,
            max_attempts=3,
        )
    )

    assert claimed is job
    assert job.status == "processing"
    assert job.attempts == 0
    assert job.lease_until is not None
    sql = str(db.statement.compile(dialect=postgresql.dialect()))
    assert "FOR UPDATE SKIP LOCKED" in sql
    assert "ORDER BY memory_jobs.priority ASC" in sql
