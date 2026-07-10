"""Opt-in PostgreSQL integration coverage for memory write-safety transactions."""

import os
import uuid

import pytest
from langgraph.store.memory import InMemoryStore
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.models.conversation import Conversation
from app.db.models.memory_job import MemoryJob
from app.db.models.user import User
from app.db.repositories import user_repo
from app.services import account_service, memory_service
from app.services.memory_selection import select_memory
from config.settings import settings


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LOCAL_INTEGRATION") != "1",
    reason="set RUN_LOCAL_INTEGRATION=1 with an isolated migrated PostgreSQL",
)


class _RecordingCheckpointer:
    def __init__(self) -> None:
        self.deleted_threads: list[str] = []

    async def adelete_thread(self, thread_id: str) -> None:
        self.deleted_threads.append(thread_id)


@pytest.mark.asyncio
async def test_memory_epochs_jobs_and_account_cascade_on_postgres():
    engine = create_async_engine(settings.ASYNC_DATABASE_URL)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    user_id = uuid.uuid4()
    conversation_id = uuid.uuid4()
    job_id = uuid.uuid4()
    store = InMemoryStore()

    try:
        async with sessions() as db:
            db.add(
                User(
                    id=user_id,
                    email=f"integration-{user_id}@example.com",
                    hashed_password="not-used",
                )
            )
            await db.flush()
            db.add(
                Conversation(
                    id=conversation_id,
                    user_id=user_id,
                    title="Integration",
                )
            )
            await db.flush()
            db.add(
                MemoryJob(
                    id=job_id,
                    operation="delete_episode",
                    user_id=user_id,
                    conversation_id=conversation_id,
                    target_revision=0,
                    consent_version=0,
                    data_epoch=0,
                    dedupe_key=f"integration:{job_id}",
                )
            )
            await db.commit()

            assert await memory_service.set_memory_consent(
                db, user_id, enabled=True
            )
            first = await user_repo.get_memory_access_snapshot(db, user_id)
            assert first == user_repo.MemoryAccessSnapshot(True, 1, 0)

            assert await memory_service.set_memory_consent(
                db, user_id, enabled=True
            )
            unchanged = await user_repo.get_memory_access_snapshot(db, user_id)
            assert unchanged == first

            await store.aput(
                memory_service.memory_namespace(user_id, "semantic"),
                "old",
                {
                    "kind": "preference",
                    "content": "old epoch",
                    "status": "active",
                    "explicit": True,
                    "data_epoch": 0,
                },
                index=False,
            )
            assert await memory_service.clear_memory(db, store, user_id) == 1

            cleared = await user_repo.get_memory_access_snapshot(db, user_id)
            assert cleared == user_repo.MemoryAccessSnapshot(False, 2, 1)
            stored_job_status = await db.scalar(
                select(MemoryJob.status).where(MemoryJob.id == job_id)
            )
            assert stored_job_status == "canceled"

            assert await memory_service.set_memory_consent(
                db, user_id, enabled=True
            )
            current = await user_repo.get_memory_access_snapshot(db, user_id)
            assert current == user_repo.MemoryAccessSnapshot(True, 3, 1)

            await store.aput(
                memory_service.memory_namespace(user_id, "semantic"),
                "residual",
                {
                    "kind": "preference",
                    "content": "must stay invisible",
                    "status": "active",
                    "explicit": True,
                    "data_epoch": 0,
                },
                index=False,
            )
            selection = await select_memory(
                store,
                user_id,
                "anything",
                episode_limit=0,
                expected_data_epoch=current.data_epoch,
            )
            assert selection.semantic == ()

            checkpointer = _RecordingCheckpointer()
            deleted_count = await account_service.delete_account(
                db,
                user_id,
                store=store,
                checkpointer=checkpointer,
            )
            assert deleted_count == 1
            assert checkpointer.deleted_threads == [str(conversation_id)]

            assert await db.scalar(
                select(User.id).where(User.id == user_id)
            ) is None
            assert await db.scalar(
                select(Conversation.id).where(Conversation.id == conversation_id)
            ) is None
            assert await db.scalar(
                select(MemoryJob.id).where(MemoryJob.id == job_id)
            ) is None
            assert await store.asearch(
                memory_service.memory_user_prefix(user_id),
                limit=10,
            ) == []
    finally:
        await engine.dispose()
