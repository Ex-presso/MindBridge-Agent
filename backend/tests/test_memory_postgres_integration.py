"""Opt-in PostgreSQL integration coverage for memory write-safety transactions."""

import asyncio
import os
import uuid

import pytest
from langgraph.store.memory import InMemoryStore
from sqlalchemy import delete, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.models.api_key import UserApiKey
from app.db.models.conversation import Conversation
from app.db.models.memory_job import MemoryJob
from app.db.models.message import Message
from app.db.models.user import User
from app.db.repositories import api_key_repo, user_repo
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
async def test_runtime_sessions_are_read_committed_on_postgres():
    engine = create_async_engine(
        settings.ASYNC_DATABASE_URL,
        isolation_level=settings.DATABASE_ISOLATION_LEVEL,
    )
    sessions = async_sessionmaker(engine)
    try:
        async with sessions() as db:
            assert await db.scalar(text("SHOW transaction_isolation")) == (
                "read committed"
            )
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_api_key_mutation_and_chat_barriers_are_linearizable_on_postgres():
    engine = create_async_engine(
        settings.ASYNC_DATABASE_URL,
        isolation_level=settings.DATABASE_ISOLATION_LEVEL,
    )
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    user_id = uuid.uuid4()

    async def replace_key():
        async with sessions() as mutation_db:
            access = await user_repo.get_memory_access_for_update(
                mutation_db,
                user_id,
            )
            assert access is not None
            await api_key_repo.upsert(
                mutation_db,
                user_id=user_id,
                provider="openai",
                api_key_encrypted="new-ciphertext",
            )
            await mutation_db.commit()

    async def read_key_after_barrier():
        async with sessions() as chat_db:
            access = await user_repo.get_memory_access_for_chat(
                chat_db,
                user_id,
            )
            assert access is not None
            key = await api_key_repo.get_by_provider(
                chat_db,
                user_id,
                "openai",
            )
            return None if key is None else key.api_key_encrypted

    try:
        async with sessions() as setup_db:
            setup_db.add(
                User(
                    id=user_id,
                    email=f"credential-barrier-{user_id}@example.com",
                    hashed_password="not-used",
                )
            )
            await setup_db.flush()
            setup_db.add(
                UserApiKey(
                    user_id=user_id,
                    provider="openai",
                    api_key_encrypted="old-ciphertext",
                )
            )
            await setup_db.commit()

        # A running chat owns KEY SHARE. Credential replacement must wait, and
        # can only commit after the old-key call releases its barrier.
        async with sessions() as chat_db:
            await chat_db.begin()
            access = await user_repo.get_memory_access_for_chat(chat_db, user_id)
            assert access is not None
            old_key = await api_key_repo.get_by_provider(
                chat_db,
                user_id,
                "openai",
            )
            assert old_key is not None
            assert old_key.api_key_encrypted == "old-ciphertext"

            replace_task = asyncio.create_task(replace_key())
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(
                    asyncio.shield(replace_task),
                    timeout=0.15,
                )
            await chat_db.commit()
            await asyncio.wait_for(replace_task, timeout=1.0)

        assert await read_key_after_barrier() == "new-ciphertext"

        # The reverse ordering also blocks: once DELETE owns FOR UPDATE, a new
        # chat cannot pass its KEY SHARE barrier until DELETE commits, and then
        # its fresh READ COMMITTED key lookup observes absence.
        async with sessions() as mutation_db:
            await mutation_db.begin()
            access = await user_repo.get_memory_access_for_update(
                mutation_db,
                user_id,
            )
            assert access is not None
            assert await api_key_repo.delete_by_provider(
                mutation_db,
                user_id,
                "openai",
            )

            read_task = asyncio.create_task(read_key_after_barrier())
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(
                    asyncio.shield(read_task),
                    timeout=0.15,
                )
            await mutation_db.commit()
            assert await asyncio.wait_for(read_task, timeout=1.0) is None

        async with sessions() as cleanup_db:
            await cleanup_db.execute(delete(User).where(User.id == user_id))
            await cleanup_db.commit()
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_account_delete_lock_blocks_new_chat_barrier_on_postgres():
    engine = create_async_engine(settings.ASYNC_DATABASE_URL)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    user_id = uuid.uuid4()

    async def enter_chat_barrier():
        async with sessions() as chat_db:
            access = await user_repo.get_memory_access_for_chat(
                chat_db,
                user_id,
            )
            if access is not None and not access.account_deletion_pending:
                chat_db.add(
                    Conversation(
                        user_id=user_id,
                        title="Must not be inserted",
                    )
                )
                await chat_db.commit()
            return access

    try:
        async with sessions() as setup_db:
            setup_db.add(
                User(
                    id=user_id,
                    email=f"barrier-{user_id}@example.com",
                    hashed_password="not-used",
                )
            )
            await setup_db.commit()

        async with sessions() as delete_db:
            await delete_db.begin()
            assert await user_repo.lock_for_account_deletion(delete_db, user_id)
            await delete_db.execute(
                update(User)
                .where(User.id == user_id)
                .values(
                    memory_enabled=False,
                    account_deletion_pending=True,
                )
            )

            chat_task = asyncio.create_task(enter_chat_barrier())
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(asyncio.shield(chat_task), timeout=0.15)

            await delete_db.commit()
            access = await asyncio.wait_for(chat_task, timeout=1.0)
            assert access is not None
            assert access.account_deletion_pending is True

        async with sessions() as verify_db:
            assert await verify_db.scalar(
                select(Conversation.id).where(Conversation.user_id == user_id)
            ) is None
            await verify_db.execute(delete(User).where(User.id == user_id))
            await verify_db.commit()
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_memory_job_composite_ownership_and_cascades_on_postgres():
    engine = create_async_engine(settings.ASYNC_DATABASE_URL)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    user_a = uuid.uuid4()
    user_b = uuid.uuid4()
    conversation_a = uuid.uuid4()
    conversation_b = uuid.uuid4()
    message_a = uuid.uuid4()
    message_b = uuid.uuid4()
    api_key_a = uuid.uuid4()
    api_key_b = uuid.uuid4()

    def job(*, dedupe: str, user_id=user_a, conversation_id=conversation_a,
            source_user_message_id=None, api_key_id=None):
        return MemoryJob(
            operation="extract_episode",
            user_id=user_id,
            conversation_id=conversation_id,
            target_revision=1,
            consent_version=1,
            data_epoch=1,
            source_user_message_id=source_user_message_id,
            api_key_id=api_key_id,
            dedupe_key=f"integration:{dedupe}:{uuid.uuid4()}",
        )

    async def assert_rejected(candidate: MemoryJob) -> None:
        async with sessions() as rejected_db:
            rejected_db.add(candidate)
            with pytest.raises(IntegrityError):
                await rejected_db.commit()
            await rejected_db.rollback()

    try:
        async with sessions() as setup_db:
            setup_db.add_all(
                [
                    User(
                        id=user_a,
                        email=f"ownership-a-{user_a}@example.com",
                        hashed_password="not-used",
                    ),
                    User(
                        id=user_b,
                        email=f"ownership-b-{user_b}@example.com",
                        hashed_password="not-used",
                    ),
                ]
            )
            await setup_db.flush()
            setup_db.add_all(
                [
                    Conversation(
                        id=conversation_a,
                        user_id=user_a,
                        title="A",
                    ),
                    Conversation(
                        id=conversation_b,
                        user_id=user_b,
                        title="B",
                    ),
                ]
            )
            await setup_db.flush()
            setup_db.add_all(
                [
                    Message(
                        id=message_a,
                        conversation_id=conversation_a,
                        role="user",
                        content="A",
                    ),
                    Message(
                        id=message_b,
                        conversation_id=conversation_b,
                        role="user",
                        content="B",
                    ),
                    UserApiKey(
                        id=api_key_a,
                        user_id=user_a,
                        provider="openai",
                        api_key_encrypted="encrypted-a",
                    ),
                    UserApiKey(
                        id=api_key_b,
                        user_id=user_b,
                        provider="openai",
                        api_key_encrypted="encrypted-b",
                    ),
                ]
            )
            await setup_db.commit()

        await assert_rejected(
            job(
                dedupe="wrong-conversation-owner",
                conversation_id=conversation_b,
            )
        )
        await assert_rejected(
            job(
                dedupe="wrong-message-conversation",
                source_user_message_id=message_b,
            )
        )
        await assert_rejected(
            job(
                dedupe="wrong-api-key-owner",
                api_key_id=api_key_b,
            )
        )

        message_job = job(
            dedupe="message-cascade",
            source_user_message_id=message_a,
        )
        key_job = job(dedupe="key-cascade", api_key_id=api_key_a)
        conversation_job = job(dedupe="conversation-cascade")
        async with sessions() as valid_db:
            valid_db.add_all([message_job, key_job, conversation_job])
            await valid_db.commit()

            await valid_db.execute(delete(Message).where(Message.id == message_a))
            await valid_db.commit()
            assert await valid_db.scalar(
                select(MemoryJob.id).where(MemoryJob.id == message_job.id)
            ) is None

            await valid_db.execute(
                delete(UserApiKey).where(UserApiKey.id == api_key_a)
            )
            await valid_db.commit()
            assert await valid_db.scalar(
                select(MemoryJob.id).where(MemoryJob.id == key_job.id)
            ) is None

            await valid_db.execute(
                delete(Conversation).where(Conversation.id == conversation_a)
            )
            await valid_db.commit()
            assert await valid_db.scalar(
                select(MemoryJob.id).where(
                    MemoryJob.id == conversation_job.id
                )
            ) is None

            await valid_db.execute(
                delete(User).where(User.id.in_((user_a, user_b)))
            )
            await valid_db.commit()
    finally:
        await engine.dispose()


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
