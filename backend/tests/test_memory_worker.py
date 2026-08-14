"""Focused production enqueue, worker gate, and Store writer checks."""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
import uuid

from app.api.v1 import chat as chat_api
from app.db.models.memory_job import MemoryJob
from app.db.repositories.user_repo import MemoryAccessSnapshot
from app.schemas.episode_extraction import EpisodeClaim, EpisodeDraft
from app.schemas.memory_values import EpisodeMemoryValue
from app.services.episode_extraction import EpisodeExtractionOutcome
from app.services import memory_worker


class _MappingResult:
    def __init__(self, job: MemoryJob):
        self.job = job

    def mappings(self):
        return self

    def one_or_none(self):
        return {
            "user_id": self.job.user_id,
            "conversation_id": self.job.conversation_id,
        }


class _Db:
    def __init__(self, job: MemoryJob):
        self.job = job
        self.flush_count = 0

    async def execute(self, statement):
        return _MappingResult(self.job)

    async def flush(self):
        self.flush_count += 1


def _job(operation: str = "extract_episode") -> MemoryJob:
    user_id = uuid.uuid4()
    conversation_id = uuid.uuid4()
    return MemoryJob(
        id=uuid.uuid4(),
        operation=operation,
        user_id=user_id,
        conversation_id=conversation_id,
        target_revision=2,
        consent_version=3,
        data_epoch=4,
        source_user_message_id=(uuid.uuid4() if operation == "extract_episode" else None),
        source_assistant_message_id=(
            uuid.uuid4() if operation == "extract_episode" else None
        ),
        api_key_id=(uuid.uuid4() if operation == "extract_episode" else None),
        provider="openai" if operation == "extract_episode" else None,
        model="test-model" if operation == "extract_episode" else None,
        base_url=None,
        status="processing",
        attempts=1,
        priority=100,
        lease_until=datetime.now(timezone.utc) + timedelta(seconds=90),
        dedupe_key=f"test:{uuid.uuid4()}",
    )


def _arrange_locked_job(monkeypatch, job: MemoryJob, *, enabled: bool = True):
    access = MemoryAccessSnapshot(
        enabled=enabled,
        consent_version=job.consent_version,
        data_epoch=job.data_epoch,
    )
    conversation = SimpleNamespace(
        id=job.conversation_id,
        user_id=job.user_id,
        memory_revision=job.target_revision,
        memory_crisis_seen=False,
        memory_crisis_reviewed=True,
        memory_crisis_review_version=memory_worker.CRISIS_DETECTOR_VERSION,
    )
    monkeypatch.setattr(
        memory_worker.user_repo,
        "get_memory_access_for_worker",
        AsyncMock(return_value=access),
    )
    monkeypatch.setattr(
        memory_worker.conversation_repo,
        "get_owned_for_update",
        AsyncMock(return_value=conversation),
    )
    monkeypatch.setattr(
        memory_worker.memory_job_repo,
        "get_processing_for_update",
        AsyncMock(return_value=job),
    )
    return access


def test_successful_turn_enqueues_exact_extraction_snapshot(monkeypatch):
    user_id = uuid.uuid4()
    conversation_id = uuid.uuid4()
    user_message_id = uuid.uuid4()
    assistant_message_id = uuid.uuid4()
    api_key_id = uuid.uuid4()
    access = MemoryAccessSnapshot(True, 7, 8)
    enqueue = AsyncMock()
    monkeypatch.setattr(chat_api.settings, "MEMORY_ENABLED", True)
    monkeypatch.setattr(
        chat_api.user_repo,
        "get_memory_access_snapshot",
        AsyncMock(return_value=access),
    )
    monkeypatch.setattr(
        chat_api.memory_job_repo,
        "enqueue_extract_episode",
        enqueue,
    )
    conversation = SimpleNamespace(
        id=conversation_id,
        memory_crisis_seen=False,
        memory_crisis_reviewed=True,
        memory_crisis_review_version=chat_api.CRISIS_DETECTOR_VERSION,
    )

    asyncio.run(
        chat_api._enqueue_episode_after_success(
            object(),
            user_id=user_id,
            conversation=conversation,
            target_revision=4,
            user_message=SimpleNamespace(id=user_message_id),
            assistant_message=SimpleNamespace(id=assistant_message_id),
            api_key=SimpleNamespace(id=api_key_id, base_url="https://llm.test/v1"),
            provider="openai_compatible",
            model="local-model",
        )
    )

    assert enqueue.await_args.kwargs == {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "target_revision": 4,
        "consent_version": 7,
        "data_epoch": 8,
        "source_user_message_id": user_message_id,
        "source_assistant_message_id": assistant_message_id,
        "api_key_id": api_key_id,
        "provider": "openai_compatible",
        "model": "local-model",
        "base_url": "https://llm.test/v1",
    }


def test_worker_writes_grounded_episode(monkeypatch):
    job = _job()
    db = _Db(job)
    _arrange_locked_job(monkeypatch, job)
    monkeypatch.setattr(memory_worker.settings, "MEMORY_ENABLED", True)
    monkeypatch.setattr(
        memory_worker.api_key_repo,
        "get_owned_by_id",
        AsyncMock(
            return_value=SimpleNamespace(
                provider=job.provider,
                base_url=job.base_url,
                api_key_encrypted="encrypted",
            )
        ),
    )
    monkeypatch.setattr(
        memory_worker.memory_job_repo,
        "list_extract_source_user_ids",
        AsyncMock(return_value=(job.source_user_message_id,)),
    )
    claim_text = "Walking helped me settle after work."
    messages = [
        SimpleNamespace(
            id=job.source_user_message_id,
            role="user",
            content=claim_text,
        ),
        SimpleNamespace(
            id=job.source_assistant_message_id,
            role="assistant",
            content="Thanks for sharing.",
        ),
    ]
    monkeypatch.setattr(
        memory_worker.message_repo,
        "list_by_ids_for_conversation",
        AsyncMock(return_value=messages),
    )
    monkeypatch.setattr(memory_worker, "decrypt_value", lambda _: "key")
    monkeypatch.setattr(memory_worker, "get_llm", lambda *args, **kwargs: object())
    draft = EpisodeDraft(
        claims=(
            EpisodeClaim(
                claim=claim_text,
                evidence_message_id=str(job.source_user_message_id),
                evidence_quote=claim_text,
            ),
        ),
        topics=("Walking",),
    )
    monkeypatch.setattr(
        memory_worker,
        "extract_episode_draft",
        AsyncMock(
            return_value=EpisodeExtractionOutcome(
                status="accepted",
                draft=draft,
                input_tokens=12,
                output_tokens=5,
            )
        ),
    )
    store = SimpleNamespace(
        aget=AsyncMock(return_value=None),
        aput=AsyncMock(),
        adelete=AsyncMock(),
    )

    asyncio.run(
        memory_worker._process_locked_job(
            db,
            store,
            job_id=job.id,
            lease_until=job.lease_until,
            vector_enabled=True,
        )
    )

    stored = EpisodeMemoryValue.model_validate(store.aput.await_args.args[2])
    assert stored.target_revision == 2
    assert stored.summary == claim_text
    assert store.aput.await_args.kwargs["index"] == ["summary"]
    assert job.status == "succeeded"
    assert (job.input_tokens, job.output_tokens) == (12, 5)


def test_worker_treats_same_revision_as_idempotent_success(monkeypatch):
    job = _job()
    db = _Db(job)
    _arrange_locked_job(monkeypatch, job)
    monkeypatch.setattr(memory_worker.settings, "MEMORY_ENABLED", True)
    monkeypatch.setattr(
        memory_worker.api_key_repo,
        "get_owned_by_id",
        AsyncMock(
            return_value=SimpleNamespace(
                provider=job.provider,
                base_url=job.base_url,
            )
        ),
    )
    existing = EpisodeMemoryValue(
        conversation_id=str(job.conversation_id),
        claims=(
            EpisodeClaim(
                claim="Existing grounded claim.",
                evidence_message_id=str(job.source_user_message_id),
                evidence_quote="Existing grounded claim.",
            ),
        ),
        summary="Existing grounded claim.",
        topics=("grounded",),
        status="active",
        crisis=False,
        target_revision=job.target_revision,
        data_epoch=job.data_epoch,
    )
    monkeypatch.setattr(
        memory_worker.message_repo,
        "list_by_ids_for_conversation",
        AsyncMock(
            return_value=[
                SimpleNamespace(
                    id=job.source_user_message_id,
                    role="user",
                    content="Existing grounded claim.",
                )
            ]
        ),
    )
    store = SimpleNamespace(
        aget=AsyncMock(
            return_value=SimpleNamespace(value=existing.model_dump(mode="json"))
        ),
        aput=AsyncMock(),
        adelete=AsyncMock(),
    )

    asyncio.run(
        memory_worker._process_locked_job(
            db,
            store,
            job_id=job.id,
            lease_until=job.lease_until,
            vector_enabled=True,
        )
    )

    assert job.status == "succeeded"
    store.aput.assert_not_awaited()


def test_worker_supersedes_stale_extract_and_delete_bypasses_consent(monkeypatch):
    extract_job = _job()
    extract_db = _Db(extract_job)
    _arrange_locked_job(monkeypatch, extract_job)
    locked_conversation = (
        memory_worker.conversation_repo.get_owned_for_update.return_value
    )
    locked_conversation.memory_revision += 1
    monkeypatch.setattr(memory_worker.settings, "MEMORY_ENABLED", True)
    store = SimpleNamespace(
        aget=AsyncMock(),
        aput=AsyncMock(),
        adelete=AsyncMock(),
    )

    asyncio.run(
        memory_worker._process_locked_job(
            extract_db,
            store,
            job_id=extract_job.id,
            lease_until=extract_job.lease_until,
            vector_enabled=False,
        )
    )
    assert extract_job.status == "superseded"
    store.aput.assert_not_awaited()

    delete_job = _job("delete_episode")
    delete_db = _Db(delete_job)
    _arrange_locked_job(monkeypatch, delete_job, enabled=False)
    monkeypatch.setattr(memory_worker.settings, "MEMORY_ENABLED", False)
    asyncio.run(
        memory_worker._process_locked_job(
            delete_db,
            store,
            job_id=delete_job.id,
            lease_until=delete_job.lease_until,
            vector_enabled=False,
        )
    )
    assert delete_job.status == "succeeded"
    store.adelete.assert_awaited_once()
