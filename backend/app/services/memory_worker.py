"""Background memory jobs.

A job either extracts memories from one finished turn or deletes the memories
of one conversation. Extraction calls a model, so it runs here instead of in
the chat request. Jobs arrive from RabbitMQ; the chat request only writes an
outbox row (see outbox_relay).

No database lock is held while the model runs. The job's consent snapshot is
checked before the call and again after it, and a write that lost that race
is undone. A redelivered job finds its earlier write by source message id.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, fields
from typing import Any

from aio_pika.abc import AbstractIncomingMessage

from app.core.auth.encryption import decrypt_value
from app.core.llm.provider import get_llm
from app.core.memory import MemoryClient, get_memory_client
from app.core.messaging import JobQueue, attempt_of
from app.core.observability.context import bind_request_id, reset_request_id
from app.core.observability.metrics import MEMORY_JOB_SECONDS, MEMORY_JOBS
from app.core.observability.tracing import tracer
from app.db.repositories import (
    api_key_repo,
    conversation_repo,
    message_repo,
    user_repo,
)
from app.services.outbox_relay import run_outbox_relay
from config.settings import settings

logger = logging.getLogger(__name__)

_MODEL_TIMEOUT_SECONDS = 60.0


_RECONNECT_SECONDS = 5.0


@dataclass(frozen=True)
class MemoryTask:
    """What a queued job asks for. Ids and settings only, never message text."""

    operation: str  # "extract" or "delete_conversation"
    user_id: str
    conversation_id: str
    # The consent state when the turn was saved; see _closed_gate.
    consent_version: int = 0
    data_epoch: int = 0
    message_id: str | None = None
    api_key_id: str | None = None
    provider: str | None = None
    model: str | None = None

    @property
    def event_type(self) -> str:
        return f"memory.{self.operation}"

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "MemoryTask":
        return cls(**{f.name: payload[f.name] for f in fields(cls) if f.name in payload})


@dataclass(frozen=True)
class _ExtractionInput:
    text: str
    llm: Any


async def _closed_gate(db, task: MemoryTask) -> str | None:
    """Return why this job may no longer write memory, or None if it still may.

    The job carries the consent version and data epoch that were current when
    its turn was saved. Any consent change or memory clear since then moves
    one of them, which closes the gate.
    """
    access = await user_repo.get_memory_access_snapshot(db, uuid.UUID(task.user_id))
    if access is None or access.account_deletion_pending:
        return "user_gone"
    if not settings.MEMORY_ENABLED or not access.enabled:
        return "consent_off"
    if (access.consent_version, access.data_epoch) != (task.consent_version, task.data_epoch):
        return "consent_changed"
    conversation = await conversation_repo.get_by_id(db, uuid.UUID(task.conversation_id))
    if conversation is None:
        return "conversation_gone"
    if conversation.memory_crisis_seen:
        return "crisis"
    return None


async def _extraction_input(db, task: MemoryTask) -> _ExtractionInput | str:
    """Load the user message and the user's own model, or say what is missing."""
    if not (task.message_id and task.api_key_id and task.provider):
        return "invalid_task"
    messages = await message_repo.list_by_ids_for_conversation(
        db, uuid.UUID(task.conversation_id), (uuid.UUID(task.message_id),)
    )
    if not messages or messages[0].role != "user":
        return "message_gone"
    key = await api_key_repo.get_owned_by_id(
        db, uuid.UUID(task.user_id), uuid.UUID(task.api_key_id)
    )
    if key is None:
        return "credential_gone"
    llm = get_llm(
        task.provider,
        api_key=decrypt_value(key.api_key_encrypted),
        base_url=key.base_url,
        model=task.model,
        temperature=0,
        timeout=_MODEL_TIMEOUT_SECONDS,
    )
    return _ExtractionInput(text=messages[0].content, llm=llm)


async def _extract(session_factory: Any, memory: MemoryClient, task: MemoryTask) -> str:
    async with session_factory() as db:
        closed = await _closed_gate(db, task)
        source = closed or await _extraction_input(db, task)
    if isinstance(source, str):
        # An earlier attempt may have written before the gate closed.
        if task.message_id:
            await memory.delete_for_message(task.user_id, task.message_id)
        return source

    if await memory.has_message(task.user_id, task.message_id):
        return "already_stored"

    added = await memory.add_user_message(
        source.llm,
        user_id=task.user_id,
        conversation_id=task.conversation_id,
        message_id=task.message_id,
        data_epoch=task.data_epoch,
        text=source.text,
    )

    async with session_factory() as db:
        closed = await _closed_gate(db, task)
    if closed:
        # Consent changed while the model was running. Undo the write.
        await memory.delete(record.id for record in added)
        return closed
    return "stored" if added else "nothing_to_store"


async def handle_task(session_factory: Any, memory: MemoryClient, task: MemoryTask) -> str:
    """Run one job and return its outcome. Raises on a failure worth retrying."""
    if task.operation == "delete_conversation":
        await memory.delete_for_conversation(task.user_id, task.conversation_id)
        return "deleted"
    return await _extract(session_factory, memory, task)


async def _consume(session_factory: Any, queue: JobQueue, message: AbstractIncomingMessage) -> None:
    """Process one delivery, then acknowledge, retry or bury it."""
    try:
        task = MemoryTask.from_payload(json.loads(message.body))
    except Exception:
        # Retrying cannot fix a payload that does not parse.
        MEMORY_JOBS.labels("unknown", "invalid").inc()
        await queue.bury(message)
        return

    # Jobs have no HTTP request; the job id plays the same role in the logs.
    log_token = bind_request_id(f"job-{message.message_id}")
    started = time.perf_counter()
    outcome = "error"
    try:
        with tracer.start_as_current_span(
            "memory.job",
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            span.set_attribute("memory.operation", task.operation)
            span.set_attribute("memory.attempt", attempt_of(message))
            memory = await get_memory_client()
            outcome = await handle_task(session_factory, memory, task)
            span.set_attribute("memory.outcome", outcome)
        await message.ack()
        logger.info("Memory job finished; operation=%s outcome=%s", task.operation, outcome)
    except asyncio.CancelledError:
        # Shutdown: the unacknowledged message is redelivered later.
        raise
    except Exception as exc:
        # The error class only: a provider error can echo the user's message.
        logger.warning(
            "Memory job failed; operation=%s attempt=%s error_type=%s",
            task.operation,
            attempt_of(message),
            type(exc).__name__,
        )
        # A deletion is a privacy promise, so it is retried without limit.
        exhausted = (
            task.operation == "extract"
            and attempt_of(message) >= settings.MEMORY_WORKER_MAX_ATTEMPTS
        )
        if exhausted:
            outcome = "dead"
            await queue.bury(message)
        else:
            outcome = "retry"
            await queue.retry_later(message)
    finally:
        MEMORY_JOBS.labels(task.operation, outcome).inc()
        MEMORY_JOB_SECONDS.labels(task.operation).observe(time.perf_counter() - started)
        reset_request_id(log_token)


async def _run_consumer(session_factory: Any, queue: JobQueue) -> None:
    async for message in queue.messages():
        await _consume(session_factory, queue, message)


async def run_memory_pipeline(session_factory: Any) -> None:
    """Run the outbox relay and the job consumer until cancelled.

    Both share one RabbitMQ connection. If the broker cannot be reached, or
    either side fails, everything is torn down and started again after a
    pause; outbox rows simply wait in PostgreSQL in the meantime.
    """
    while True:
        queue: JobQueue | None = None
        try:
            queue = await JobQueue.connect(settings.RABBITMQ_URL)
            logger.info("Memory pipeline connected to RabbitMQ.")
            async with asyncio.TaskGroup() as group:
                group.create_task(run_outbox_relay(session_factory, queue))
                group.create_task(_run_consumer(session_factory, queue))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            reason = exc.exceptions[0] if isinstance(exc, ExceptionGroup) else exc
            logger.warning(
                "Memory pipeline stopped; error_type=%s. Retrying in %ss.",
                type(reason).__name__,
                _RECONNECT_SECONDS,
            )
        finally:
            if queue is not None:
                try:
                    await queue.close()
                except Exception:
                    pass
        await asyncio.sleep(_RECONNECT_SECONDS)
