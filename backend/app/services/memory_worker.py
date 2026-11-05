"""Background memory jobs.

A job either extracts memories from one finished turn or deletes the memories
of one conversation. Extraction calls a model, so it runs here instead of in
the chat request.

No database lock is held while the model runs. The job's consent snapshot is
checked before the call and again after it, and a write that lost that race
is undone. A redelivered job finds its earlier write by source message id.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from app.core.auth.encryption import decrypt_value
from app.core.llm.provider import get_llm
from app.core.memory import MemoryClient, get_memory_client
from app.core.observability.context import bind_request_id, reset_request_id
from app.core.observability.metrics import MEMORY_JOB_SECONDS, MEMORY_JOBS
from app.core.observability.tracing import tracer
from app.db.models.memory_job import MemoryJob
from app.db.repositories import (
    api_key_repo,
    conversation_repo,
    memory_job_repo,
    message_repo,
    user_repo,
)
from config.settings import settings

logger = logging.getLogger(__name__)

_MODEL_TIMEOUT_SECONDS = 60.0
# Outcomes that mean the job did what it was asked. Any other outcome names
# the reason the job no longer applied.
_DONE = frozenset({"stored", "nothing_to_store", "already_stored", "deleted"})


@dataclass(frozen=True)
class _ExtractionInput:
    text: str
    llm: Any


async def _closed_gate(db, job: MemoryJob) -> str | None:
    """Return why this job may no longer write memory, or None if it still may.

    The job carries the consent version and data epoch that were current when
    its turn was saved. Any consent change or memory clear since then moves
    one of them, which closes the gate.
    """
    access = await user_repo.get_memory_access_snapshot(db, job.user_id)
    if access is None or access.account_deletion_pending:
        return "user_gone"
    if not settings.MEMORY_ENABLED or not access.enabled:
        return "consent_off"
    if (access.consent_version, access.data_epoch) != (job.consent_version, job.data_epoch):
        return "consent_changed"
    conversation = await conversation_repo.get_by_id(db, job.conversation_id)
    if conversation is None:
        return "conversation_gone"
    if conversation.memory_crisis_seen:
        return "crisis"
    return None


async def _extraction_input(db, job: MemoryJob) -> _ExtractionInput | str:
    """Load the user message and the user's own model, or say what is missing."""
    messages = await message_repo.list_by_ids_for_conversation(
        db, job.conversation_id, (job.source_user_message_id,)
    )
    if not messages or messages[0].role != "user":
        return "message_gone"
    key = await api_key_repo.get_owned_by_id(db, job.user_id, job.api_key_id)
    if key is None:
        return "credential_gone"
    llm = get_llm(
        job.provider,
        api_key=decrypt_value(key.api_key_encrypted),
        base_url=key.base_url,
        model=job.model,
        temperature=0,
        timeout=_MODEL_TIMEOUT_SECONDS,
    )
    return _ExtractionInput(text=messages[0].content, llm=llm)


async def _extract(session_factory: Any, memory: MemoryClient, job: MemoryJob) -> str:
    user_id = str(job.user_id)
    message_id = str(job.source_user_message_id)

    async with session_factory() as db:
        closed = await _closed_gate(db, job)
        source = closed or await _extraction_input(db, job)
    if isinstance(source, str):
        # An earlier attempt may have written before the gate closed.
        await memory.delete_for_message(user_id, message_id)
        return source

    if await memory.has_message(user_id, message_id):
        return "already_stored"

    added = await memory.add_user_message(
        source.llm,
        user_id=user_id,
        conversation_id=str(job.conversation_id),
        message_id=message_id,
        data_epoch=job.data_epoch,
        text=source.text,
    )

    async with session_factory() as db:
        closed = await _closed_gate(db, job)
    if closed:
        # Consent changed while the model was running. Undo the write.
        await memory.delete(record.id for record in added)
        return closed
    return "stored" if added else "nothing_to_store"


async def handle_job(session_factory: Any, memory: MemoryClient, job: MemoryJob) -> str:
    """Run one job and return its outcome. Raises on a failure worth retrying."""
    if job.operation == "delete_episode":
        await memory.delete_for_conversation(str(job.user_id), str(job.conversation_id))
        return "deleted"
    return await _extract(session_factory, memory, job)


async def process_claimed_job(
    session_factory: Any,
    claimed: MemoryJob,
    *,
    max_attempts: int,
) -> None:
    """Process one leased job and record how it ended."""
    lease_until = claimed.lease_until
    if lease_until is None:
        return

    # Jobs have no HTTP request; the job id plays the same role in the logs.
    log_token = bind_request_id(f"job-{claimed.id}")
    started = time.perf_counter()
    outcome = "error"
    try:
        with tracer.start_as_current_span(
            "memory.job",
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            span.set_attribute("memory.operation", claimed.operation)
            memory = await get_memory_client()
            outcome = await handle_job(session_factory, memory, claimed)
            span.set_attribute("memory.outcome", outcome)
        async with session_factory() as db:
            async with db.begin():
                await memory_job_repo.finish_claim(
                    db,
                    job_id=claimed.id,
                    lease_until=lease_until,
                    status="succeeded" if outcome in _DONE else "canceled",
                    error_code=None if outcome in _DONE else outcome,
                )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        # The error class only: a provider error can echo the user's message.
        logger.warning(
            "Memory job failed; operation=%s error_type=%s",
            claimed.operation,
            type(exc).__name__,
        )
        async with session_factory() as db:
            async with db.begin():
                await memory_job_repo.retry_claim(
                    db,
                    job_id=claimed.id,
                    lease_until=lease_until,
                    max_attempts=max_attempts,
                    error_code=type(exc).__name__,
                )
    else:
        logger.info("Memory job finished; operation=%s outcome=%s", claimed.operation, outcome)
    finally:
        MEMORY_JOBS.labels(claimed.operation, outcome).inc()
        MEMORY_JOB_SECONDS.labels(claimed.operation).observe(time.perf_counter() - started)
        reset_request_id(log_token)


async def run_memory_worker(session_factory: Any, *, stop_event: asyncio.Event) -> None:
    """Poll and process one leased job at a time until application shutdown."""
    logger.info("Memory worker started.")
    while not stop_event.is_set():
        try:
            async with session_factory() as db:
                async with db.begin():
                    claimed = await memory_job_repo.claim_next(
                        db,
                        lease_seconds=settings.MEMORY_WORKER_LEASE_SECONDS,
                        max_attempts=settings.MEMORY_WORKER_MAX_ATTEMPTS,
                    )
            if claimed is not None:
                await process_claimed_job(
                    session_factory,
                    claimed,
                    max_attempts=settings.MEMORY_WORKER_MAX_ATTEMPTS,
                )
                continue
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Memory worker poll failed; error_type=%s", type(exc).__name__)
        try:
            await asyncio.wait_for(
                stop_event.wait(),
                timeout=settings.MEMORY_WORKER_POLL_SECONDS,
            )
        except TimeoutError:
            pass
    logger.info("Memory worker stopped.")
