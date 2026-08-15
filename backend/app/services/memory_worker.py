"""Leased outbox worker for grounded episodic-memory writes and deletions."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import datetime, timezone
import logging
from typing import Any
import uuid

from app.core.agent.safety import CRISIS_DETECTOR_VERSION
from app.core.auth.encryption import decrypt_value
from app.core.llm.provider import get_llm
from app.db.models.memory_job import MemoryJob
from app.db.repositories import (
    api_key_repo,
    conversation_repo,
    memory_job_repo,
    message_repo,
    user_repo,
)
from app.schemas.episode_extraction import EpisodeDraft, EpisodeSourceMessage
from app.schemas.memory_values import EpisodeMemoryValue
from app.services.episode_extraction import (
    EpisodeInputValidationError,
    EpisodeModelInvocationError,
    EpisodeModelTimeoutError,
    StructuredOutputUnsupportedError,
    extract_episode_draft,
)
from app.services.memory_service import episode_namespace
from app.services.semantic_memory import (
    derive_semantic_candidates,
    write_semantic_candidates,
)
from config.settings import settings


logger = logging.getLogger(__name__)
_STORE_TIMEOUT_SECONDS = 15.0
_MAX_SOURCE_MESSAGES = 40
_MAX_TRANSCRIPT_CHARS = 12_000
_MAX_MESSAGE_CHARS = 4_000


class _RetryableJobError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _BlockedJobError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _complete_locked(
    job: MemoryJob,
    *,
    status: str,
    error_code: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> None:
    job.status = status
    job.lease_until = None
    job.error_code = error_code
    job.input_tokens = input_tokens
    job.output_tokens = output_tokens
    job.updated_at = datetime.now(timezone.utc)


def _parse_existing_episode(
    item: Any,
    *,
    conversation_id: uuid.UUID,
) -> EpisodeMemoryValue | None:
    value = getattr(item, "value", None)
    if not isinstance(value, Mapping):
        return None
    try:
        episode = EpisodeMemoryValue.model_validate(dict(value))
    except Exception:
        return None
    if episode.conversation_id != str(conversation_id):
        return None
    return episode


async def _read_episode(
    store: Any,
    *,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID,
) -> EpisodeMemoryValue | None:
    try:
        async with asyncio.timeout(_STORE_TIMEOUT_SECONDS):
            item = await store.aget(
                episode_namespace(user_id),
                str(conversation_id),
                refresh_ttl=False,
            )
    except Exception as exc:
        raise _RetryableJobError("store_read_failed") from exc
    return _parse_existing_episode(item, conversation_id=conversation_id)


async def _episode_evidence_is_grounded(
    db: Any,
    *,
    conversation_id: uuid.UUID,
    episode: EpisodeMemoryValue,
) -> bool:
    evidence_ids = tuple(
        dict.fromkeys(uuid.UUID(claim.evidence_message_id) for claim in episode.claims)
    )
    rows = await message_repo.list_by_ids_for_conversation(
        db,
        conversation_id,
        evidence_ids,
    )
    by_id = {row.id: row for row in rows}
    return all(
        (message := by_id.get(uuid.UUID(claim.evidence_message_id))) is not None
        and message.role == "user"
        and claim.evidence_quote in message.content
        for claim in episode.claims
    )


async def _load_grounded_sources(
    db: Any,
    job: MemoryJob,
    prior: EpisodeDraft | None,
    *,
    prior_revision: int,
) -> tuple[tuple[EpisodeSourceMessage, ...], dict[uuid.UUID, str]]:
    if (
        job.source_user_message_id is None
        or job.source_assistant_message_id is None
    ):
        raise _BlockedJobError("source_ids_missing")

    new_source_ids = await memory_job_repo.list_extract_source_user_ids(
        db,
        user_id=job.user_id,
        conversation_id=job.conversation_id,
        consent_version=job.consent_version,
        data_epoch=job.data_epoch,
        after_revision=prior_revision,
        through_revision=job.target_revision,
        limit=_MAX_SOURCE_MESSAGES,
    )
    prior_quotes: dict[uuid.UUID, list[str]] = {}
    if prior is not None:
        for claim in prior.claims:
            message_id = uuid.UUID(claim.evidence_message_id)
            prior_quotes.setdefault(message_id, []).append(claim.evidence_quote)
    prior_source_ids = tuple(prior_quotes)
    candidate_new_ids = tuple(dict.fromkeys(new_source_ids))
    if job.source_user_message_id not in candidate_new_ids:
        raise _BlockedJobError("source_turn_missing")

    lookup_ids = tuple(
        dict.fromkeys(
            (*prior_source_ids, *candidate_new_ids, job.source_assistant_message_id)
        )
    )
    rows = await message_repo.list_by_ids_for_conversation(
        db,
        job.conversation_id,
        lookup_ids,
    )
    by_id = {row.id: row for row in rows}
    assistant = by_id.get(job.source_assistant_message_id)
    if assistant is None or assistant.role != "assistant":
        raise _BlockedJobError("source_assistant_invalid")

    full_contents: dict[uuid.UUID, str] = {}
    for message_id in (*prior_source_ids, *candidate_new_ids):
        message = by_id.get(message_id)
        if message is None or message.role != "user":
            raise _BlockedJobError("source_user_invalid")
        full_contents[message_id] = message.content

    projected_prior: dict[uuid.UUID, str] = {}
    for message_id, quotes in prior_quotes.items():
        projection = "\n".join(dict.fromkeys(quotes))
        projected_prior[message_id] = (
            projection
            if len(projection) <= _MAX_MESSAGE_CHARS
            else full_contents[message_id]
        )
    if any(
        quote not in full_contents[message_id]
        for message_id, quotes in prior_quotes.items()
        for quote in quotes
    ):
        raise _BlockedJobError("prior_evidence_invalid")

    selected_new: set[uuid.UUID] = set()
    total_chars = sum(len(content) for content in projected_prior.values())
    if total_chars > _MAX_TRANSCRIPT_CHARS:
        raise _BlockedJobError("source_budget_exceeded")
    final_count = len(prior_source_ids)
    for message_id in reversed(candidate_new_ids):
        projected_chars = len(projected_prior.get(message_id, ""))
        candidate_chars = total_chars - projected_chars + len(
            full_contents[message_id]
        )
        candidate_count = final_count + (message_id not in prior_quotes)
        if (
            candidate_count <= _MAX_SOURCE_MESSAGES
            and candidate_chars <= _MAX_TRANSCRIPT_CHARS
        ):
            selected_new.add(message_id)
            total_chars = candidate_chars
            final_count = candidate_count
        elif message_id == job.source_user_message_id:
            raise _BlockedJobError("source_budget_exceeded")

    if job.source_user_message_id not in selected_new:
        raise _BlockedJobError("source_turn_missing")

    sources: list[EpisodeSourceMessage] = []
    for message_id in prior_source_ids:
        if message_id in selected_new:
            continue
        sources.append(
            EpisodeSourceMessage(
                id=str(message_id),
                role="user",
                content=projected_prior[message_id],
            )
        )
    for message_id in candidate_new_ids:
        if message_id not in selected_new:
            continue
        sources.append(
            EpisodeSourceMessage(
                id=str(message_id),
                role="user",
                content=full_contents[message_id],
            )
        )
    return tuple(sources), full_contents


def _draft_is_grounded_in_full_messages(
    draft: EpisodeDraft,
    full_contents: Mapping[uuid.UUID, str],
) -> bool:
    return all(
        (content := full_contents.get(uuid.UUID(claim.evidence_message_id)))
        is not None
        and claim.evidence_quote in content
        for claim in draft.claims
    )


async def _process_locked_job(
    db: Any,
    store: Any,
    *,
    job_id: uuid.UUID,
    lease_until: datetime,
    vector_enabled: bool,
) -> None:
    # FOR SHARE blocks consent, clear, account deletion, and credential writes
    # while a transcript is outside PostgreSQL at the provider or Store.
    initial_job_result = await db.execute(
        MemoryJob.__table__.select().where(MemoryJob.id == job_id)
    )
    row = initial_job_result.mappings().one_or_none()
    if row is None:
        return
    user_id = row["user_id"]
    conversation_id = row["conversation_id"]

    access = await user_repo.get_memory_access_for_worker(db, user_id)
    if access is None:
        return
    conversation = await conversation_repo.get_owned_for_update(
        db,
        conversation_id,
        user_id,
    )
    if conversation is None:
        return
    job = await memory_job_repo.get_processing_for_update(
        db,
        job_id,
        lease_until=lease_until,
    )
    if job is None:
        return

    if job.operation == "delete_episode":
        try:
            async with asyncio.timeout(_STORE_TIMEOUT_SECONDS):
                await store.adelete(
                    episode_namespace(job.user_id),
                    str(job.conversation_id),
                )
        except Exception as exc:
            raise _RetryableJobError("store_delete_failed") from exc
        _complete_locked(job, status="succeeded")
        await db.flush()
        return

    if job.operation != "extract_episode":
        raise _BlockedJobError("operation_unsupported")
    if (
        not settings.MEMORY_ENABLED
        or not access.enabled
        or access.account_deletion_pending
        or access.consent_version != job.consent_version
        or access.data_epoch != job.data_epoch
    ):
        _complete_locked(job, status="canceled", error_code="memory_gate_changed")
        await db.flush()
        return
    if (
        conversation.memory_crisis_seen
        or not conversation.memory_crisis_reviewed
        or conversation.memory_crisis_review_version
        != CRISIS_DETECTOR_VERSION
    ):
        _complete_locked(job, status="filtered", error_code="crisis_blocked")
        await db.flush()
        return
    if conversation.memory_revision > job.target_revision:
        _complete_locked(job, status="superseded", error_code="newer_revision")
        await db.flush()
        return
    if conversation.memory_revision < job.target_revision:
        raise _RetryableJobError("revision_not_ready")

    key_record = (
        await api_key_repo.get_owned_by_id(db, job.user_id, job.api_key_id)
        if job.api_key_id is not None
        else None
    )
    if key_record is None:
        _complete_locked(job, status="superseded", error_code="credential_missing")
        await db.flush()
        return
    if key_record.provider != job.provider or key_record.base_url != job.base_url:
        _complete_locked(job, status="superseded", error_code="credential_changed")
        await db.flush()
        return
    if not job.provider or not job.model:
        raise _BlockedJobError("provider_config_missing")

    existing = await _read_episode(
        store,
        user_id=job.user_id,
        conversation_id=job.conversation_id,
    )
    if existing is not None and existing.data_epoch > job.data_epoch:
        raise _BlockedJobError("store_epoch_ahead")
    if existing is not None and existing.data_epoch == job.data_epoch:
        grounded = await _episode_evidence_is_grounded(
            db,
            conversation_id=job.conversation_id,
            episode=existing,
        )
        if not grounded:
            if existing.target_revision >= job.target_revision:
                raise _BlockedJobError("store_evidence_invalid")
            existing = None
        elif existing.status != "active" or existing.crisis:
            if existing.target_revision >= job.target_revision:
                _complete_locked(
                    job,
                    status="superseded",
                    error_code="episode_inactive",
                )
                await db.flush()
                return
            existing = None
    if existing is not None and existing.data_epoch == job.data_epoch:
        if existing.target_revision == job.target_revision:
            _complete_locked(job, status="succeeded")
            await db.flush()
            return
        if existing.target_revision > job.target_revision:
            _complete_locked(job, status="superseded", error_code="newer_episode")
            await db.flush()
            return

    prior_value = (
        existing
        if existing is not None
        and existing.data_epoch == job.data_epoch
        and existing.status == "active"
        and not existing.crisis
        and existing.target_revision < job.target_revision
        else None
    )
    prior = (
        EpisodeDraft(claims=prior_value.claims, topics=prior_value.topics)
        if prior_value is not None
        else None
    )
    sources, full_source_contents = await _load_grounded_sources(
        db,
        job,
        prior,
        prior_revision=prior_value.target_revision if prior_value else 0,
    )

    try:
        api_key = decrypt_value(key_record.api_key_encrypted)
        llm = get_llm(
            job.provider,
            api_key=api_key,
            base_url=key_record.base_url,
            model=job.model,
            temperature=0.0,
        )
    except Exception as exc:
        raise _BlockedJobError("credential_unavailable") from exc

    outcome = await extract_episode_draft(
        llm,
        sources,
        prior_draft=prior,
    )
    if outcome.status.startswith("filtered_"):
        _complete_locked(
            job,
            status="filtered",
            error_code=outcome.status,
            input_tokens=outcome.input_tokens,
            output_tokens=outcome.output_tokens,
        )
        await db.flush()
        return
    if outcome.status != "accepted" or outcome.draft is None:
        raise _RetryableJobError("invalid_output")
    if not _draft_is_grounded_in_full_messages(
        outcome.draft,
        full_source_contents,
    ):
        raise _BlockedJobError("output_evidence_invalid")

    # Semantic promotion is additive: the episode below is the job's actual
    # deliverable, so a failure here is logged and skipped rather than costing
    # the user their episodic memory. The gates above already ran.
    try:
        semantic_candidates = derive_semantic_candidates(
            sources,
            max_content_chars=settings.MEMORY_SEMANTIC_ITEM_MAX_CHARS,
        )
        _, semantic_failures = await write_semantic_candidates(
            store,
            user_id=job.user_id,
            conversation_id=job.conversation_id,
            candidates=semantic_candidates,
            data_epoch=job.data_epoch,
            timeout_seconds=_STORE_TIMEOUT_SECONDS,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        semantic_failures = 1
        logger.warning(
            "Semantic promotion skipped; error_type=%s",
            type(exc).__name__,
        )
    if semantic_failures:
        logger.warning(
            "Semantic promotion incomplete; job=%s failures=%s",
            job.id,
            semantic_failures,
        )

    # Re-read the external value immediately before the idempotent write. Every
    # MindBridge writer holds the Conversation lock above, so revisions cannot
    # cross within this application even though Store has no compare-and-swap.
    latest = await _read_episode(
        store,
        user_id=job.user_id,
        conversation_id=job.conversation_id,
    )
    if latest is not None and latest.data_epoch > job.data_epoch:
        raise _BlockedJobError("store_epoch_ahead")
    if latest is not None and latest.data_epoch == job.data_epoch:
        if not await _episode_evidence_is_grounded(
            db,
            conversation_id=job.conversation_id,
            episode=latest,
        ):
            if latest.target_revision >= job.target_revision:
                raise _BlockedJobError("store_evidence_invalid")
            latest = None
        elif latest.status != "active" or latest.crisis:
            if latest.target_revision >= job.target_revision:
                _complete_locked(
                    job,
                    status="superseded",
                    error_code="episode_inactive",
                )
                await db.flush()
                return
            latest = None
    if latest is not None and latest.data_epoch == job.data_epoch:
        if latest.target_revision == job.target_revision:
            _complete_locked(job, status="succeeded")
            await db.flush()
            return
        if latest.target_revision > job.target_revision:
            _complete_locked(job, status="superseded", error_code="newer_episode")
            await db.flush()
            return

    value = EpisodeMemoryValue(
        conversation_id=str(job.conversation_id),
        claims=outcome.draft.claims,
        summary=outcome.draft.summary,
        topics=outcome.draft.topics,
        status="active",
        crisis=False,
        target_revision=job.target_revision,
        data_epoch=job.data_epoch,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    try:
        async with asyncio.timeout(_STORE_TIMEOUT_SECONDS):
            await store.aput(
                episode_namespace(job.user_id),
                str(job.conversation_id),
                value.model_dump(mode="json"),
                index=["summary"] if vector_enabled else False,
            )
    except Exception as exc:
        raise _RetryableJobError("store_write_failed") from exc
    _complete_locked(
        job,
        status="succeeded",
        input_tokens=outcome.input_tokens,
        output_tokens=outcome.output_tokens,
    )
    await db.flush()


async def process_claimed_job(
    session_factory: Any,
    store: Any,
    claimed: MemoryJob,
    *,
    vector_enabled: bool,
    max_attempts: int,
) -> None:
    """Process exactly one lease and persist only safe status codes."""
    lease_until = claimed.lease_until
    if lease_until is None:
        return
    processed = False
    try:
        async with session_factory() as db:
            async with db.begin():
                await _process_locked_job(
                    db,
                    store,
                    job_id=claimed.id,
                    lease_until=lease_until,
                    vector_enabled=vector_enabled,
                )
                processed = True
    except asyncio.CancelledError:
        raise
    except (EpisodeInputValidationError, StructuredOutputUnsupportedError) as exc:
        code = (
            "input_invalid"
            if isinstance(exc, EpisodeInputValidationError)
            else "structured_output_unsupported"
        )
        async with session_factory() as db:
            async with db.begin():
                await memory_job_repo.finish_claim(
                    db,
                    job_id=claimed.id,
                    lease_until=lease_until,
                    status="blocked",
                    error_code=code,
                )
    except _BlockedJobError as exc:
        async with session_factory() as db:
            async with db.begin():
                await memory_job_repo.finish_claim(
                    db,
                    job_id=claimed.id,
                    lease_until=lease_until,
                    status="blocked",
                    error_code=exc.code,
                )
    except (
        _RetryableJobError,
        EpisodeModelInvocationError,
        EpisodeModelTimeoutError,
    ) as exc:
        code = exc.code if isinstance(exc, _RetryableJobError) else "model_failed"
        async with session_factory() as db:
            async with db.begin():
                await memory_job_repo.retry_claim(
                    db,
                    job_id=claimed.id,
                    lease_until=lease_until,
                    max_attempts=max_attempts,
                    error_code=code,
                )
    except Exception as exc:
        logger.warning(
            "Memory job failed; job=%s error_type=%s",
            claimed.id,
            type(exc).__name__,
        )
        async with session_factory() as db:
            async with db.begin():
                await memory_job_repo.retry_claim(
                    db,
                    job_id=claimed.id,
                    lease_until=lease_until,
                    max_attempts=max_attempts,
                    error_code=(
                        "db_commit_failed" if processed else "worker_failed"
                    ),
                    count_failure=not processed,
                    force_retry=processed,
                )


async def run_memory_worker(
    session_factory: Any,
    store: Any,
    *,
    vector_enabled: bool,
    stop_event: asyncio.Event,
) -> None:
    """Poll and process one leased job at a time until application shutdown."""
    logger.info("Memory worker started.")
    while not stop_event.is_set():
        claimed: MemoryJob | None = None
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
                    store,
                    claimed,
                    vector_enabled=vector_enabled,
                    max_attempts=settings.MEMORY_WORKER_MAX_ATTEMPTS,
                )
                continue
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Memory worker poll failed; error_type=%s",
                type(exc).__name__,
            )

        try:
            await asyncio.wait_for(
                stop_event.wait(),
                timeout=settings.MEMORY_WORKER_POLL_SECONDS,
            )
        except TimeoutError:
            pass
    logger.info("Memory worker stopped.")
