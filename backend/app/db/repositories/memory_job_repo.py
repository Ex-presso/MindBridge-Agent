import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import and_, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.memory_job import MemoryJob

UNFINISHED_STATUSES = frozenset({"pending", "processing", "retry"})


async def enqueue_extract_episode(
    db: AsyncSession,
    *,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID,
    target_revision: int,
    consent_version: int,
    data_epoch: int,
    source_user_message_id: uuid.UUID,
    source_assistant_message_id: uuid.UUID,
    api_key_id: uuid.UUID,
    provider: str,
    model: str,
    base_url: str | None,
) -> MemoryJob:
    """Enqueue one completed turn in the assistant persistence transaction."""
    job = MemoryJob(
        operation="extract_episode",
        user_id=user_id,
        conversation_id=conversation_id,
        target_revision=target_revision,
        consent_version=consent_version,
        data_epoch=data_epoch,
        source_user_message_id=source_user_message_id,
        source_assistant_message_id=source_assistant_message_id,
        api_key_id=api_key_id,
        provider=provider,
        model=model,
        base_url=base_url,
        status="pending",
        priority=100,
        dedupe_key=(
            f"extract_episode:{user_id}:{conversation_id}:{target_revision}"
        ),
    )
    db.add(job)
    await db.flush()
    return job


async def enqueue_delete_episode(
    db: AsyncSession,
    *,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID,
    target_revision: int,
    consent_version: int,
    data_epoch: int,
) -> MemoryJob:
    """Durably request physical removal after a sticky crisis transition."""
    job = MemoryJob(
        operation="delete_episode",
        user_id=user_id,
        conversation_id=conversation_id,
        target_revision=target_revision,
        consent_version=consent_version,
        data_epoch=data_epoch,
        status="pending",
        priority=0,
        dedupe_key=f"delete_episode:crisis:{user_id}:{conversation_id}",
    )
    db.add(job)
    await db.flush()
    return job


async def cancel_unfinished_for_user(
    db: AsyncSession,
    user_id: uuid.UUID,
) -> int:
    """Cancel queued or leased jobs before clearing a user's memory data."""
    result = await db.execute(
        update(MemoryJob)
        .where(
            MemoryJob.user_id == user_id,
            MemoryJob.status.in_(UNFINISHED_STATUSES),
        )
        .values(
            status="canceled",
            lease_until=None,
            updated_at=datetime.now(timezone.utc),
        )
    )
    return int(result.rowcount or 0)


async def claim_next(
    db: AsyncSession,
    *,
    lease_seconds: int,
    max_attempts: int,
) -> MemoryJob | None:
    """Lease the next due job without blocking another worker."""
    now = datetime.now(timezone.utc)
    result = await db.execute(
        select(MemoryJob)
        .where(
            or_(
                and_(
                    MemoryJob.status.in_(("pending", "retry")),
                    MemoryJob.available_at <= now,
                    or_(
                        MemoryJob.operation == "delete_episode",
                        MemoryJob.attempts < max_attempts,
                    ),
                ),
                and_(
                    MemoryJob.status == "processing",
                    MemoryJob.lease_until <= now,
                ),
            )
        )
        .order_by(
            MemoryJob.priority.asc(),
            MemoryJob.available_at.asc(),
            MemoryJob.created_at.asc(),
        )
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    job = result.scalar_one_or_none()
    if job is None:
        return None
    if job.operation != "delete_episode" and job.attempts >= max_attempts:
        job.status = "blocked"
        job.lease_until = None
        job.error_code = "attempts_exhausted"
        job.updated_at = now
        await db.flush()
        return None
    job.status = "processing"
    job.lease_until = now + timedelta(seconds=lease_seconds)
    job.error_code = None
    job.updated_at = now
    await db.flush()
    return job


async def get_processing_for_update(
    db: AsyncSession,
    job_id: uuid.UUID,
    *,
    lease_until: datetime,
) -> MemoryJob | None:
    """Lock a lease only when its timestamp token still belongs to this worker."""
    result = await db.execute(
        select(MemoryJob)
        .where(
            MemoryJob.id == job_id,
            MemoryJob.status == "processing",
            MemoryJob.lease_until == lease_until,
        )
        .with_for_update()
    )
    return result.scalar_one_or_none()


async def list_extract_source_user_ids(
    db: AsyncSession,
    *,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID,
    consent_version: int,
    data_epoch: int,
    after_revision: int,
    through_revision: int,
    limit: int = 40,
) -> tuple[uuid.UUID, ...]:
    """Return consented successful-turn sources needed for a rolling draft."""
    result = await db.execute(
        select(MemoryJob.source_user_message_id)
        .where(
            MemoryJob.operation == "extract_episode",
            MemoryJob.user_id == user_id,
            MemoryJob.conversation_id == conversation_id,
            MemoryJob.consent_version == consent_version,
            MemoryJob.data_epoch == data_epoch,
            MemoryJob.target_revision > after_revision,
            MemoryJob.target_revision <= through_revision,
            MemoryJob.source_user_message_id.is_not(None),
            MemoryJob.status.in_(
                ("pending", "processing", "retry", "succeeded", "superseded")
            ),
        )
        .order_by(MemoryJob.target_revision.desc())
        .limit(limit)
    )
    return tuple(reversed(result.scalars().all()))


async def finish_claim(
    db: AsyncSession,
    *,
    job_id: uuid.UUID,
    lease_until: datetime,
    status: str,
    error_code: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> bool:
    """Finish the exact leased attempt with content-safe metadata."""
    job = await get_processing_for_update(
        db,
        job_id,
        lease_until=lease_until,
    )
    if job is None:
        return False
    job.status = status
    job.lease_until = None
    job.error_code = error_code
    job.input_tokens = input_tokens
    job.output_tokens = output_tokens
    job.updated_at = datetime.now(timezone.utc)
    await db.flush()
    return True


async def retry_claim(
    db: AsyncSession,
    *,
    job_id: uuid.UUID,
    lease_until: datetime,
    max_attempts: int,
    error_code: str,
    count_failure: bool = True,
    force_retry: bool = False,
) -> bool:
    """Retry a transient failure or block it after the bounded attempt count."""
    job = await get_processing_for_update(
        db,
        job_id,
        lease_until=lease_until,
    )
    if job is None:
        return False
    now = datetime.now(timezone.utc)
    if count_failure:
        job.attempts += 1
    if (
        job.operation != "delete_episode"
        and not force_retry
        and job.attempts >= max_attempts
    ):
        job.status = "blocked"
        job.available_at = now
    else:
        job.status = "retry"
        delay_seconds = min(300, 5 * (2 ** min(max(job.attempts - 1, 0), 6)))
        job.available_at = now + timedelta(seconds=delay_seconds)
    job.lease_until = None
    job.error_code = error_code[:64]
    job.updated_at = now
    await db.flush()
    return True
