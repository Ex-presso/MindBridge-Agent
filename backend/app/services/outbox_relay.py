"""Moves events from the outbox table to RabbitMQ.

Writing to the database and publishing to a broker are two systems with no
shared transaction. If the chat request did both, a crash between them would
either lose the job or announce a turn that was rolled back. Instead the
request only inserts an outbox row, in the transaction that saves the turn,
and this relay publishes it afterwards.

The row is deleted only after the broker confirms the message. A crash before
that leaves the row to be published again, so delivery is at least once and
the consumer has to tolerate duplicates.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.core.messaging import JobQueue
from app.db.repositories import outbox_repo

logger = logging.getLogger(__name__)

_BATCH = 50
# Fallback for events written by another instance, or missed while down.
_POLL_SECONDS = 5.0
_wake = asyncio.Event()


def notify() -> None:
    """Tell the relay that a transaction has just committed new events."""
    _wake.set()


async def publish_waiting(session_factory: Any, queue: JobQueue) -> int:
    """Publish one batch of waiting events and return how many were sent."""
    async with session_factory() as db:
        async with db.begin():
            events = await outbox_repo.lock_oldest(db, _BATCH)
            for event in events:
                await queue.publish(event.payload, message_id=str(event.id))
            await outbox_repo.delete_many(db, [event.id for event in events])
    return len(events)


async def run_outbox_relay(session_factory: Any, queue: JobQueue) -> None:
    """Publish events as they appear, until cancelled."""
    while True:
        _wake.clear()
        published = await publish_waiting(session_factory, queue)
        if published == _BATCH:
            continue  # more may be waiting
        try:
            await asyncio.wait_for(_wake.wait(), timeout=_POLL_SECONDS)
        except TimeoutError:
            pass
