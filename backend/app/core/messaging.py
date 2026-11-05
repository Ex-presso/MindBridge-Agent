"""RabbitMQ queues for background jobs.

One direct exchange, four queues:

    job        -> memory.jobs             read by the worker
    retry.5s   -> memory.jobs.retry.5s    no consumer, expires back to "job"
    retry.30s  -> memory.jobs.retry.30s   no consumer, expires back to "job"
    dead       -> memory.jobs.dead        parked for a person to inspect

A failed job is republished to a retry queue. Nothing consumes that queue and
its messages carry a time to live; when it runs out RabbitMQ dead-letters the
message back to the work queue. That gives a delayed retry without a plugin
and without a worker that sleeps while holding the message. There is one
queue per delay because RabbitMQ only expires the message at the head of a
queue, so mixed delays in one queue would hold each other up.

Delivery is at least once: a message is redelivered if the worker dies before
acknowledging it, so the handler must tolerate seeing the same job twice.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import aio_pika
from aio_pika.abc import AbstractIncomingMessage

EXCHANGE = "mindbridge.memory"
WORK_QUEUE = "memory.jobs"
DEAD_QUEUE = "memory.jobs.dead"
# Delay before the second attempt, the third attempt, and every later one.
RETRY_DELAYS_SECONDS = (5, 30)
ATTEMPT_HEADER = "x-attempt"


def attempt_of(message: AbstractIncomingMessage) -> int:
    """Which attempt this delivery is, starting at 1."""
    return int((message.headers or {}).get(ATTEMPT_HEADER, 1))


class JobQueue:
    """A connection with the topology declared on it."""

    def __init__(self, connection, channel, exchange, work_queue):
        self._connection = connection
        self._channel = channel
        self._exchange = exchange
        self._work_queue = work_queue

    @classmethod
    async def connect(cls, url: str) -> "JobQueue":
        connection = await aio_pika.connect_robust(url)
        # Publisher confirms are on by default: publish() returns only after
        # the broker has taken responsibility for the message.
        channel = await connection.channel()
        # One unacknowledged message at a time: jobs run one after another.
        await channel.set_qos(prefetch_count=1)
        exchange = await channel.declare_exchange(
            EXCHANGE, aio_pika.ExchangeType.DIRECT, durable=True
        )

        work_queue = await channel.declare_queue(
            WORK_QUEUE,
            durable=True,
            # A rejected message goes to the dead queue instead of vanishing.
            arguments={"x-dead-letter-exchange": EXCHANGE, "x-dead-letter-routing-key": "dead"},
        )
        await work_queue.bind(exchange, routing_key="job")

        for delay in RETRY_DELAYS_SECONDS:
            retry_queue = await channel.declare_queue(
                f"{WORK_QUEUE}.retry.{delay}s",
                durable=True,
                arguments={
                    "x-message-ttl": delay * 1000,
                    "x-dead-letter-exchange": EXCHANGE,
                    "x-dead-letter-routing-key": "job",
                },
            )
            await retry_queue.bind(exchange, routing_key=f"retry.{delay}s")

        dead_queue = await channel.declare_queue(DEAD_QUEUE, durable=True)
        await dead_queue.bind(exchange, routing_key="dead")
        return cls(connection, channel, exchange, work_queue)

    async def publish(
        self,
        body: dict[str, Any],
        *,
        message_id: str,
        headers: dict[str, Any] | None = None,
        routing_key: str = "job",
    ) -> None:
        """Publish a job and wait for the broker to confirm it."""
        await self._exchange.publish(
            aio_pika.Message(
                body=json.dumps(body).encode(),
                message_id=message_id,
                content_type="application/json",
                # Written to disk, so it survives a broker restart.
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                headers=headers or {},
            ),
            routing_key=routing_key,
            # Fail loudly if no queue is bound, rather than drop the message.
            mandatory=True,
        )

    async def retry_later(
        self,
        message: AbstractIncomingMessage,
        *,
        count_attempt: bool = True,
    ) -> None:
        """Send the job to a retry queue, then ack this delivery.

        Publishing first means a crash in between leaves a duplicate, never a
        lost job. `count_attempt=False` postpones the job without using up one
        of its attempts.
        """
        attempt = attempt_of(message)
        delay = RETRY_DELAYS_SECONDS[min(attempt, len(RETRY_DELAYS_SECONDS)) - 1]
        if not count_attempt:
            attempt -= 1
        await self._exchange.publish(
            aio_pika.Message(
                body=message.body,
                message_id=message.message_id,
                content_type=message.content_type,
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                headers={**(message.headers or {}), ATTEMPT_HEADER: attempt + 1},
            ),
            routing_key=f"retry.{delay}s",
            mandatory=True,
        )
        await message.ack()

    async def bury(self, message: AbstractIncomingMessage) -> None:
        """Give up on the job: the broker moves it to the dead queue."""
        await message.reject(requeue=False)

    async def messages(self) -> AsyncIterator[AbstractIncomingMessage]:
        """Yield jobs one at a time. The caller must ack, retry or bury each."""
        async with self._work_queue.iterator() as iterator:
            async for message in iterator:
                yield message

    async def close(self) -> None:
        await self._connection.close()
