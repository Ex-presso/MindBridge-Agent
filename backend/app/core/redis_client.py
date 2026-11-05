"""Shared async Redis client."""

from __future__ import annotations

from redis.asyncio import Redis

from config.settings import settings

_client: Redis | None = None


def get_redis() -> Redis | None:
    """Return the process-wide client, or None when REDIS_URL is unset."""
    global _client
    if settings.REDIS_URL is None:
        return None
    if _client is None:
        _client = Redis.from_url(settings.REDIS_URL, decode_responses=True)
    return _client
