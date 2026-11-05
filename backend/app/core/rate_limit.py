"""Shared slowapi limiter.

Lives in its own module so both main.py (global default + exception handler)
and per-route decorators (e.g. stricter auth limits) use one instance.

Counters live in Redis when REDIS_URL is set, so every backend instance
enforces one shared limit and a restart does not reset it. Without Redis, or
while Redis is unreachable, each process counts in its own memory.
"""
from slowapi import Limiter
from slowapi.util import get_remote_address

from config.settings import settings

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[f"{settings.RATE_LIMIT_PER_MINUTE}/minute"],
    storage_uri=settings.REDIS_URL or "memory://",
    in_memory_fallback_enabled=True,
    key_prefix="mindbridge",
)
