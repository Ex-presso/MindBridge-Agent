"""Shared slowapi limiter.

Lives in its own module so both main.py (global default + exception handler)
and per-route decorators (e.g. stricter auth limits) use one instance.
"""
from slowapi import Limiter
from slowapi.util import get_remote_address

from config.settings import settings

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[f"{settings.RATE_LIMIT_PER_MINUTE}/minute"],
)
