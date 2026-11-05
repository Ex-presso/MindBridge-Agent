"""Wrappers that record how one operation went."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

from app.core.observability.metrics import LLM_CALL_SECONDS


@contextmanager
def observe_llm_call(operation: str) -> Iterator[None]:
    """Time one model call. Operation is chat, summarize, crisis or title."""
    started = time.perf_counter()
    outcome = "error"
    try:
        yield
        outcome = "ok"
    finally:
        LLM_CALL_SECONDS.labels(operation, outcome).observe(time.perf_counter() - started)
