"""Wrappers that record how one operation went."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

from opentelemetry.trace import Span, StatusCode

from app.core.observability.metrics import LLM_CALL_SECONDS
from app.core.observability.tracing import tracer


@contextmanager
def observe_llm_call(operation: str) -> Iterator[Span]:
    """Time one model call and open a span for it.

    Operation is chat, summarize, crisis or title. The span never records the
    exception itself: provider errors can echo the whole prompt.
    """
    started = time.perf_counter()
    outcome = "error"
    with tracer.start_as_current_span(
        f"llm.{operation}",
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        try:
            yield span
            outcome = "ok"
        finally:
            if outcome == "error":
                span.set_status(StatusCode.ERROR)
            LLM_CALL_SECONDS.labels(operation, outcome).observe(time.perf_counter() - started)
