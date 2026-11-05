"""Prometheus metrics, scraped from /metrics.

Every label takes values from a small fixed set: route templates, provider
names, outcome strings. A label fed by user input, such as a model name, would
create one time series per distinct value and grow without bound; that detail
goes on trace spans instead.
"""

from __future__ import annotations

import time

from prometheus_client import Counter, Histogram

# Model calls and streamed turns run for seconds, not the milliseconds that the
# default buckets are spaced for.
_SLOW_BUCKETS = (0.25, 0.5, 1, 2, 4, 8, 15, 30, 60, 120)

HTTP_REQUEST_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency, including the time to stream the body.",
    ["method", "route", "status"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30, 60, 120),
)
CHAT_FIRST_TOKEN_SECONDS = Histogram(
    "chat_first_token_seconds",
    "Time from the start of a chat request to its first streamed token.",
    ["provider"],
    buckets=_SLOW_BUCKETS,
)
CHAT_TURNS = Counter(
    "chat_turns_total",
    "Chat turns by outcome.",
    ["provider", "outcome"],
)
LLM_TOKENS = Counter(
    "llm_tokens_total",
    "Tokens reported by the model provider for chat turns.",
    ["provider", "direction"],
)
LLM_CALL_SECONDS = Histogram(
    "llm_call_duration_seconds",
    "Latency of one model call made by the agent.",
    ["operation", "outcome"],
    buckets=_SLOW_BUCKETS,
)
CRISIS_TURNS = Counter(
    "crisis_turns_total",
    "User turns routed to the crisis path.",
    ["category"],
)
RAG_SEARCH_SECONDS = Histogram(
    "rag_search_duration_seconds",
    "Knowledge base search latency: query embedding plus the vector scan.",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 4),
)
RAG_SEARCHES = Counter(
    "rag_searches_total",
    "Knowledge base searches by outcome: served, empty or unavailable.",
    ["outcome"],
)
# Shows where live traffic sits relative to RAG_MIN_SIMILARITY.
RAG_TOP_SIMILARITY = Histogram(
    "rag_top_similarity",
    "Similarity of the best match, before the floor is applied.",
    buckets=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
)
MEMORY_JOBS = Counter(
    "memory_jobs_total",
    "Background memory jobs by outcome; error means the attempt will be retried.",
    ["operation", "outcome"],
)
# How far the worker is behind: a rising lag means jobs arrive faster than
# they are processed.
MEMORY_JOB_LAG_SECONDS = Histogram(
    "memory_job_lag_seconds",
    "Time from a turn being saved to its memory job starting.",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 300, 1800),
)
MEMORY_JOB_SECONDS = Histogram(
    "memory_job_duration_seconds",
    "Time to process one background memory job.",
    ["operation"],
    buckets=_SLOW_BUCKETS,
)


def record_chat_turn(provider: str, outcome: str, usage: dict) -> None:
    """Count one finished chat turn and the tokens the provider reported for it."""
    CHAT_TURNS.labels(provider, outcome).inc()
    for direction in ("input", "output"):
        LLM_TOKENS.labels(provider, direction).inc(usage.get(direction, 0))


class MetricsMiddleware:
    """Times every HTTP request and labels it with the matched route template.

    The template (/api/v1/conversations/{conversation_id}) keeps the label set
    bounded; the raw path would add a series per conversation id.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        # The scrape endpoint itself is not worth a time series.
        if scope["type"] != "http" or scope["path"] == "/metrics":
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        status = 500

        async def send_and_record_status(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_and_record_status)
        finally:
            # The router stores the matched route on the scope while handling it.
            route = getattr(scope.get("route"), "path", "unmatched")
            HTTP_REQUEST_SECONDS.labels(scope["method"], route, str(status)).observe(
                time.perf_counter() - started
            )
