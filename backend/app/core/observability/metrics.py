"""Prometheus metrics, scraped from /metrics.

Every label takes values from a small fixed set: route templates, provider
names, outcome strings. A label fed by user input, such as a model name, would
create one time series per distinct value and grow without bound; that detail
goes on trace spans instead.
"""

from __future__ import annotations

import time

from prometheus_client import Histogram

HTTP_REQUEST_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency, including the time to stream the body.",
    ["method", "route", "status"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30, 60, 120),
)


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
