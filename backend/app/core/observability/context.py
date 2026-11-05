"""Request correlation: one id per request, stamped on every log line.

The id lives in a ContextVar, so it follows the request through every await
and into tasks the request spawns, without being passed as an argument.
"""

from __future__ import annotations

import logging
import re
import uuid
from contextvars import ContextVar, Token

from app.core.observability.tracing import current_trace_id

_HEADER = b"x-request-id"
# A caller-supplied id is echoed into logs, so only accept a plain token.
_SAFE_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")

_request_id: ContextVar[str] = ContextVar("request_id", default="-")


def current_request_id() -> str:
    return _request_id.get()


def bind_request_id(value: str) -> Token[str]:
    """Set the id for work that does not start from an HTTP request."""
    return _request_id.set(value)


def reset_request_id(token: Token[str]) -> None:
    _request_id.reset(token)


class RequestIdLogFilter(logging.Filter):
    """Copies the request id and trace id onto each record for the formatter.

    The trace id is what you paste into the tracing UI to open the request
    that produced a log line.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        record.trace_id = current_trace_id()
        return True


class RequestIdMiddleware:
    """Assigns the request id and returns it in the X-Request-ID header.

    An id sent by the caller is reused, so one id can span a proxy and this
    service. Written as plain ASGI so a streamed response body is still
    produced inside the request's context.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = dict(scope["headers"]).get(_HEADER, b"").decode("latin-1")
        request_id = incoming if _SAFE_ID.fullmatch(incoming) else uuid.uuid4().hex
        token = _request_id.set(request_id)

        async def send_with_id(message):
            if message["type"] == "http.response.start":
                message["headers"] = [*message.get("headers", []), (_HEADER, request_id.encode())]
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            _request_id.reset(token)
