"""OpenTelemetry tracing.

A trace records one request as a tree of timed spans. Tracing is off unless
OTEL_EXPORTER_OTLP_ENDPOINT is set. Until then `tracer` hands out no-op spans,
so instrumented code never has to ask whether tracing is on.

Spans carry sizes, counts and identifiers, never message text: the
conversations here are sensitive.
"""

from __future__ import annotations

from opentelemetry import trace

from config.settings import settings

tracer = trace.get_tracer("mindbridge")


def configure_tracing(app) -> None:
    """Export spans over OTLP/HTTP and open one server span per request."""
    if not settings.OTEL_EXPORTER_OTLP_ENDPOINT:
        return

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(
        resource=Resource.create({"service.name": settings.OTEL_SERVICE_NAME})
    )
    endpoint = settings.OTEL_EXPORTER_OTLP_ENDPOINT.rstrip("/")
    # Batching exports spans from a background thread, off the request path.
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces"))
    )
    trace.set_tracer_provider(provider)
    # Health checks and metric scrapes would bury the real requests, and the
    # per-message send/receive spans would add one span per streamed token.
    FastAPIInstrumentor.instrument_app(
        app,
        excluded_urls="health,metrics",
        exclude_spans=["send", "receive"],
    )


def current_trace_id() -> str:
    """Hex trace id of the active span, or "-" outside a recorded trace."""
    context = trace.get_current_span().get_span_context()
    return format(context.trace_id, "032x") if context.is_valid else "-"


def annotate_chat_span(conversation_id: str, provider: str, model: str) -> None:
    """Attach the turn's identifiers to the request span so traces are searchable."""
    trace.get_current_span().set_attributes(
        {
            "conversation.id": conversation_id,
            "llm.provider": provider,
            "llm.model": model,
        }
    )
