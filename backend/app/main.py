"""MindBridge backend — FastAPI application with LangGraph checkpointing."""
import asyncio
import logging
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api import router
from app.core.observability.context import RequestIdLogFilter, RequestIdMiddleware
from app.core.observability.metrics import MetricsMiddleware
from app.core.observability.tracing import configure_tracing
from app.core.rate_limit import limiter
from config.settings import settings


# ── Logging ───────────────────────────────────────────────────────────────────
def configure_logging() -> None:
    """Configure application-wide logging once at startup."""
    root_logger = logging.getLogger()
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    if root_logger.handlers:
        root_logger.setLevel(level)
        return

    formatter = logging.Formatter(settings.LOG_FORMAT)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    handlers: list[logging.Handler] = [stream_handler]

    if settings.LOG_FILE:
        log_path = Path(settings.PROJECT_ROOT) / settings.LOG_FILE
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=settings.LOG_MAX_BYTES,
            backupCount=settings.LOG_BACKUP_COUNT,
        )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    root_logger.setLevel(level)
    for handler in handlers:
        # On the handler, not the logger, so records from library loggers get
        # the request id too.
        handler.addFilter(RequestIdLogFilter())
        root_logger.addHandler(handler)


def configure_langsmith() -> None:
    """Export LangSmith settings to os.environ so LangChain/LangGraph pick them up.

    LangChain reads tracing config from the process environment, but this app
    loads config via pydantic .env (which does not populate os.environ), so we
    bridge it here. Works for both local dev and Docker. No-op when disabled.
    """
    if not settings.LANGSMITH_TRACING:
        return
    import os
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
    os.environ["LANGSMITH_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
    if settings.LANGSMITH_API_KEY:
        os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY.get_secret_value()
    else:
        logging.getLogger(__name__).warning(
            "LANGSMITH_TRACING is on but LANGSMITH_API_KEY is unset; traces won't be sent."
        )


configure_logging()
configure_langsmith()
logger = logging.getLogger(__name__)

# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting MindBridge backend...")

    # 1. Dev/Docker convenience: adopt known legacy create_all databases and
    #    apply Alembic migrations. Production keeps migrations as an explicit
    #    deployment step by setting AUTO_CREATE_TABLES=False.
    from app.db.engine import AsyncSessionLocal, engine
    if settings.AUTO_CREATE_TABLES:
        from app.db.schema_migrations import ensure_schema_at_head

        plan = await ensure_schema_at_head(engine)
        logger.info("Database schema is at Alembic head (%s).", plan.reason)
    else:
        logger.info(
            "AUTO_CREATE_TABLES=False — skipping automatic Alembic upgrade "
            "(expecting a manually migrated schema)."
        )
    app.state.db_session = AsyncSessionLocal

    # 2. Set up LangGraph PostgreSQL checkpointer
    from psycopg_pool import AsyncConnectionPool
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    pool = AsyncConnectionPool(
        conninfo=settings.CHECKPOINT_DATABASE_URL,
        max_size=20,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    )
    await pool.open()
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()  # creates langgraph checkpoint tables
    app.state.checkpointer = checkpointer
    app.state.pg_pool = pool

    # 2b. Background memory jobs: the outbox relay and the RabbitMQ consumer.
    memory_pipeline_task = None
    if settings.MEMORY_WORKER_ENABLED and settings.RABBITMQ_URL:
        from app.services.memory_worker import run_memory_pipeline

        memory_pipeline_task = asyncio.create_task(
            run_memory_pipeline(AsyncSessionLocal),
            name="memory-pipeline",
        )
    elif settings.MEMORY_ENABLED:
        logger.warning(
            "MEMORY_ENABLED is on but RABBITMQ_URL is unset: memory jobs will "
            "wait in the outbox and nothing new will be remembered."
        )

    logger.info("MindBridge backend ready.")
    try:
        yield
    finally:
        logger.info("Shutting down...")
        if memory_pipeline_task is not None:
            memory_pipeline_task.cancel()
            await asyncio.gather(memory_pipeline_task, return_exceptions=True)
        await pool.close()


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MindBridge API",
    description="Mental health chatbot with Rogerian therapy and RAG",
    version="1.0.0",
    lifespan=lifespan,
)

configure_tracing(app)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
    expose_headers=["X-Conversation-Id", "X-Request-ID"],
    max_age=86400,
)
app.add_middleware(MetricsMiddleware)
# Added last so it wraps everything above, including CORS preflight handling.
app.add_middleware(RequestIdMiddleware)

app.include_router(router, prefix="/api")


@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    """Scraped by Prometheus. Unauthenticated, so keep it on an internal network."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/", status_code=200, tags=["Health"])
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
async def health_detail():
    return {
        "status": "ok",
        "database": "connected",
        "checkpointer": "ready",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if not settings.DEBUG else "debug",
    )
