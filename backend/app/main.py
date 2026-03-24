"""MindBridge backend — FastAPI application with LangGraph checkpointing."""
import logging
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.api import router
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
        root_logger.addHandler(handler)


configure_logging()
logger = logging.getLogger(__name__)

# ── Rate limiting ─────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=[f"{settings.RATE_LIMIT_PER_MINUTE}/minute"])


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting MindBridge backend...")

    # 1. Create DB tables (SQLAlchemy create_all; Alembic used for migrations in prod)
    from app.db.engine import engine, Base, AsyncSessionLocal
    from app.db.models import User, Conversation, Message  # ensure models are registered
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    app.state.db_session = AsyncSessionLocal

    # 2. Set up LangGraph PostgreSQL checkpointer
    from psycopg_pool import AsyncConnectionPool
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    pool = AsyncConnectionPool(
        conninfo=settings.CHECKPOINT_DATABASE_URL,
        max_size=20,
        open=False,
    )
    await pool.open()
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()  # creates langgraph checkpoint tables
    app.state.checkpointer = checkpointer
    app.state.pg_pool = pool

    # 3. Create agents (stateless for legacy endpoint, stateful for session endpoint)
    from app.core.agent.agent import Agent
    app.state.stateless_agents = {
        "openai": Agent("openai", checkpointer=None),
        "google_genai": Agent("google_genai", checkpointer=None),
    }
    app.state.stateful_agents = {
        "openai": Agent("openai", checkpointer=checkpointer),
        "google_genai": Agent("google_genai", checkpointer=checkpointer),
    }

    logger.info("MindBridge backend ready.")
    yield

    # Cleanup
    logger.info("Shutting down...")
    await pool.close()


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MindBridge API",
    description="Mental health chatbot with Rogerian therapy and RAG",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
    expose_headers=["X-Conversation-Id"],
    max_age=86400,
)

app.include_router(router, prefix="/api")


@app.get("/", status_code=200, tags=["Health"])
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
async def health_detail(request: Request):
    return {
        "status": "ok",
        "database": "connected",
        "checkpointer": "ready",
        "agents": list(request.app.state.stateful_agents.keys()),
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
