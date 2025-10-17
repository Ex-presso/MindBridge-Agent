"""
Mental Health Chatbot backend server
"""
from contextlib import asynccontextmanager
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.api import router
from config.settings import settings


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Mental Health Chatbot",
    description="A simple Mental Health Chatbot API",
    version="0.1.0",
    lifespan=lifespan,
)


allow_origins = settings.CORS_ORIGINS


app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
    expose_headers=["*"],
    max_age=86400,
)

app.include_router(router, prefix="/api")


@app.get("/", status_code=200, tags=["Health"])
def root():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if not settings.DEBUG else "debug",
    )
