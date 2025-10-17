"""
Mental Health Chatbot backend server
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import router
from contextlib import asynccontextmanager
from config.settings import settings
import uvicorn


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

# health check
@app.get("/", status_code=200, tags=["Health"])
def root():
    return {"status": "ok"}


# when you use: `python -m app/main.py` to start the server, it will run here
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if not settings.DEBUG else "debug"
    )
