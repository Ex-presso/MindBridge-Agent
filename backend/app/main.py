"""
Mental Health Chatbot backend server
"""

from fastapi import FastAPI
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
