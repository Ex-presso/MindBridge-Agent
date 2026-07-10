# app/api/__init__.py

from fastapi import APIRouter
from .v1.auth import router as auth_router
from .v1.chat import router as chat_router
from .v1.conversations import router as conversations_router
from .v1.api_keys import router as api_keys_router
from .v1.memory import router as memory_router

router = APIRouter()
router.include_router(auth_router, prefix="/v1")
router.include_router(chat_router, prefix="/v1", tags=["Chat"])
router.include_router(conversations_router, prefix="/v1")
router.include_router(api_keys_router, prefix="/v1")
router.include_router(memory_router, prefix="/v1")
