# app/api/__init__.py

from fastapi import APIRouter
from .v1.chat import router as chat_router
from .v1.model import router as model_router

router = APIRouter()
router.include_router(chat_router, prefix="/v1", tags=["Chat"])
router.include_router(model_router, prefix="/v1", tags=["Model"])