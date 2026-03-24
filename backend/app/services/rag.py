from __future__ import annotations

import asyncio
from typing import Optional

from langchain_core.vectorstores.base import VectorStoreRetriever

from app.core.rag.vector_store import VectorStore
from config.settings import settings

_vector_store: Optional[VectorStore] = None
_lock = asyncio.Lock()


def _load_store_sync() -> VectorStore:
    """Blocking I/O — runs in thread executor."""
    store = VectorStore(backend="pgvector")
    try:
        store.load()
    except Exception:
        store = VectorStore(backend="faiss")
        try:
            store.load()
        except FileNotFoundError:
            store.build()
    return store


async def _ensure_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    async with _lock:
        if _vector_store is None:
            loop = asyncio.get_event_loop()
            _vector_store = await loop.run_in_executor(None, _load_store_sync)
    return _vector_store


async def get_retriever(k: int | None = None) -> VectorStoreRetriever:
    """Return a cached retriever. Safe to call concurrently."""
    store = await _ensure_vector_store()
    return store.get_retriever(k)
