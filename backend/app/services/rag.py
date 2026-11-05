from __future__ import annotations

import asyncio
from typing import Optional

from langchain_core.vectorstores.base import VectorStoreRetriever

from app.core.rag.vector_store import VectorStore

_vector_store: Optional[VectorStore] = None
_lock = asyncio.Lock()


def _load_store_sync() -> VectorStore:
    """Blocking I/O, runs in a thread executor.

    The agent tool calls retriever.ainvoke(), so pgvector needs async_mode=True.
    """
    store = VectorStore(async_mode=True)
    store.load()
    return store


async def _ensure_vector_store() -> VectorStore:
    global _vector_store
    store = _vector_store
    if store is not None:
        return store

    async with _lock:
        store = _vector_store
        if store is None:
            loop = asyncio.get_event_loop()
            store = await loop.run_in_executor(None, _load_store_sync)
            _vector_store = store
    return store


async def get_retriever(k: int | None = None) -> VectorStoreRetriever:
    """Return a cached retriever. Safe to call concurrently."""
    store = await _ensure_vector_store()
    return store.get_retriever(k)
