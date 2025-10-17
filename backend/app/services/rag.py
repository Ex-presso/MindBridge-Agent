from __future__ import annotations

from threading import Lock
from typing import Optional

from langchain_core.vectorstores.base import VectorStoreRetriever

from app.core.rag.vector_store import VectorStore
from config.settings import settings

_vector_store: Optional[VectorStore] = None
_lock = Lock()


def _ensure_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    with _lock:
        if _vector_store is None:
            store = VectorStore()
            try:
                store.load()
            except FileNotFoundError:
                index_dir = settings.INDEX_DIR
                index_dir.parent.mkdir(parents=True, exist_ok=True)
                store.build()
            _vector_store = store
        return _vector_store


def get_retriever(k: int = 3) -> VectorStoreRetriever:
    """Return a cached retriever, building the FAISS index on first access if missing."""
    store = _ensure_vector_store()
    return store.get_retriever(k)
