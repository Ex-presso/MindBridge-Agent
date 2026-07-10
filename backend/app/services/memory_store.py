"""Initialize the durable memory Store with an optional vector index."""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langgraph.store.postgres.aio import AsyncPostgresStore

from app.core.rag.vector_store import get_embeddings
from config.settings import settings


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemoryStoreRuntime:
    """Store plus the retrieval capability that actually initialized."""

    store: Any
    vector_enabled: bool


async def initialize_memory_store(
    pool: Any,
    *,
    store_factory: Callable[..., Any] = AsyncPostgresStore,
    embedding_factory: Callable[[], Any] = get_embeddings,
) -> MemoryStoreRuntime:
    """Create the Store, degrading vector failure to safe key-value access.

    Inspection and deletion must remain available even if the embedding model
    or pgvector index cannot initialize. A plain Store preserves those privacy
    controls and semantic fact lookup; episode similarity is disabled rather
    than silently replaced with unrelated recency results.
    """
    if settings.MEMORY_ENABLED:
        try:
            embeddings = await asyncio.to_thread(embedding_factory)
            indexed_store = store_factory(
                pool,
                index={
                    "dims": settings.MEMORY_EMBED_DIMS,
                    "embed": embeddings,
                    "fields": ["summary"],
                    "distance_type": "cosine",
                    "ann_index_config": {"kind": "hnsw"},
                },
            )
            await indexed_store.setup()
            return MemoryStoreRuntime(
                store=indexed_store,
                vector_enabled=True,
            )
        except Exception as exc:
            logger.warning(
                "Memory vector index unavailable; using key-value Store (%s).",
                type(exc).__name__,
            )

    store = store_factory(pool)
    await store.setup()
    return MemoryStoreRuntime(store=store, vector_enabled=False)
