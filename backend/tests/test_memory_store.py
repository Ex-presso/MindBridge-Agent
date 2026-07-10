"""Memory Store initialization keeps privacy controls available on fallback."""

import asyncio

from app.services import memory_store


class _FakeStore:
    instances = []
    fail_indexed_setup = False

    def __init__(self, pool, *, index=None):
        self.pool = pool
        self.index_config = index
        self.setup_calls = 0
        type(self).instances.append(self)

    async def setup(self):
        self.setup_calls += 1
        if self.index_config and type(self).fail_indexed_setup:
            raise RuntimeError("vector DDL unavailable")


def _reset_store():
    _FakeStore.instances = []
    _FakeStore.fail_indexed_setup = False


def test_global_off_initializes_kv_without_loading_embeddings(monkeypatch):
    _reset_store()
    monkeypatch.setattr(memory_store.settings, "MEMORY_ENABLED", False)
    embedding_calls = []

    runtime = asyncio.run(
        memory_store.initialize_memory_store(
            object(),
            store_factory=_FakeStore,
            embedding_factory=lambda: embedding_calls.append(True),
        )
    )

    assert runtime.vector_enabled is False
    assert runtime.store.index_config is None
    assert runtime.store.setup_calls == 1
    assert embedding_calls == []


def test_global_on_initializes_summary_vector_index(monkeypatch):
    _reset_store()
    monkeypatch.setattr(memory_store.settings, "MEMORY_ENABLED", True)
    monkeypatch.setattr(memory_store.settings, "MEMORY_EMBED_DIMS", 1024)
    embeddings = object()

    runtime = asyncio.run(
        memory_store.initialize_memory_store(
            "pool",
            store_factory=_FakeStore,
            embedding_factory=lambda: embeddings,
        )
    )

    assert runtime.vector_enabled is True
    assert runtime.store.pool == "pool"
    assert runtime.store.setup_calls == 1
    assert runtime.store.index_config == {
        "dims": 1024,
        "embed": embeddings,
        "fields": ["summary"],
        "distance_type": "cosine",
        "ann_index_config": {"kind": "hnsw"},
    }


def test_embedding_failure_falls_back_to_kv(monkeypatch):
    _reset_store()
    monkeypatch.setattr(memory_store.settings, "MEMORY_ENABLED", True)

    def fail_embeddings():
        raise RuntimeError("model unavailable")

    runtime = asyncio.run(
        memory_store.initialize_memory_store(
            "pool",
            store_factory=_FakeStore,
            embedding_factory=fail_embeddings,
        )
    )

    assert runtime.vector_enabled is False
    assert len(_FakeStore.instances) == 1
    assert runtime.store.index_config is None
    assert runtime.store.setup_calls == 1


def test_vector_setup_failure_retries_plain_store(monkeypatch):
    _reset_store()
    _FakeStore.fail_indexed_setup = True
    monkeypatch.setattr(memory_store.settings, "MEMORY_ENABLED", True)

    runtime = asyncio.run(
        memory_store.initialize_memory_store(
            "pool",
            store_factory=_FakeStore,
            embedding_factory=object,
        )
    )

    assert runtime.vector_enabled is False
    assert len(_FakeStore.instances) == 2
    indexed, fallback = _FakeStore.instances
    assert indexed.index_config is not None
    assert indexed.setup_calls == 1
    assert fallback.index_config is None
    assert fallback.setup_calls == 1
