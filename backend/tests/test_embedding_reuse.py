"""RAG and memory Selection reuse one local embedding model instance."""

from app.core.rag import vector_store


def test_sentence_transformer_wrappers_share_model_and_lock(monkeypatch):
    model = object()
    monkeypatch.setattr(vector_store.settings, "EMBEDDING_PROVIDER", "sentence_transformers")
    monkeypatch.setattr(vector_store.settings, "EMBEDDING_MODEL", "test-model")
    monkeypatch.setattr(vector_store.settings, "EMBEDDING_QUERY_INSTRUCTION", "")
    monkeypatch.setattr(vector_store, "_load_sentence_transformer", lambda _name: model)
    vector_store._sentence_transformer_lock.cache_clear()

    rag_embeddings = vector_store.get_embeddings()
    memory_embeddings = vector_store.get_embeddings()

    assert rag_embeddings._model is model
    assert memory_embeddings._model is model
    assert rag_embeddings._lock is memory_embeddings._lock
