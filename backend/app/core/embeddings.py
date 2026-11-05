"""Embedding model factory shared by the knowledge base and durable memory."""

from __future__ import annotations

from functools import lru_cache
from threading import RLock
from typing import Any

from langchain_core.embeddings import Embeddings

from config.settings import settings


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str) -> Any:
    """Load each local embedding model once per process."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


@lru_cache(maxsize=4)
def _sentence_transformer_lock(model_name: str) -> RLock:
    """Serialize encode calls when several callers share one model instance."""
    return RLock()


class SentenceTransformerEmbeddings(Embeddings):
    """Local sentence-transformers model with an optional query instruction.

    Instruction-tuned models such as Qwen3-Embedding accept an
    "Instruct: ...\\nQuery: ..." prefix on the query side only; documents are
    embedded plain. EMBEDDING_QUERY_INSTRUCTION="" disables the prefix.
    """

    def __init__(self, model_name: str, query_instruction: str):
        self._model_name = model_name
        self._lock = _sentence_transformer_lock(model_name)
        self._query_instruction = query_instruction

    @property
    def _model(self) -> Any:
        # Loaded on first use, so code that only lists or deletes stored
        # vectors never pays for the model.
        return _load_sentence_transformer(self._model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            return self._model.encode(
                texts,
                show_progress_bar=len(texts) > 50,
            ).tolist()

    def embed_query(self, text: str) -> list[float]:
        if self._query_instruction:
            text = f"Instruct: {self._query_instruction}\nQuery: {text}"
        with self._lock:
            return self._model.encode(text).tolist()


def get_embeddings() -> Embeddings:
    """Create the embedding model selected by EMBEDDING_PROVIDER.

    - "sentence_transformers": local model loaded in this process
    - "local": OpenAI-compatible server such as LM Studio or Ollama
    """
    provider = settings.EMBEDDING_PROVIDER.lower()

    if provider == "sentence_transformers":
        return SentenceTransformerEmbeddings(
            settings.EMBEDDING_MODEL,
            settings.EMBEDDING_QUERY_INSTRUCTION,
        )

    if provider == "local":
        from langchain_openai import OpenAIEmbeddings
        from pydantic import SecretStr

        return OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            base_url=settings.EMBEDDING_BASE_URL,
            api_key=SecretStr("lm-studio"),  # LM Studio doesn't need a real key
            # Send plain strings: LM Studio rejects tiktoken token-ID batches.
            check_embedding_ctx_length=False,
        )

    raise ValueError(f"Unknown embedding provider: {provider}")
