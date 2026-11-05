"""Query side of the counseling knowledge base."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from app.core.observability.metrics import (
    RAG_SEARCH_SECONDS,
    RAG_SEARCHES,
    RAG_TOP_SIMILARITY,
)
from app.core.observability.tracing import tracer
from app.core.rag.knowledge_base import example_id, open_store, read_index_state
from config.settings import settings

_QUESTION_DISPLAY_CHARS = 500


@dataclass(frozen=True)
class RetrievedExample:
    id: str
    question: str
    answer: str
    # Cosine similarity between the query and the indexed question.
    similarity: float


class KnowledgeBaseUnavailable(RuntimeError):
    """The index is missing, incomplete, or built with another embedding model."""


def _verify_index() -> None:
    """Fail loudly instead of searching an index that would answer wrongly.

    A missing or half-built index returns few or no results, and vectors from
    another embedding model return unrelated ones. Neither raises on its own.
    """
    name = settings.RAG_COLLECTION
    rebuild = "Rebuild it with: python -m app.core.rag.knowledge_base"
    state = read_index_state(name)
    if state is None or state[1] == 0:
        raise KnowledgeBaseUnavailable(f"Knowledge base '{name}' has not been built. {rebuild}")
    metadata, stored = state
    if stored < int(metadata.get("examples", 0)):
        raise KnowledgeBaseUnavailable(
            f"Knowledge base '{name}' is incomplete: {stored} of "
            f"{metadata['examples']} examples stored. {rebuild}"
        )
    if metadata.get("embedding_model") != settings.EMBEDDING_MODEL:
        raise KnowledgeBaseUnavailable(
            f"Knowledge base '{name}' was embedded with "
            f"{metadata.get('embedding_model')!r}, but EMBEDDING_MODEL is "
            f"{settings.EMBEDDING_MODEL!r}. {rebuild}"
        )


class ExampleRetriever:
    """Finds the indexed client questions closest to a query."""

    def __init__(self, store: Any):
        self._store = store

    @classmethod
    def open(cls) -> "ExampleRetriever":
        """Blocking: checks the index, then loads the embedding model."""
        _verify_index()
        return cls(open_store(async_mode=True))

    async def search(
        self,
        query: str,
        k: int,
        *,
        min_similarity: float | None = None,
    ) -> list[RetrievedExample]:
        """Return up to k examples, best first, dropping those below the floor.

        A nearest-neighbour search always returns k rows, however unrelated.
        The floor keeps examples about a different concern out of the prompt.
        """
        floor = settings.RAG_MIN_SIMILARITY if min_similarity is None else min_similarity
        with tracer.start_as_current_span("rag.search") as span:
            started = time.perf_counter()
            hits = await self._store.asimilarity_search_with_score(query, k=k)
            RAG_SEARCH_SECONDS.observe(time.perf_counter() - started)
            examples = [
                RetrievedExample(
                    id=example_id(document.page_content),
                    question=document.page_content,
                    answer=document.metadata.get("answer", ""),
                    # pgvector returns cosine distance; similarity is its complement.
                    similarity=1.0 - distance,
                )
                for document, distance in hits
            ]
            served = [example for example in examples if example.similarity >= floor]
            if examples:
                RAG_TOP_SIMILARITY.observe(examples[0].similarity)
                span.set_attribute("rag.top_similarity", round(examples[0].similarity, 4))
            span.set_attribute("rag.floor", floor)
            span.set_attribute("rag.served", len(served))
            RAG_SEARCHES.labels("served" if served else "empty").inc()
            return served


def _clip(text: str, limit: int) -> str:
    """Shorten text to the limit, ending on a sentence boundary when one exists."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    boundary = max(cut.rfind(mark) for mark in (". ", "? ", "! ", "\n"))
    if boundary > limit // 2:
        cut = cut[: boundary + 1]
    return cut.rstrip() + " [...]"


def format_examples(examples: list[RetrievedExample]) -> str:
    """Render retrieved pairs as the tool result shown to the model."""
    blocks = []
    for index, example in enumerate(examples, start=1):
        blocks.append(
            f"[Example {index}]\n"
            f"Client: {_clip(example.question, _QUESTION_DISPLAY_CHARS)}\n"
            f"Counselor: {_clip(example.answer, settings.RAG_ANSWER_MAX_CHARS)}"
        )
    return "\n\n".join(blocks)


_retriever: ExampleRetriever | None = None
_lock = asyncio.Lock()


async def search_examples(
    query: str,
    k: int | None = None,
    *,
    min_similarity: float | None = None,
) -> list[RetrievedExample]:
    """Search with one retriever shared by the whole process."""
    global _retriever
    if _retriever is None:
        async with _lock:
            if _retriever is None:
                try:
                    _retriever = await asyncio.to_thread(ExampleRetriever.open)
                except KnowledgeBaseUnavailable:
                    RAG_SEARCHES.labels("unavailable").inc()
                    raise
    return await _retriever.search(
        query,
        k or settings.RAG_TOP_K,
        min_similarity=min_similarity,
    )
