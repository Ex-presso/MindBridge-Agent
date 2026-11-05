"""Query side of the counseling knowledge base."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from app.core.rag.knowledge_base import example_id, open_store
from config.settings import settings

_QUESTION_DISPLAY_CHARS = 500


@dataclass(frozen=True)
class RetrievedExample:
    id: str
    question: str
    answer: str
    # Cosine similarity between the query and the indexed question.
    similarity: float


class ExampleRetriever:
    """Finds the indexed client questions closest to a query."""

    def __init__(self, store: Any):
        self._store = store

    @classmethod
    def open(cls) -> "ExampleRetriever":
        """Blocking: loads the embedding model."""
        return cls(open_store(async_mode=True))

    async def search(self, query: str, k: int) -> list[RetrievedExample]:
        hits = await self._store.asimilarity_search_with_score(query, k=k)
        return [
            RetrievedExample(
                id=example_id(document.page_content),
                question=document.page_content,
                answer=document.metadata.get("answer", ""),
                # pgvector returns cosine distance; similarity is its complement.
                similarity=1.0 - distance,
            )
            for document, distance in hits
        ]


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


async def search_examples(query: str, k: int | None = None) -> list[RetrievedExample]:
    """Search with one retriever shared by the whole process."""
    global _retriever
    if _retriever is None:
        async with _lock:
            if _retriever is None:
                _retriever = await asyncio.to_thread(ExampleRetriever.open)
    return await _retriever.search(query, k or settings.RAG_TOP_K)
