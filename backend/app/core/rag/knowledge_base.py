"""Counseling knowledge base: MentalChat16K question/answer pairs in pgvector.

Each indexed item is one client question together with the counselor's full
answer. Only the question is embedded. A search query describes the user's
concern, so it reads like another client's question rather than like a
counselor's reply, and the matching answer comes back whole.

Build or rebuild the index:
    uv run python -m app.core.rag.knowledge_base
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from app.core.embeddings import get_embeddings
from config.settings import settings

DATASET_NAME = "ShenLab/MentalChat16K"
# Fixed namespace: an example keeps the same id across rebuilds, so evaluation
# gold labels that refer to ids stay valid.
_ID_NAMESPACE = uuid.UUID("6f1d3c52-8f0e-4b7a-9d55-2c1a7e4b9a10")


@dataclass(frozen=True)
class QAExample:
    id: str
    question: str
    answer: str


def example_id(question: str) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, question.casefold()))


def load_examples(max_samples: int | None = None) -> list[QAExample]:
    """Load the dataset, dropping empty rows and repeated questions.

    When a question appears more than once, its first answer is kept.
    """
    from datasets import load_dataset

    examples: list[QAExample] = []
    seen: set[str] = set()
    for row in load_dataset(DATASET_NAME, split="train"):
        question = " ".join((row.get("input") or "").split())
        question = question[: settings.RAG_QUESTION_MAX_CHARS]
        answer = (row.get("output") or "").strip()
        if not question or not answer:
            continue
        identifier = example_id(question)
        if identifier in seen:
            continue
        seen.add(identifier)
        examples.append(QAExample(id=identifier, question=question, answer=answer))
        if max_samples and len(examples) >= max_samples:
            break
    return examples


def open_store(
    *,
    collection_name: str | None = None,
    async_mode: bool = False,
    **kwargs: Any,
) -> Any:
    """Open the pgvector collection that holds the knowledge base."""
    from langchain_postgres import PGVector

    return PGVector(
        embeddings=get_embeddings(),
        collection_name=collection_name or settings.RAG_COLLECTION,
        connection=settings.DATABASE_URL,
        use_jsonb=True,
        async_mode=async_mode,
        **kwargs,
    )


def build_index(
    *,
    collection_name: str | None = None,
    max_samples: int | None = None,
    batch_size: int = 500,
) -> int:
    """Embed every question and store it with its answer.

    Replaces any existing collection. Returns the number of indexed examples.
    """
    from langchain_core.documents import Document
    from tqdm import tqdm

    examples = load_examples(max_samples)
    store = open_store(collection_name=collection_name, pre_delete_collection=True)
    try:
        # Batches keep each INSERT under PostgreSQL's 65,535 bind parameters.
        for start in tqdm(
            range(0, len(examples), batch_size),
            desc="Indexing questions",
            unit="batch",
        ):
            batch = examples[start : start + batch_size]
            store.add_documents(
                [
                    Document(page_content=item.question, metadata={"answer": item.answer})
                    for item in batch
                ],
                ids=[item.id for item in batch],
            )
    except Exception:
        # A half-built collection would answer searches with partial results.
        store.delete_collection()
        raise
    return len(examples)


if __name__ == "__main__":
    print(f"Embedding model: {settings.EMBEDDING_MODEL} ({settings.EMBEDDING_PROVIDER})")
    count = build_index()
    print(f"Indexed {count} question/answer pairs into '{settings.RAG_COLLECTION}'.")
