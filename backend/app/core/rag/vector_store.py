"""pgvector-backed vector store for the counseling knowledge base."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.core.embeddings import get_embeddings
from config.settings import settings

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.vectorstores.base import VectorStoreRetriever


class VectorStore:
    """Builds and loads the pgvector collection."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        collection_name: str | None = None,
        max_samples: int | None = None,
        async_mode: bool = False,
    ):
        self.chunk_size = chunk_size or settings.RAG_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.RAG_CHUNK_OVERLAP
        self.collection_name = collection_name or settings.PGVECTOR_COLLECTION
        self.max_samples = max_samples
        # async_mode=True is required for callers that use retriever.ainvoke()
        # (e.g. the LangGraph agent). Sync eval scripts using retriever.invoke()
        # should leave it False.
        self.async_mode = async_mode

        self.ds: Any = None
        self.vectorstore: Any = None
        self.embedding = get_embeddings()
        self.text_splitter: Any = None

    def load_documents(self) -> None:
        from datasets import load_dataset
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self.ds = load_dataset("ShenLab/MentalChat16K")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        if self.max_samples and len(self.ds["train"]) > self.max_samples:
            self.ds["train"] = self.ds["train"].select(range(self.max_samples))

    def _example_to_documents(self) -> list[Document]:
        from langchain_core.documents import Document
        from tqdm import tqdm

        assert self.ds is not None, "Dataset not loaded"
        assert self.text_splitter is not None, "Text splitter not initialized"
        docs: list[Document] = []

        for example_idx, ex in enumerate(tqdm(self.ds["train"], desc="Building chunks")):
            assistant_output: str = ex.get("output", "") or ""
            chunks = self.text_splitter.split_text(assistant_output)

            for chunk_idx, chunk in enumerate(chunks):
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": "MentalChat16K",
                            "chunk_size": self.chunk_size,
                            "chunk_overlap": self.chunk_overlap,
                            "source_example_idx": example_idx,
                            "chunk_idx": chunk_idx,
                        },
                    )
                )
        return docs

    def _get_pgvector_store(self):
        from langchain_postgres import PGVector

        return PGVector(
            embeddings=self.embedding,
            collection_name=self.collection_name,
            connection=settings.DATABASE_URL,
            use_jsonb=True,
            async_mode=self.async_mode,
        )

    def build_pgvector(self, batch_size: int = 1000) -> Any:
        """Build the pgvector index, inserting in batches.

        PostgreSQL caps a single statement at 65,535 bind parameters; PGVector
        binds 4 per row, so single-shot inserts fail past ~16k chunks. Batched
        adds keep us well under that for any chunking configuration.
        """
        self.load_documents()
        docs = self._example_to_documents()

        from langchain_postgres import PGVector
        from tqdm import tqdm

        self.vectorstore = PGVector(
            embeddings=self.embedding,
            collection_name=self.collection_name,
            connection=settings.DATABASE_URL,
            use_jsonb=True,
            pre_delete_collection=True,
        )

        try:
            for i in tqdm(
                range(0, len(docs), batch_size),
                desc="Inserting chunks",
                unit="batch",
            ):
                self.vectorstore.add_documents(docs[i : i + batch_size])
        except Exception:
            # Drop the partial collection so a re-run sees it as missing
            # rather than caching half-populated state.
            try:
                self.vectorstore.delete_collection()
            except Exception:
                pass
            raise

        return self.vectorstore

    def load_pgvector(self) -> Any:
        self.vectorstore = self._get_pgvector_store()
        return self.vectorstore

    def build(self) -> Any:
        return self.build_pgvector()

    def load(self) -> Any:
        return self.load_pgvector()

    def get_retriever(self, k: int | None = None) -> VectorStoreRetriever:
        assert self.vectorstore is not None, "Vectorstore not built/loaded"
        k = k or settings.RAG_TOP_K
        return self.vectorstore.as_retriever(search_kwargs={"k": k})


if __name__ == "__main__":
    store = VectorStore()
    print(f"Embedding provider: {settings.EMBEDDING_PROVIDER}")
    print(f"Embedding model: {settings.EMBEDDING_MODEL}")
    print("Building pgvector index...")
    store.build()

    query = "I feel anxious all the time and can't sleep well."
    results = store.get_retriever(k=3).invoke(query)
    for r in results:
        print("---- Page Content ----")
        print(r.page_content[:400])
        print(r.metadata)
