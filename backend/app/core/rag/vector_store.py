"""Vector store abstraction supporting FAISS (legacy) and pgvector backends."""

from __future__ import annotations

from typing import Any

from datasets import load_dataset
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores.base import VectorStoreRetriever
from tqdm import tqdm

from config.settings import settings


def get_embeddings() -> Embeddings:
    """Create embedding model based on settings.

    Supports:
      - "local": OpenAI-compatible API (LM Studio, Ollama, etc.)
      - "gemini": Google Generative AI embeddings
    """
    provider = settings.EMBEDDING_PROVIDER.lower()

    if provider == "local":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_base=settings.EMBEDDING_BASE_URL,
            openai_api_key="lm-studio",  # LM Studio doesn't need a real key
        )
    elif provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL or "gemini-embedding-001",
            google_api_key=settings.GEMINI_API_KEY,
        )
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


class VectorStore:
    """Unified vector store that can use either FAISS or pgvector as backend."""

    def __init__(
        self,
        backend: str = "pgvector",
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        collection_name: str | None = None,
    ):
        self.backend = backend
        self.chunk_size = chunk_size or settings.RAG_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.RAG_CHUNK_OVERLAP
        self.collection_name = collection_name or settings.PGVECTOR_COLLECTION

        self.ds: Any = None
        self.vectorstore: Any = None
        self.embedding = get_embeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def load_documents(self) -> None:
        self.ds = load_dataset("ShenLab/MentalChat16K")

    def _example_to_documents(self) -> list[Document]:
        assert self.ds is not None, "Dataset not loaded"
        docs: list[Document] = []

        for ex in tqdm(self.ds["train"], desc="Building chunks"):
            assistant_output: str = ex.get("output", "") or ""
            chunks = self.text_splitter.split_text(assistant_output)

            for chunk in chunks:
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": "MentalChat16K",
                            "chunk_size": self.chunk_size,
                            "chunk_overlap": self.chunk_overlap,
                        },
                    )
                )
        return docs

    # ── pgvector backend ──────────────────────────────────────────────

    def _get_pgvector_store(self):
        from langchain_postgres import PGVector

        return PGVector(
            embeddings=self.embedding,
            collection_name=self.collection_name,
            connection=settings.DATABASE_URL,
            use_jsonb=True,
        )

    def build_pgvector(self) -> Any:
        self.load_documents()
        docs = self._example_to_documents()

        from langchain_postgres import PGVector

        self.vectorstore = PGVector.from_documents(
            documents=docs,
            embedding=self.embedding,
            collection_name=self.collection_name,
            connection=settings.DATABASE_URL,
            use_jsonb=True,
            pre_delete_collection=True,
        )
        return self.vectorstore

    def load_pgvector(self) -> Any:
        self.vectorstore = self._get_pgvector_store()
        return self.vectorstore

    # ── FAISS backend (legacy) ────────────────────────────────────────

    def build_faiss(self) -> Any:
        from langchain_community.vectorstores.faiss import FAISS

        self.load_documents()
        docs = self._example_to_documents()

        self.vectorstore = FAISS.from_documents(docs, self.embedding)
        index_dir = settings.INDEX_DIR
        index_dir.parent.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(str(index_dir))
        return self.vectorstore

    def load_faiss(self) -> Any:
        from langchain_community.vectorstores.faiss import FAISS

        index_dir = settings.INDEX_DIR
        if not index_dir.exists():
            raise FileNotFoundError(f"Index not found at {index_dir}")

        self.vectorstore = FAISS.load_local(
            str(index_dir),
            self.embedding,
            allow_dangerous_deserialization=True,
        )
        return self.vectorstore

    # ── Unified interface ─────────────────────────────────────────────

    def build(self) -> Any:
        if self.backend == "pgvector":
            return self.build_pgvector()
        return self.build_faiss()

    def load(self) -> Any:
        if self.backend == "pgvector":
            return self.load_pgvector()
        return self.load_faiss()

    def get_retriever(self, k: int | None = None) -> VectorStoreRetriever:
        assert self.vectorstore is not None, "Vectorstore not built/loaded"
        k = k or settings.RAG_TOP_K
        return self.vectorstore.as_retriever(search_kwargs={"k": k})


if __name__ == "__main__":
    store = VectorStore(backend="pgvector")
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
