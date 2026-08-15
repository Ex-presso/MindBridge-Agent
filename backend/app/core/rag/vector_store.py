"""Vector store abstraction supporting FAISS (legacy) and pgvector backends."""

from __future__ import annotations

from functools import lru_cache
from threading import RLock
from typing import TYPE_CHECKING, Any

from config.settings import settings

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores.base import VectorStoreRetriever


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str) -> Any:
    """Load each local embedding model once per process."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


@lru_cache(maxsize=4)
def _sentence_transformer_lock(model_name: str) -> RLock:
    """Serialize encode calls when RAG and memory share one model instance."""
    return RLock()


def get_embeddings() -> Embeddings:
    """Create embedding model based on settings.

    Supports:
      - "sentence_transformers": Local model via sentence-transformers (recommended)
      - "local": OpenAI-compatible API (LM Studio, Ollama, etc.)
      - "gemini": Google Generative AI embeddings
    """
    provider = settings.EMBEDDING_PROVIDER.lower()

    if provider == "sentence_transformers":
        from langchain_core.embeddings import Embeddings as BaseEmbeddings

        class SentenceTransformerEmbeddings(BaseEmbeddings):
            """Qwen3-Embedding wrapper with asymmetric query instruction.

            Qwen3-Embedding is trained for retrieval with an "Instruct: …\\nQuery: …"
            prefix on the *query* side only; documents are embedded plain. Applying
            it lifts retrieval quality with no reindex (documents are unchanged).
            Set EMBEDDING_QUERY_INSTRUCTION="" to disable (e.g. for a non-instruct
            embedding model).
            """

            def __init__(self, model, model_name: str, query_instruction: str):
                self._model = model
                self._lock = _sentence_transformer_lock(model_name)
                self._query_instruction = query_instruction

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

        return SentenceTransformerEmbeddings(
            _load_sentence_transformer(settings.EMBEDDING_MODEL),
            settings.EMBEDDING_MODEL,
            settings.EMBEDDING_QUERY_INSTRUCTION,
        )

    elif provider == "local":
        from langchain_openai import OpenAIEmbeddings
        from pydantic import SecretStr

        return OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            base_url=settings.EMBEDDING_BASE_URL,
            api_key=SecretStr("lm-studio"),  # LM Studio doesn't need a real key
            # Send plain strings: LM Studio rejects tiktoken token-ID batches.
            check_embedding_ctx_length=False,
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
        max_samples: int | None = None,
        async_mode: bool = False,
    ):
        self.backend = backend
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

    # ── pgvector backend ──────────────────────────────────────────────

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
