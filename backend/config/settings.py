from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from pydantic import SecretStr

class Settings(BaseSettings):
    HOST: str = "127.0.0.1"
    PORT: int = 8080
    DEBUG: bool = False
    OPENAI_MODEL: str = "gpt-5"
    GEMINI_MODEL: str = "gemini-2.5-flash"
    OPENAI_API_KEY: SecretStr | None = None
    GEMINI_API_KEY: SecretStr | None = None
    MODEL_TEMPERATURE: float = 0.3

    # RAG settings. chunk_size=2000 is the IR-benchmark winner (NDCG@5 0.537
    # vs 0.443 at 1000); overlap had no measurable effect. See docs/EVALUATION.md.
    RAG_CHUNK_SIZE: int = 2000
    RAG_CHUNK_OVERLAP: int = 100
    RAG_TOP_K: int = 3

    # Long-conversation summarization: once the history exceeds
    # SUMMARY_TRIGGER_MESSAGES conversational turns, older messages are folded
    # into a running summary and pruned, keeping the last SUMMARY_KEEP_RECENT
    # verbatim. Bounds the context window on long chats.
    SUMMARY_TRIGGER_MESSAGES: int = 20
    SUMMARY_KEEP_RECENT: int = 6

    # Embedding settings: "sentence_transformers", "local" (LM Studio / Ollama), or "gemini"
    EMBEDDING_PROVIDER: str = "sentence_transformers"
    EMBEDDING_MODEL: str = "Qwen/Qwen3-Embedding-0.6B"
    EMBEDDING_BASE_URL: str = "http://localhost:1234/v1"
    # Query-side instruction for Qwen3-Embedding (applied to queries only, not
    # documents — the model's trained retrieval format). Measured on the 30-query
    # IR benchmark it gave NO net gain (generic model-card instruction: +0.9pp
    # NDCG@5 but -3pp Recall@5/Hit@5; domain-specific wording hurt more), so it's
    # off by default. Set a non-empty value to re-enable. See docs/EVALUATION.md.
    EMBEDDING_QUERY_INSTRUCTION: str = ""

    # PostgreSQL / pgvector
    DATABASE_URL: str = "postgresql+psycopg://mindbridge:mindbridge_dev@localhost:5432/mindbridge"
    PGVECTOR_COLLECTION: str = "mental_health_docs"

    PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]  # => backend/
    DATA_DIR: Path = PROJECT_ROOT / "data"
    INDEX_DIR: Path = DATA_DIR / "MentalChat16K_faiss_index"

    CORS_ORIGINS: list[str] = [
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "http://localhost:7860",
        "http://127.0.0.1:7860",
    ]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    CORS_ALLOW_HEADERS: list[str] = ["*"]

    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "app.log"
    LOG_MAX_BYTES: int = 1000000
    LOG_BACKUP_COUNT: int = 5
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Auth
    SECRET_KEY: str = "change-me-in-production-use-32-random-bytes"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = 20

    # Max input length
    MAX_MESSAGE_LENGTH: int = 4000

    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    @property
    def ASYNC_DATABASE_URL(self) -> str:
        """asyncpg URL for SQLAlchemy async engine."""
        return self.DATABASE_URL.replace("postgresql+psycopg", "postgresql+asyncpg")

    @property
    def CHECKPOINT_DATABASE_URL(self) -> str:
        """psycopg3 URL for LangGraph checkpointer (no driver prefix)."""
        return self.DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")

settings = Settings()