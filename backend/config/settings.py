from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from pydantic import Field, SecretStr
from typing import Literal


class Settings(BaseSettings):
    HOST: str = "127.0.0.1"
    PORT: int = 8080
    DEBUG: bool = False
    DEFAULT_LLM_MODEL: str = "deepseek-v4-flash"
    DEFAULT_LLM_BASE_URL: str = "https://api.deepseek.com"
    OPENAI_MODEL: str = "gpt-5"
    GEMINI_MODEL: str = "gemini-2.5-flash"
    OPENAI_API_KEY: SecretStr | None = None
    GEMINI_API_KEY: SecretStr | None = None
    MODEL_TEMPERATURE: float = 0.3

    # RAG settings. The current retrieval benchmark must be rerun before these
    # production defaults are presented as an evaluated optimum.
    RAG_CHUNK_SIZE: int = 2000
    RAG_CHUNK_OVERLAP: int = 100
    RAG_TOP_K: int = 3

    # Long-conversation working-memory compaction (token-based, Claude Code
    # autoCompact style; was message-count). Once the estimated token count of the
    # working context exceeds COMPACT_TRIGGER_TOKENS, fold older turns into a running
    # summary and prune them, keeping ~COMPACT_KEEP_RECENT_TOKENS of recent turns
    # verbatim. Cheap-first: microCompact replaces old (re-derivable) tool results
    # with a placeholder before paying for LLM summarization. See docs/MEMORY.md.
    COMPACT_TRIGGER_TOKENS: int = 6000
    COMPACT_KEEP_RECENT_TOKENS: int = 2000
    COMPACT_MICRO_KEEP_RESULTS: int = 3  # keep the last N tool results verbatim
    COMPACT_MICRO_MIN_CHARS: int = 120  # only compact tool results longer than this

    # Global kill switch for durable, cross-conversation memory. Per-user
    # consent is stored separately on User.memory_enabled and defaults off.
    MEMORY_ENABLED: bool = False
    MEMORY_SELECT_SEMANTIC_LIMIT: int = Field(default=12, ge=1, le=100)
    MEMORY_SELECT_EPISODE_TOP_K: int = Field(default=3, ge=1, le=20)
    MEMORY_EPISODE_MIN_SCORE: float = Field(default=0.55, ge=-1, le=1)
    MEMORY_SEMANTIC_ITEM_MAX_CHARS: int = Field(default=280, ge=50, le=2000)
    MEMORY_EPISODE_SUMMARY_MAX_CHARS: int = Field(default=800, ge=100, le=5000)
    MEMORY_EPISODE_TOPIC_MAX_CHARS: int = Field(default=80, ge=10, le=500)
    MEMORY_CONTEXT_MAX_CHARS: int = Field(default=4000, ge=500, le=20000)
    MEMORY_EMBED_DIMS: int = Field(default=1024, ge=1)
    # One lightweight in-process outbox worker. It remains enabled when the
    # memory kill switch is off so previously queued privacy deletions still run.
    MEMORY_WORKER_ENABLED: bool = True
    MEMORY_WORKER_POLL_SECONDS: float = Field(default=1.0, ge=0.1, le=60.0)
    MEMORY_WORKER_LEASE_SECONDS: int = Field(default=90, ge=35, le=600)
    MEMORY_WORKER_MAX_ATTEMPTS: int = Field(default=3, ge=1, le=10)

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
    DATABASE_URL: str = (
        "postgresql+psycopg://mindbridge:mindbridge_dev@localhost:5432/mindbridge"
    )
    # Credential/account deletion barriers rely on a fresh statement snapshot
    # after a blocked row lock is released. Do not make this configurable to a
    # snapshot isolation level without adding a credential epoch protocol.
    DATABASE_ISOLATION_LEVEL: Literal["READ COMMITTED"] = "READ COMMITTED"
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
    CORS_ALLOW_METHODS: list[str] = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
    CORS_ALLOW_HEADERS: list[str] = ["*"]

    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "app.log"
    LOG_MAX_BYTES: int = 1000000
    LOG_BACKUP_COUNT: int = 5
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Auth
    SECRET_KEY: str = "change-me-in-production-use-32-random-bytes"
    # Separate key for encrypting user API keys (Fernet). Falls back to
    # SECRET_KEY when unset — set it in production so rotating the JWT signing
    # key doesn't also invalidate every stored API key. Rotating ENCRYPTION_KEY
    # itself invalidates existing ciphertexts (expected key-rotation behavior).
    ENCRYPTION_KEY: str | None = None
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    # Refresh-cookie Secure flag — must be True behind HTTPS in production.
    COOKIE_SECURE: bool = False

    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = 20
    # Stricter limit for auth endpoints (brute-force protection).
    AUTH_RATE_LIMIT: str = "5/minute"

    # Feature flags
    # Legacy /chat/completions is unauthenticated and uses system keys — off by
    # default; enable only for a trusted local OpenAI-compatible client.
    ENABLE_LEGACY_CHAT_ENDPOINT: bool = False
    # Legacy flag name: safely adopt create_all databases and upgrade Alembic to
    # head on startup (dev/Docker convenience). Run migrations explicitly in prod.
    AUTO_CREATE_TABLES: bool = True

    # Max input length
    MAX_MESSAGE_LENGTH: int = 4000

    # LangSmith tracing (optional, off by default). When enabled, every LangGraph
    # run, tool call, and LLM request is traced to LangSmith. Get a key at
    # https://smith.langchain.com. These are exported to os.environ at startup
    # (main.configure_tracing) because LangChain reads them from the process
    # environment, not from this pydantic .env.
    LANGSMITH_TRACING: bool = False
    LANGSMITH_API_KEY: SecretStr | None = None
    LANGSMITH_PROJECT: str = "mindbridge"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"

    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
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
