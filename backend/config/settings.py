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


    PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]  # => backend/
    DATA_DIR: Path = PROJECT_ROOT / "data"
    INDEX_DIR: Path = DATA_DIR / "MentalChat16K_faiss_index"

    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    CORS_ALLOW_HEADERS: list[str] = ["*"]

    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "app.log"
    LOG_MAX_BYTES: int = 1000000
    LOG_BACKUP_COUNT: int = 5
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

settings = Settings()