from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = os.getenv("PORT", 8080)
    DEBUG: bool = os.getenv("DEBUG", False)
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "app.log")
    LOG_MAX_BYTES: int = os.getenv("LOG_MAX_BYTES", 1000000)
    LOG_BACKUP_COUNT: int = os.getenv("LOG_BACKUP_COUNT", 5)
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

settings = Settings()