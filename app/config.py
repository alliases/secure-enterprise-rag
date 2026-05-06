# File: app/config.py
# Purpose: Centralized application configuration with Pydantic BaseSettings.

from functools import lru_cache

from pydantic import PostgresDsn, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Core Infrastructure
    postgres_dsn: PostgresDsn
    redis_url: str
    qdrant_host: str
    qdrant_port: int

    # Security & Auth (Wrapped in SecretStr to prevent accidental logging)
    openai_api_key: SecretStr
    jwt_secret: SecretStr
    jwt_algorithm: str = "HS256"
    redis_encryption_key: SecretStr

    # ML Models
    embedding_model: str = "text-embedding-3-small"
    local_model_revision: str | None = (
        None  # e.g., "main" or specific commit hash for safetensors
    )
    llm_model: str = "gpt-4o"

    # RAG Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Observability
    log_level: str = "INFO"

    # Configuration for loading from .env file
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached instance of the Settings object.
    Singleton pattern ensures .env is parsed only once.
    """
    return Settings()  # type: ignore[call-arg]
