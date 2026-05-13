import os
from functools import lru_cache
from typing import Literal

from pydantic import PostgresDsn, RedisDsn, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Core Infrastructure
    postgres_dsn: PostgresDsn
    redis_url: RedisDsn  # Enforces valid Redis URI at startup
    qdrant_host: str
    qdrant_port: int

    # Security & Auth (Wrapped in SecretStr to prevent accidental logging)
    openai_api_key: SecretStr
    jwt_secret: SecretStr
    jwt_algorithm: str = "HS256"
    redis_encryption_key: SecretStr

    # Security Headers & CORS
    allowed_origins: list[str] = ["http://localhost:3000"]

    # ML Models
    embedding_model: str = "text-embedding-3-small"
    local_model_revision: str | None = None
    llm_model: str = "gpt-4o"

    # RAG Configuration
    chunk_size: int = 150
    chunk_overlap: int = 20

    # Observability & Environment
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    app_env: Literal["local", "docker", "production"] = "local"

    # Base configuration for Pydantic Settings
    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached instance of the Settings object.
    Dynamically loads the target .env file based on APP_ENV OS variable.
    """
    env_state = os.getenv("APP_ENV", "local")
    target_env_file = f".env.{env_state}"

    # Overriding the default env_file dynamically
    return Settings(_env_file=target_env_file)  # type: ignore[call-arg]
