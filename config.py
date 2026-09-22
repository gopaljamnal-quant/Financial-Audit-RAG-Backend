"""
config.py
=========

Centralised, strictly-typed application configuration.

All runtime configuration is sourced from environment variables (or a
mounted ``.env`` file) via :class:`pydantic_settings.BaseSettings`. No
secret material is ever hard-coded. This module is imported everywhere
else in the codebase as the single source of truth for configuration,
so it must remain free of side effects beyond settings resolution.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import AnyUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strictly-typed application settings loaded from the environment.

    :cvar model_config: Pydantic settings configuration -- reads from a
        local ``.env`` file when present, is case-insensitive on
        environment variable names, and forbids unknown/extra fields
        to fail fast on configuration drift in an audited environment.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",
    )

    # --- Application metadata -------------------------------------------------
    APP_NAME: str = Field(default="financial-audit-rag-backend")
    APP_ENV: str = Field(default="production", description="One of: local, staging, production.")
    LOG_LEVEL: str = Field(default="INFO")
    API_V1_PREFIX: str = Field(default="/api/v1")
    ALLOWED_CORS_ORIGINS: List[str] = Field(default_factory=lambda: ["https://localhost"])

    # --- Azure OpenAI -----------------------------------------------------------
    AZURE_OPENAI_ENDPOINT: AnyUrl = Field(..., description="e.g. https://<resource>.openai.azure.com/")
    AZURE_OPENAI_API_KEY: str = Field(..., repr=False)
    AZURE_OPENAI_API_VERSION: str = Field(default="2024-08-01-preview")
    AZURE_OPENAI_CHAT_DEPLOYMENT: str = Field(..., description="Deployment name for the chat/completions model.")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = Field(
        default="text-embedding-3-large",
        description="Deployment name for the embedding model.",
    )
    AZURE_OPENAI_EMBEDDING_DIMENSIONS: int = Field(default=3072, gt=0)

    # --- Azure AI Search ----------------------------------------------------------
    AZURE_SEARCH_ENDPOINT: AnyUrl = Field(..., description="e.g. https://<service>.search.windows.net")
    AZURE_SEARCH_API_KEY: str = Field(..., repr=False)
    AZURE_SEARCH_INDEX_NAME: str = Field(...)
    AZURE_SEARCH_VECTOR_FIELD: str = Field(default="content_vector")
    AZURE_SEARCH_CONTENT_FIELD: str = Field(default="content")
    AZURE_SEARCH_SEMANTIC_CONFIG: str = Field(default="default-semantic-config")
    AZURE_SEARCH_TOP_K: int = Field(default=8, gt=0, le=50)

    # --- Database (async SQLAlchemy) ----------------------------------------------
    DATABASE_URL: str = Field(
        ...,
        description="Async SQLAlchemy DSN, e.g. postgresql+asyncpg://user:pass@host:5432/dbname",
    )
    DATABASE_POOL_SIZE: int = Field(default=10, gt=0)
    DATABASE_MAX_OVERFLOW: int = Field(default=20, ge=0)
    DATABASE_ECHO: bool = Field(default=False)

    # --- MLflow -------------------------------------------------------------------
    MLFLOW_TRACKING_URI: str = Field(default="http://localhost:5000")
    MLFLOW_EXPERIMENT_NAME: str = Field(default="financial-audit-rag")

    # --- RAG / chunking -------------------------------------------------------------
    CHUNK_TOKEN_SIZE: int = Field(default=512, gt=0)
    CHUNK_TOKEN_OVERLAP: int = Field(default=64, ge=0)
    MAX_CONTEXT_TOKENS: int = Field(default=8000, gt=0)
    MAX_COMPLETION_TOKENS: int = Field(default=1200, gt=0)
    TIKTOKEN_ENCODING: str = Field(default="cl100k_base")

    @field_validator("CHUNK_TOKEN_OVERLAP")
    @classmethod
    def _overlap_must_be_smaller_than_chunk(cls, v: int, info) -> int:
        """Validate that the chunk overlap is strictly smaller than the chunk size.

        :param v: The proposed overlap value.
        :param info: Pydantic validation context, used to read sibling fields.
        :raises ValueError: If ``CHUNK_TOKEN_OVERLAP`` is not smaller than
            ``CHUNK_TOKEN_SIZE``, which would otherwise produce a
            non-advancing or infinite chunking loop.
        :return: The validated overlap value.
        """
        chunk_size = info.data.get("CHUNK_TOKEN_SIZE")
        if chunk_size is not None and v >= chunk_size:
            raise ValueError("CHUNK_TOKEN_OVERLAP must be strictly smaller than CHUNK_TOKEN_SIZE")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached, process-wide :class:`Settings` singleton.

    Using :func:`functools.lru_cache` ensures the environment is parsed
    exactly once per process and that the same validated instance is
    shared across the application (FastAPI dependency injection,
    services, and startup hooks alike).

    :return: The cached application settings instance.
    """
    return Settings()  # type: ignore[call-arg]


settings = get_settings()
