from __future__ import annotations

import os
from dataclasses import dataclass


class SettingsError(RuntimeError):
    """Raised when required environment configuration is missing."""


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SettingsError(f"Missing required environment variable: {name}")
    return value


def _get_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise SettingsError(f"Environment variable {name} must be an integer.") from exc


def _get_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise SettingsError(f"Environment variable {name} must be a float.") from exc


@dataclass(frozen=True)
class Settings:
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_api_version: str
    azure_openai_chat_deployment: str
    azure_openai_embedding_deployment: str
    azure_openai_embedding_dimensions: int
    azure_search_endpoint: str
    azure_search_api_key: str
    azure_search_index_name: str
    frontend_origin: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    chunking_strategy: str
    temperature: float

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            azure_openai_endpoint=_require_env("AZURE_OPENAI_ENDPOINT"),
            azure_openai_api_key=_require_env("AZURE_OPENAI_API_KEY"),
            azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            azure_openai_chat_deployment=_require_env("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            azure_openai_embedding_deployment=_require_env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            azure_openai_embedding_dimensions=_get_int("AZURE_OPENAI_EMBEDDING_DIMENSIONS", 1536),
            azure_search_endpoint=_require_env("AZURE_SEARCH_ENDPOINT"),
            azure_search_api_key=_require_env("AZURE_SEARCH_API_KEY"),
            azure_search_index_name=os.getenv("AZURE_SEARCH_INDEX_NAME", "ca-rag-documents"),
            frontend_origin=os.getenv("FRONTEND_ORIGIN", "http://localhost:3000"),
            chunk_size=_get_int("RAG_CHUNK_SIZE", 800),
            chunk_overlap=_get_int("RAG_CHUNK_OVERLAP", 120),
            top_k=_get_int("RAG_TOP_K", 5),
            chunking_strategy=os.getenv("RAG_CHUNKING_STRATEGY", "Paragraph-based"),
            temperature=_get_float("AZURE_OPENAI_TEMPERATURE", 0.2),
        )
