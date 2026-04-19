from __future__ import annotations

import os
from dataclasses import dataclass


class SettingsError(RuntimeError):
    """Raised when required environment configuration is missing."""


def _get_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _require_value(value: str | None, message: str) -> str:
    if value:
        return value
    raise SettingsError(message)


def _normalize_foundry_base_url(raw_value: str) -> str:
    cleaned = raw_value.rstrip("/")
    if cleaned.endswith("/openai/v1"):
        return f"{cleaned}/"
    if cleaned.endswith(".services.ai.azure.com") or cleaned.endswith(".openai.azure.com"):
        return f"{cleaned}/openai/v1/"
    return f"{cleaned}/"


def _resolve_foundry_base_url(
    direct_names: tuple[str, ...],
    endpoint_names: tuple[str, ...],
    resource_names: tuple[str, ...],
) -> str:
    direct_base_url = _get_env(*direct_names)
    if direct_base_url:
        return _normalize_foundry_base_url(direct_base_url)

    endpoint = _get_env(*endpoint_names)
    if endpoint:
        return _normalize_foundry_base_url(endpoint)

    resource_name = _get_env(*resource_names)
    if resource_name:
        return f"https://{resource_name}.services.ai.azure.com/openai/v1/"

    raise SettingsError(
        "Missing Foundry endpoint configuration. Set a chat or embedding base URL/endpoint "
        "(for example AZURE_FOUNDRY_CHAT_BASE_URL or AZURE_FOUNDRY_EMBEDDING_BASE_URL), "
        "or use the shared AZURE_FOUNDRY_BASE_URL / AZURE_FOUNDRY_ENDPOINT variables."
    )


def _resolve_search_endpoint() -> str:
    endpoint = _get_env("AZURE_SEARCH_ENDPOINT")
    if endpoint:
        return endpoint.rstrip("/")

    service_name = _get_env("AZURE_SEARCH_SERVICE_NAME")
    if service_name:
        return f"https://{service_name}.search.windows.net"

    raise SettingsError(
        "Missing Azure AI Search endpoint configuration. Set AZURE_SEARCH_ENDPOINT "
        "or AZURE_SEARCH_SERVICE_NAME."
    )


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
    foundry_chat_base_url: str
    foundry_chat_api_key: str | None
    foundry_chat_deployment: str
    foundry_chat_fallback_base_url: str | None
    foundry_chat_fallback_api_key: str | None
    foundry_chat_fallback_deployment: str | None
    foundry_embedding_base_url: str
    foundry_embedding_api_key: str | None
    foundry_embedding_deployment: str
    foundry_embedding_dimensions: int
    azure_search_endpoint: str
    azure_search_api_key: str | None
    azure_search_index_name: str
    frontend_origin: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    chunking_strategy: str
    temperature: float

    @property
    def auth_mode(self) -> str:
        return (
            "api_key"
            if self.foundry_chat_api_key
            or self.foundry_chat_fallback_api_key
            or self.foundry_embedding_api_key
            or self.azure_search_api_key
            else "entra_id"
        )

    @classmethod
    def from_env(cls) -> "Settings":
        chat_base_url = _resolve_foundry_base_url(
            direct_names=("AZURE_FOUNDRY_CHAT_BASE_URL", "AZURE_FOUNDRY_BASE_URL"),
            endpoint_names=("AZURE_FOUNDRY_CHAT_ENDPOINT", "AZURE_FOUNDRY_ENDPOINT"),
            resource_names=("AZURE_FOUNDRY_CHAT_RESOURCE_NAME", "AZURE_FOUNDRY_RESOURCE_NAME"),
        )
        embedding_base_url = _resolve_foundry_base_url(
            direct_names=("AZURE_FOUNDRY_EMBEDDING_BASE_URL", "AZURE_FOUNDRY_BASE_URL"),
            endpoint_names=("AZURE_FOUNDRY_EMBEDDING_ENDPOINT", "AZURE_FOUNDRY_ENDPOINT"),
            resource_names=("AZURE_FOUNDRY_EMBEDDING_RESOURCE_NAME", "AZURE_FOUNDRY_RESOURCE_NAME"),
        )
        chat_fallback_base_url = _get_env(
            "AZURE_FOUNDRY_CHAT_FALLBACK_BASE_URL",
            "AZURE_FOUNDRY_CHAT_FALLBACK_ENDPOINT",
        )

        return cls(
            foundry_chat_base_url=chat_base_url,
            foundry_chat_api_key=_get_env(
                "AZURE_FOUNDRY_CHAT_API_KEY",
                "AZURE_FOUNDRY_API_KEY",
                "AZURE_OPENAI_API_KEY",
            ),
            foundry_chat_deployment=os.getenv("AZURE_FOUNDRY_CHAT_DEPLOYMENT", "Kimi-K2.5"),
            foundry_chat_fallback_base_url=(
                _normalize_foundry_base_url(chat_fallback_base_url)
                if chat_fallback_base_url
                else None
            ),
            foundry_chat_fallback_api_key=_get_env(
                "AZURE_FOUNDRY_CHAT_FALLBACK_API_KEY",
            ),
            foundry_chat_fallback_deployment=_get_env(
                "AZURE_FOUNDRY_CHAT_FALLBACK_DEPLOYMENT",
            ),
            foundry_embedding_base_url=embedding_base_url,
            foundry_embedding_api_key=_get_env(
                "AZURE_FOUNDRY_EMBEDDING_API_KEY",
                "AZURE_FOUNDRY_API_KEY",
                "AZURE_OPENAI_API_KEY",
            ),
            foundry_embedding_deployment=os.getenv(
                "AZURE_FOUNDRY_EMBEDDING_DEPLOYMENT",
                "embed-v-4-0",
            ),
            foundry_embedding_dimensions=_get_int("AZURE_FOUNDRY_EMBEDDING_DIMENSIONS", 1536),
            azure_search_endpoint=_resolve_search_endpoint(),
            azure_search_api_key=_get_env("AZURE_SEARCH_API_KEY"),
            azure_search_index_name=os.getenv("AZURE_SEARCH_INDEX_NAME", "ca-rag-documents"),
            frontend_origin=os.getenv("FRONTEND_ORIGIN", "http://localhost:3000"),
            chunk_size=_get_int("RAG_CHUNK_SIZE", 800),
            chunk_overlap=_get_int("RAG_CHUNK_OVERLAP", 120),
            top_k=_get_int("RAG_TOP_K", 5),
            chunking_strategy=os.getenv("RAG_CHUNKING_STRATEGY", "Paragraph-based"),
            temperature=_get_float("AZURE_FOUNDRY_TEMPERATURE", 0.2),
        )
