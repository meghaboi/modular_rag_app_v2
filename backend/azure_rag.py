from __future__ import annotations

import logging
from typing import Any

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchableField,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI

from backend.config import Settings
from backend.models import ChatTurn, SourceChunk, UsageStats
from models.chunking_strategies import ChunkingStrategyFactory
from prompts import get_provider

logger = logging.getLogger(__name__)


class AzureRagError(RuntimeError):
    """Raised when Azure-backed RAG operations fail."""


class AzureRagService:
    """Azure-only RAG service using AI Foundry deployments and Azure AI Search."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._prompt_provider = get_provider("llm")
        self._chunker = ChunkingStrategyFactory.get_strategy(settings.chunking_strategy)
        self._aoai_client = AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint,
        )
        search_credential = AzureKeyCredential(settings.azure_search_api_key)
        self._index_client = SearchIndexClient(
            endpoint=settings.azure_search_endpoint,
            credential=search_credential,
        )
        self._search_client = SearchClient(
            endpoint=settings.azure_search_endpoint,
            index_name=settings.azure_search_index_name,
            credential=search_credential,
        )
        self._ensure_index()

    def ingest_document(self, session_id: str, file_name: str, text: str) -> int:
        chunks = self._chunker.chunk_text(
            text,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        cleaned_chunks = [chunk.strip() for chunk in chunks if chunk and chunk.strip()]
        if not cleaned_chunks:
            raise AzureRagError("Chunking produced no indexable content.")

        embeddings = self._embed_texts(cleaned_chunks)
        payload = [
            {
                "id": f"{session_id}-{index}",
                "session_id": session_id,
                "file_name": file_name,
                "chunk_id": index,
                "content": chunk,
                "content_vector": embedding,
            }
            for index, (chunk, embedding) in enumerate(zip(cleaned_chunks, embeddings))
        ]

        results = self._search_client.upload_documents(payload)
        failed_uploads = [result.key for result in results if not result.succeeded]
        if failed_uploads:
            raise AzureRagError(
                f"Azure AI Search failed to index chunks: {', '.join(failed_uploads)}"
            )

        logger.info("Indexed %s chunks for session %s", len(payload), session_id)
        return len(payload)

    def answer_question(
        self,
        session_id: str,
        question: str,
        history: list[ChatTurn] | None = None,
    ) -> tuple[str, list[SourceChunk], UsageStats]:
        contexts = self.retrieve_contexts(session_id, question)
        if not contexts:
            return (
                "No indexed context was found for this question in the uploaded file.",
                [],
                UsageStats(),
            )

        context_block = "\n\n".join(
            f"[{index}] {context.content}" for index, context in enumerate(contexts, start=1)
        )
        history_block = self._format_history(history or [])
        if history_block:
            user_prompt = self._prompt_provider.get_prompt(
                "chat",
                context=context_block,
                conversation_history=history_block,
                user_message=question,
            )
        else:
            user_prompt = self._prompt_provider.get_prompt(
                "query",
                context=context_block,
                question=question,
            )

        user_prompt += (
            "\n\nUse only the provided context. When you rely on a snippet, cite it as [1], [2], etc."
        )

        response = self._aoai_client.chat.completions.create(
            model=self.settings.azure_openai_chat_deployment,
            temperature=self.settings.temperature,
            messages=[
                {
                    "role": "system",
                    "content": self._prompt_provider.get_prompt("system"),
                },
                {"role": "user", "content": user_prompt},
            ],
        )
        usage = response.usage
        answer = response.choices[0].message.content or ""
        usage_stats = UsageStats(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
        )
        return answer, contexts, usage_stats

    def retrieve_contexts(self, session_id: str, question: str) -> list[SourceChunk]:
        query_embedding = self._embed_text(question)
        vector_query = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=self.settings.top_k,
            fields="content_vector",
        )

        results = self._search_client.search(
            search_text=question,
            vector_queries=[vector_query],
            filter=f"session_id eq '{session_id}'",
            top=self.settings.top_k,
            select=["file_name", "chunk_id", "content"],
        )

        contexts: list[SourceChunk] = []
        for item in results:
            contexts.append(
                SourceChunk(
                    file_name=item["file_name"],
                    chunk_id=item["chunk_id"],
                    content=item["content"],
                    score=item.get("@search.score"),
                )
            )
        return contexts

    def delete_session(self, session_id: str) -> None:
        results = self._search_client.search(
            search_text="*",
            filter=f"session_id eq '{session_id}'",
            top=1000,
            select=["id"],
        )
        batch = [{"id": item["id"]} for item in results]
        if batch:
            self._search_client.delete_documents(batch)

    def _embed_text(self, text: str) -> list[float]:
        response = self._aoai_client.embeddings.create(
            model=self.settings.azure_openai_embedding_deployment,
            input=text,
            dimensions=self.settings.azure_openai_embedding_dimensions,
        )
        return list(response.data[0].embedding)

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self._aoai_client.embeddings.create(
            model=self.settings.azure_openai_embedding_deployment,
            input=texts,
            dimensions=self.settings.azure_openai_embedding_dimensions,
        )
        return [list(item.embedding) for item in response.data]

    def _format_history(self, history: list[ChatTurn]) -> str:
        if not history:
            return ""
        recent_turns = history[-6:]
        return "\n".join(f"{turn.role.title()}: {turn.content}" for turn in recent_turns)

    def _ensure_index(self) -> None:
        existing = [index.name for index in self._index_client.list_indexes()]
        if self.settings.azure_search_index_name in existing:
            return

        fields = [
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True,
            ),
            SimpleField(
                name="session_id",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True,
            ),
            SearchableField(
                name="file_name",
                type=SearchFieldDataType.String,
                filterable=True,
            ),
            SimpleField(
                name="chunk_id",
                type=SearchFieldDataType.Int32,
                filterable=True,
                sortable=True,
            ),
            SearchableField(
                name="content",
                type=SearchFieldDataType.String,
            ),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=self.settings.azure_openai_embedding_dimensions,
                vector_search_profile_name="ca-rag-vector-profile",
            ),
        ]
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(name="ca-rag-hnsw"),
            ],
            profiles=[
                VectorSearchProfile(
                    name="ca-rag-vector-profile",
                    algorithm_configuration_name="ca-rag-hnsw",
                )
            ],
        )

        index = SearchIndex(
            name=self.settings.azure_search_index_name,
            fields=fields,
            vector_search=vector_search,
        )
        self._index_client.create_index(index)
        logger.info("Created Azure AI Search index %s", self.settings.azure_search_index_name)
