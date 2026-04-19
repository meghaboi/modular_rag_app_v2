from __future__ import annotations

import logging
import uuid

from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
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
from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

from backend.config import Settings
from backend.models import ChatTurn, SourceChunk, UsageStats
from models.chunking_strategies import ChunkingStrategyFactory
from prompts import get_provider

logger = logging.getLogger(__name__)


class AzureRagError(RuntimeError):
    """Raised when Azure-backed RAG operations fail."""


class AzureRagService:
    """Azure-only RAG service using Azure AI Foundry and Azure AI Search."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._prompt_provider = get_provider("llm")
        self._chunker = ChunkingStrategyFactory.get_strategy(settings.chunking_strategy)
        self._default_credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)

        chat_auth = (
            settings.foundry_chat_api_key
            if settings.foundry_chat_api_key
            else get_bearer_token_provider(self._default_credential, "https://ai.azure.com/.default")
        )
        embedding_auth = (
            settings.foundry_embedding_api_key
            if settings.foundry_embedding_api_key
            else get_bearer_token_provider(self._default_credential, "https://ai.azure.com/.default")
        )
        self._chat_client = OpenAI(
            api_key=chat_auth,
            base_url=settings.foundry_chat_base_url,
        )
        self._chat_fallback_client = None
        if settings.foundry_chat_fallback_base_url and settings.foundry_chat_fallback_deployment:
            fallback_auth = (
                settings.foundry_chat_fallback_api_key
                if settings.foundry_chat_fallback_api_key
                else get_bearer_token_provider(self._default_credential, "https://ai.azure.com/.default")
            )
            self._chat_fallback_client = OpenAI(
                api_key=fallback_auth,
                base_url=settings.foundry_chat_fallback_base_url,
            )
        self._embedding_client = OpenAI(
            api_key=embedding_auth,
            base_url=settings.foundry_embedding_base_url,
        )

        search_credential = (
            AzureKeyCredential(settings.azure_search_api_key)
            if settings.azure_search_api_key
            else self._default_credential
        )
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

    def ingest_document(
        self,
        session_id: str,
        file_name: str,
        text: str,
        folder_path: str = "",
        document_id: str | None = None,
    ) -> tuple[str, int]:
        resolved_document_id = document_id or str(uuid.uuid4())
        file_path = self._compose_file_path(folder_path, file_name)
        chunks = self._chunker.chunk_text(
            text,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        cleaned_chunks = [chunk.strip() for chunk in chunks if chunk and chunk.strip()]
        if not cleaned_chunks:
            raise AzureRagError("Chunking produced no indexable content.")

        embeddings = self._embed_texts(
            [self._build_embedding_input(file_path, chunk) for chunk in cleaned_chunks]
        )
        payload = [
            {
                "id": f"{resolved_document_id}-{index}",
                "session_id": session_id,
                "document_id": resolved_document_id,
                "file_name": file_name,
                "folder_path": folder_path,
                "file_path": file_path,
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

        logger.info(
            "Indexed %s chunks for session %s document %s",
            len(payload),
            session_id,
            file_path,
        )
        return resolved_document_id, len(payload)

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
            (
                f"[{index}] Path: {context.file_path}\n"
                f"Chunk: {context.chunk_id + 1}\n"
                f"Content:\n{context.content}"
            )
            for index, context in enumerate(contexts, start=1)
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
            "\n\nResponse requirements:\n"
            "- Use only the provided context.\n"
            "- Keep the answer clear and concise.\n"
            "- Use clean markdown with normal headings, bullets, and short paragraphs.\n"
            "- Cite supporting snippets as [1], [2], etc.\n"
            "- Mention the relevant file path when it improves clarity.\n"
            "- Do not end with a follow-up question unless the user explicitly asks for options."
        )

        messages = [
            {
                "role": "system",
                "content": self._prompt_provider.get_prompt("system"),
            },
            {"role": "user", "content": user_prompt},
        ]

        response = self._create_chat_completion(messages)
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
            search_fields=["file_name", "file_path", "content"],
            vector_queries=[vector_query],
            filter=f"session_id eq '{session_id}'",
            top=self.settings.top_k,
            select=["document_id", "file_name", "folder_path", "file_path", "chunk_id", "content"],
        )

        contexts: list[SourceChunk] = []
        for item in results:
            contexts.append(
                SourceChunk(
                    document_id=item["document_id"],
                    file_name=item["file_name"],
                    folder_path=item.get("folder_path", ""),
                    file_path=item.get("file_path", item["file_name"]),
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
        return self._embed_texts([text])[0]

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self._embedding_client.embeddings.create(
                model=self.settings.foundry_embedding_deployment,
                input=texts,
                dimensions=self.settings.foundry_embedding_dimensions,
            )
        except APIConnectionError as exc:
            raise AzureRagError(
                "Embedding service is unreachable. Verify the Azure AI Foundry endpoint and key."
            ) from exc
        except RateLimitError as exc:
            raise AzureRagError(
                "Embedding service is busy right now. Retry in a moment."
            ) from exc
        except APIStatusError as exc:
            raise AzureRagError(
                f"Embedding request failed with status {exc.status_code}."
            ) from exc
        return [list(item.embedding) for item in response.data]

    def _format_history(self, history: list[ChatTurn]) -> str:
        if not history:
            return ""
        recent_turns = history[-6:]
        return "\n".join(f"{turn.role.title()}: {turn.content}" for turn in recent_turns)

    def _create_chat_completion(self, messages: list[dict[str, str]]):
        try:
            return self._chat_client.chat.completions.create(
                model=self.settings.foundry_chat_deployment,
                temperature=self.settings.temperature,
                messages=messages,
            )
        except (APIConnectionError, RateLimitError, APIStatusError) as exc:
            fallback_client = self._chat_fallback_client
            fallback_deployment = self.settings.foundry_chat_fallback_deployment
            if fallback_client and fallback_deployment:
                logger.warning(
                    "Primary chat deployment %s failed, falling back to %s.",
                    self.settings.foundry_chat_deployment,
                    fallback_deployment,
                )
                try:
                    return fallback_client.chat.completions.create(
                        model=fallback_deployment,
                        temperature=self.settings.temperature,
                        messages=messages,
                    )
                except APIConnectionError as fallback_exc:
                    raise AzureRagError(
                        "Chat service is unreachable. Verify the Azure AI Foundry endpoint and key."
                    ) from fallback_exc
                except RateLimitError as fallback_exc:
                    raise AzureRagError("Chat service is busy right now. Retry in a moment.") from fallback_exc
                except APIStatusError as fallback_exc:
                    raise AzureRagError(
                        f"Chat request failed with status {fallback_exc.status_code}."
                    ) from fallback_exc

            if isinstance(exc, APIConnectionError):
                raise AzureRagError(
                    "Chat service is unreachable. Verify the Azure AI Foundry endpoint and key."
                ) from exc
            if isinstance(exc, RateLimitError):
                raise AzureRagError("Chat service is busy right now. Retry in a moment.") from exc
            raise AzureRagError(
                f"Chat request failed with status {exc.status_code}."
            ) from exc

    def _build_embedding_input(self, file_path: str, chunk: str) -> str:
        return f"Path: {file_path}\nContent:\n{chunk}"

    def _compose_file_path(self, folder_path: str, file_name: str) -> str:
        return f"{folder_path}/{file_name}" if folder_path else file_name

    def _ensure_index(self) -> None:
        required_field_names = {
            "id",
            "session_id",
            "document_id",
            "file_name",
            "folder_path",
            "file_path",
            "chunk_id",
            "content",
            "content_vector",
        }
        existing = [index.name for index in self._index_client.list_indexes()]
        if self.settings.azure_search_index_name in existing:
            existing_index = self._index_client.get_index(self.settings.azure_search_index_name)
            existing_fields = {field.name for field in existing_index.fields}
            if required_field_names.issubset(existing_fields):
                return

            logger.warning(
                "Recreating Azure AI Search index %s to apply workspace metadata fields.",
                self.settings.azure_search_index_name,
            )
            self._index_client.delete_index(self.settings.azure_search_index_name)

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
            SimpleField(
                name="document_id",
                type=SearchFieldDataType.String,
                filterable=True,
            ),
            SearchableField(
                name="file_name",
                type=SearchFieldDataType.String,
                filterable=True,
            ),
            SimpleField(
                name="folder_path",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True,
            ),
            SearchableField(
                name="file_path",
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
                vector_search_dimensions=self.settings.foundry_embedding_dimensions,
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
