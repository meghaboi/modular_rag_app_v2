from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
import os
import logging
import json
import requests
from utils.enums import RerankerModelType
from models.llm_models import StreamingLLM, LLMFactory, LLMModelType
from prompts import get_provider

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def _get_api_key(key_name: str) -> str:
    """Retrieve API key from environment variables."""
    api_key = os.environ.get(key_name)
    if not api_key:
        raise ValueError(f"{key_name} not found in environment variables.")
    return api_key

class Reranker(ABC):
    """Abstract base class for rerankers following Interface Segregation Principle"""

    @abstractmethod
    def rerank(self, query: str, documents: List[str]) -> List[Tuple[str, float]]:
        """Rerank documents based on relevance to the query"""
        pass

class VoyageReranker(Reranker):
    """Voyage AI Reranker implementation."""
    DEFAULT_MODEL = "rerank-2"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        """Initialize the Voyage AI reranker."""
        import voyageai
        self._api_key = _get_api_key("VOYAGE_API_KEY")
        self._client = voyageai.Client(api_key=self._api_key)
        self._model_name = model_name

    def rerank(self, query: str, documents: List[str]) -> List[Tuple[str, float]]:
        """Rerank documents based on relevance to the query using Voyage AI model"""
        if not documents:
            return []

        try:
            # Get reranking results from Voyage API
            reranking_output = self._client.rerank(
                query=query,
                documents=documents,
                model=self._model_name
            )
            
            # Extract document and score from each RerankingResult object
            # The results are already sorted by relevance score in descending order by the API.
            reranked_docs = []
            if reranking_output and hasattr(reranking_output, 'results'):
                for result_item in reranking_output.results:
                    # result_item.document is the document string
                    # result_item.relevance_score is the score
                    reranked_docs.append((result_item.document, result_item.relevance_score))
            else:
                log.warning("Voyage reranker did not return expected 'results' attribute. Falling back to original documents.")
                return [(doc, 0.0) for doc in documents]
            
            return reranked_docs

        except Exception as e:
            log.error(f"Error in reranking with Voyage: {e}", exc_info=True)
            return [(doc, 0.0) for doc in documents]

class CohereReranker(Reranker):
    """Cohere Reranker implementation."""
    DEFAULT_MODEL = "rerank-english-v3.0"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        """Initialize the Cohere reranker."""
        import cohere
        self._api_key = _get_api_key("COHERE_API_KEY")
        self._client = cohere.Client(self._api_key)
        self._model_name = model_name

    def rerank(self, query: str, documents: List[str]) -> List[Tuple[str, float]]:
        """Rerank documents based on relevance to the query using a Cohere model."""
        if not documents:
            return []

        try:
            results = self._client.rerank(
                query=query,
                documents=documents,
                model=self._model_name,
                top_n=len(documents)
            )

            reranked_docs = [
                (documents[result.index], result.relevance_score)
                for result in results.results
            ]

            return reranked_docs

        except Exception as e:
            log.error(f"Error in reranking with Cohere model {self._model_name}: {e}", exc_info=True)
            return [(doc, 0.0) for doc in documents]

class JinaReranker(Reranker):
    """Jina AI Reranker implementation."""
    DEFAULT_MODEL = "jina-reranker-v1-base-en"
    API_URL = "https://api.jina.ai/v1/rerank"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        """Initialize the Jina AI reranker."""
        self._api_key = _get_api_key("JINA_API_KEY")
        self._model_name = model_name
        
    def rerank(self, query: str, documents: List[str]) -> List[Tuple[str, float]]:
        """Rerank documents based on relevance to the query using Jina AI model"""
        if not documents:
            return []

        try:
            # Prepare payload for Jina API
            payload = {
                "model": self._model_name,
                "query": query,
                "documents": documents,
                "top_n": len(documents)  # Return all documents
            }
            
            # Set up headers with API key
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json"
            }
            
            # Make request to Jina API
            response = requests.post(
                self.API_URL,
                headers=headers,
                json=payload
            )
            
            # Check for successful response
            response.raise_for_status()
            result = response.json()
            
            # Process results
            reranked_docs = []
            
            # Extract results from Jina response
            # Format will be: [{"index": 0, "score": 0.92, ...}, ...]
            for item in result.get("results", []):
                doc_index = item.get("index")
                relevance_score = item.get("score")
                reranked_docs.append((documents[doc_index], relevance_score))
            
            return reranked_docs

        except requests.exceptions.RequestException as e:
            log.error(f"Error calling Jina API: {e}", exc_info=True)
            return [(doc, 0.0) for doc in documents]
        except Exception as e:
            log.error(f"Error in reranking with Jina: {e}", exc_info=True)
            return [(doc, 0.0) for doc in documents]

class LLMReranker(Reranker):
    """LLM-based Reranker implementation."""

    def __init__(self, llm_client: StreamingLLM, model_name: str = "claude-3-5-sonnet-20240620"):
        """Initialize the LLM reranker.

        Args:
            llm_client: An initialized LLM client to use for reranking.
            model_name: The LLM model name to use.
        """
        self._llm_client = llm_client
        self._model_name = model_name  # Keep model_name for backward compatibility
        self._prompt_provider = get_provider('reranker')

    def rerank(self, query: str, documents: List[str]) -> List[Tuple[str, float]]:
        """Rerank documents based on relevance to the query using an LLM."""
        if not documents:
            return []

        try:
            prompt = self._prompt_provider.get_prompt('rerank', query=query, documents=documents)
            
            # Fix: Handle both old and new LLM client interfaces
            try:
                # Try new interface (returns tuple)
                response_text, _ = self._llm_client.generate(prompt)
            except (ValueError, TypeError):
                # Fallback to old interface (returns string)
                response_text = self._llm_client.generate(prompt, context="", model_name=self._model_name)

            try:
                # Expect the LLM to return a valid JSON string.
                json_str = response_text.strip()
                # Handle cases where the JSON is wrapped in markdown
                if json_str.startswith("```json") and json_str.endswith("```"):
                    json_str = json_str[7:-3].strip()
                elif json_str.startswith("```") and json_str.endswith("```"):
                    json_str = json_str[3:-3].strip()
                
                reranked_data = json.loads(json_str)
                if not isinstance(reranked_data, list):
                    raise json.JSONDecodeError("Expected a JSON list of objects.", json_str, 0)

            except json.JSONDecodeError as e:
                log.error(f"Failed to decode JSON from LLM response: {e}", exc_info=True)
                log.error(f"LLM Response Text: {response_text}")
                return [(doc, 0.0) for doc in documents]

            reranked_docs_map = {item['document_index']: (documents[item['document_index']], float(item['relevance_score']))
                                 for item in reranked_data if 'document_index' in item and 'relevance_score' in item and 0 <= item['document_index'] < len(documents)}

            # Create the final list, preserving order from the reranked data
            reranked_docs = [reranked_docs_map[item['document_index']] for item in reranked_data if item.get('document_index') in reranked_docs_map]

            # Add any documents the LLM may have missed with a low score
            if len(reranked_docs) < len(documents):
                log.warning("LLM reranker did not return all documents. Appending missing ones.")
                seen_indices = {item['document_index'] for item in reranked_data}
                for i, doc in enumerate(documents):
                    if i not in seen_indices:
                        reranked_docs.append((doc, 0.1))

            return sorted(reranked_docs, key=lambda x: x[1], reverse=True)

        except Exception as e:
            log.error(f"Error in reranking with LLM: {e}", exc_info=True)
            return [(doc, 0.0) for doc in documents]

class RerankerFactory:
    """Factory for creating rerankers."""

    _reranker_map: Dict[str, Dict[str, Any]] = {
        "cohere_v2": {"class": CohereReranker, "model_name": "rerank-english-v2.0"},
        "cohere_v3": {"class": CohereReranker, "model_name": "rerank-english-v3.0"},
        "cohere_multilingual": {"class": CohereReranker, "model_name": "rerank-multilingual-v3.0"},
        "voyage_1": {"class": VoyageReranker, "model_name": "rerank-1"},
        "voyage_2": {"class": VoyageReranker, "model_name": "rerank-2"},
        "jina": {"class": JinaReranker, "model_name": "jina-reranker-v1-base-en"},
        "jina_v2": {"class": JinaReranker, "model_name": "jina-colbert-v2"},
        "llm": {"class": LLMReranker, "requires_llm": True},
        
        RerankerModelType.COHERE_V2: {"class": CohereReranker, "model_name": "rerank-english-v2.0"},
        RerankerModelType.COHERE_V3: {"class": CohereReranker, "model_name": "rerank-english-v3.0"},
        RerankerModelType.COHERE_MULTILINGUAL: {"class": CohereReranker, "model_name": "rerank-multilingual-v3.0"},
        RerankerModelType.VOYAGE_1: {"class": VoyageReranker, "model_name": "rerank-1"},
        RerankerModelType.VOYAGE_2: {"class": VoyageReranker, "model_name": "rerank-2"},
        RerankerModelType.JINA: {"class": JinaReranker, "model_name": "jina-reranker-v1-base-en"},
        RerankerModelType.JINA_V2: {"class": JinaReranker, "model_name": "jina-colbert-v2"},
        RerankerModelType.LLM: {"class": LLMReranker, "requires_llm": True},
    }

    @classmethod
    def create_reranker(cls, model_type, llm_client: Optional[StreamingLLM] = None) -> Reranker:
        """Create a reranker based on the model type.
        
        Args:
            model_type: Either a string (backward compatibility) or RerankerModelType enum
            llm_client: Optional LLM client for LLM-based rerankers
        """
        if isinstance(model_type, str):
            key = model_type.lower().replace('-', '_')
        else:
            key = model_type
            
        if key not in cls._reranker_map:
            raise ValueError(f"Unsupported reranker type: {model_type}")

        config = cls._reranker_map[key]
        reranker_class = config["class"]

        if config.get("requires_llm"):
            if not llm_client:
                log.info("LLM client not provided for LLMReranker, creating default.")
                # Import here to avoid circular imports
                from models.llm_models import LLMFactory, LLMModelType
                llm_client = LLMFactory.create_llm(LLMModelType.CLAUDE_3_5_SONNET)
            return reranker_class(llm_client=llm_client)
        else:
            model_name = config.get("model_name")
            return reranker_class(model_name=model_name)