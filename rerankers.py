from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import os
from enums import RerankerModelType
import requests

class Reranker(ABC):
    """Abstract base class for rerankers following Interface Segregation Principle"""

    @abstractmethod
    def rerank(self, query: str, documents: List[str]) -> List[Tuple[str, float]]:
        """Rerank documents based on relevance to the query"""
        pass

class VoyageReranker(Reranker):
    """Voyage AI Reranker implementation"""

    def __init__(self, model_name="voyage-2"):
        """Initialize the Voyage AI reranker
        
        Args:
            model_name: The model name to use (default: rerank-2)
        """
        import voyageai

        if not os.environ.get("VOYAGE_API_KEY"):
            raise ValueError("Voyage API key not found in environment variables")

        self._client = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY"))
        self._model_name = model_name

    def rerank(self, query: str, documents: List[str]) -> List[Tuple[str, float]]:
        """Rerank documents based on relevance to the query using Voyage AI model"""
        if not documents:
            return []

        try:
            # Get reranking scores from Voyage API
            scores = self._client.rerank(
                query=query,
                documents=documents,
                model=self._model_name
            )
            
            # Create sorted list of (document, score) tuples
            scored_docs = list(zip(documents, scores))
            
            # Sort by score in descending order
            reranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
            
            return reranked_docs

        except Exception as e:
            print(f"Error in reranking with Voyage: {str(e)}")
            return [(doc, 1.0) for doc in documents]  # Return original documents as fallback

class CohereRerankerV2(Reranker):
    """Cohere V2 Reranker implementation"""

    def __init__(self):
        """Initialize the Cohere V2 reranker"""
        import cohere

        if not os.environ.get("COHERE_API_KEY"):
            raise ValueError("Cohere API key not found in environment variables")

        self._client = cohere.Client(os.environ.get("COHERE_API_KEY"))

    def rerank(self, query: str, documents: List[str]) -> List[Tuple[str, float]]:
        """Rerank documents based on relevance to the query using Cohere V2 model"""
        if not documents:
            return []

        try:
            # Call Cohere reranking endpoint with V2 model
            results = self._client.rerank(
                query=query,
                documents=documents,
                model="rerank-english-v2.0",
                top_n=len(documents)
            )

            # Process results
            reranked_docs = []
            for result in results.results:
                doc_index = result.index
                relevance_score = result.relevance_score
                reranked_docs.append((documents[doc_index], relevance_score))

            return reranked_docs

        except Exception as e:
            print(f"Error in reranking: {str(e)}")
            return [(doc, 1.0) for doc in documents]  # Return original documents as fallback

class CohereRerankerV3(Reranker):
    """Cohere V3 Reranker implementation"""

    def __init__(self):
        """Initialize the Cohere V3 reranker"""
        import cohere

        if not os.environ.get("COHERE_API_KEY"):
            raise ValueError("Cohere API key not found in environment variables")

        self._client = cohere.Client(os.environ.get("COHERE_API_KEY"))

    def rerank(self, query: str, documents: List[str]) -> List[Tuple[str, float]]:
        """Rerank documents based on relevance to the query using Cohere V3 model"""
        if not documents:
            return []

        try:
            # Call Cohere reranking endpoint with V3 model
            results = self._client.rerank(
                query=query,
                documents=documents,
                model="rerank-english-v3.0",
                top_n=len(documents)
            )

            # Process results
            reranked_docs = []
            for result in results.results:
                doc_index = result.index
                relevance_score = result.relevance_score
                reranked_docs.append((documents[doc_index], relevance_score))

            return reranked_docs

        except Exception as e:
            print(f"Error in reranking: {str(e)}")
            return [(doc, 1.0) for doc in documents]  # Return original documents as fallback

class CohereRerankerMultilingual(Reranker):
    """Cohere Multilingual Reranker implementation"""

    def __init__(self):
        """Initialize the Cohere multilingual reranker"""
        import cohere

        if not os.environ.get("COHERE_API_KEY"):
            raise ValueError("Cohere API key not found in environment variables")

        self._client = cohere.Client(os.environ.get("COHERE_API_KEY"))

    def rerank(self, query: str, documents: List[str]) -> List[Tuple[str, float]]:
        """Rerank documents based on relevance to the query using Cohere multilingual model"""
        if not documents:
            return []

        try:
            # Call Cohere reranking endpoint with multilingual model
            results = self._client.rerank(
                query=query,
                documents=documents,
                model="rerank-multilingual-v3.0",
                top_n=len(documents)
            )

            # Process results
            reranked_docs = []
            for result in results.results:
                doc_index = result.index
                relevance_score = result.relevance_score
                reranked_docs.append((documents[doc_index], relevance_score))

            return reranked_docs

        except Exception as e:
            print(f"Error in reranking: {str(e)}")
            return [(doc, 1.0) for doc in documents]  # Return original documents as fallback

class JinaReranker(Reranker):
    """Jina AI Reranker implementation"""

    def __init__(self, model_name="jina-rerank-v1"):
        """Initialize the Jina AI reranker
        
        Args:
            model_name: The model name to use (default: jina-rerank-v1)
        """
        if not os.environ.get("JINA_API_KEY"):
            raise ValueError("Jina API key not found in environment variables")
            
        self._api_key = os.environ.get("JINA_API_KEY")
        self._model_name = model_name
        self._api_url = "https://api.jina.ai/v1/rerank"
        
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
                self._api_url,
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

        except Exception as e:
            print(f"Error in reranking with Jina: {str(e)}")
            return [(doc, 1.0) for doc in documents]  # Return original documents as fallback

class RerankerFactory:
    """Factory for creating rerankers (Factory Pattern)"""

    @staticmethod
    def create_reranker(reranker_name: str) -> Reranker:
        """Create a reranker based on the reranker name"""
        if reranker_name == RerankerModelType.COHERE_V2:
            return CohereRerankerV2()
        elif reranker_name == RerankerModelType.COHERE_V3:
            return CohereRerankerV3()
        elif reranker_name == RerankerModelType.COHERE_MULTILINGUAL:
            return CohereRerankerMultilingual()
        elif reranker_name == RerankerModelType.VOYAGE:
            return VoyageReranker()
        elif reranker_name == RerankerModelType.VOYAGE_2:
            return VoyageReranker(model_name="rerank-2")
        elif reranker_name == RerankerModelType.JINA:
            return JinaReranker(model_name="jina-reranker-v1-base-en")
        elif reranker_name == RerankerModelType.JINA_V2:
            return JinaReranker(model_name="jina-colbert-v2")
        else:
            raise ValueError(f"Unsupported reranker: {reranker_name}")