from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import os
from enums import RerankerModelType
import requests
from llm_models import StreamingLLM # Corrected import
from llm_models import ClaudeLLM # Corrected import for Claude model
import json # Add json import

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

class LLMReranker(Reranker):
    """LLM-based Reranker implementation"""

    def __init__(self, llm_client: StreamingLLM, model_name: str = "claude-3-7-sonnet-20240708"):
        """Initialize the LLM reranker
        
        Args:
            llm_client: The LLM client to use for reranking.
            model_name: The LLM model name to use.
        """
        self._llm_client = llm_client
        self._model_name = model_name # Store the model name

    def rerank(self, query: str, documents: List[str]) -> List[Tuple[str, float]]:
        """Rerank documents based on relevance to the query using an LLM."""
        if not documents:
            return []

        try:
            # Construct a prompt for the LLM
            prompt = f"Query: {query}\\n\\nDocuments to rerank (original indices provided):\\n"
            for i, doc in enumerate(documents):
                prompt += f"Index {i}: {doc}\\n" # Provide original index clearly
            prompt += """\\nPlease rerank the documents above based on their relevance to the query.
                Return a JSON string representing a list of objects, where each object has two keys: 'document_index' (the original index of the document as provided above) and 'relevance_score' (a float between 0.0 and 1.0, where 1.0 is most relevant).
                The list should be sorted by relevance_score in descending order.

                Example JSON output:
                [
                {"document_index": 2, "relevance_score": 0.95},
                {"document_index": 0, "relevance_score": 0.88},
                {"document_index": 1, "relevance_score": 0.75}
                ]
                """

            response_text = self._llm_client.generate(prompt, context="", model_name=self._model_name) # Pass model_name
            
            # Parse the LLM's JSON response
            try:
                # The LLM might return the JSON string within a code block (e.g., ```json ... ```)
                # or with other text. We need to extract the JSON part.
                if "```json" in response_text:
                    json_str = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text: # If no "json" tag, but still has backticks
                    json_str = response_text.split("```")[1].split("```")[0].strip()
                else:
                    json_str = response_text.strip()

                reranked_data = json.loads(json_str)
            except json.JSONDecodeError as json_e:
                print(f"Error decoding JSON from LLM response: {json_e}")
                print(f"LLM Response Text: {response_text}")
                # Fallback: return original documents if JSON parsing fails
                return [(doc, 0.5) for doc in documents] # Use a neutral score

            reranked_docs = []
            seen_indices = set()
            for item in reranked_data:
                if isinstance(item, dict) and 'document_index' in item and 'relevance_score' in item:
                    original_index = item['document_index']
                    if isinstance(original_index, int) and 0 <= original_index < len(documents):
                        if original_index not in seen_indices: # Ensure each document is added once
                           reranked_docs.append((documents[original_index], float(item['relevance_score'])))
                           seen_indices.add(original_index)
                        else:
                            print(f"Warning: Duplicate document_index {original_index} in LLM response. Skipping.")
                    else:
                        print(f"Warning: Invalid document_index {original_index} in LLM response. Skipping.")
                else:
                    print(f"Warning: Malformed item in LLM response: {item}. Skipping.")

            # Ensure all original documents are present if LLM missed some, append with low score
            if len(reranked_docs) < len(documents):
                print("Warning: LLM reranker did not return all documents. Appending missing ones with low score.")
                for i, doc in enumerate(documents):
                    if i not in seen_indices:
                        reranked_docs.append((doc, 0.1)) # Assign a low score for missing docs

            # Sort by score in descending order, as a final safety measure
            reranked_docs = sorted(reranked_docs, key=lambda x: x[1], reverse=True)

            return reranked_docs

        except Exception as e:
            print(f"Error in reranking with LLM: {str(e)}")
            # Fallback: return original documents with a default score
            return [(doc, 1.0) for doc in documents]

class RerankerFactory:
    """Factory for creating rerankers (Factory Pattern)"""

    @staticmethod
    def create_reranker(reranker_name: str, llm_client: Optional[StreamingLLM] = None) -> Reranker:
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
        elif reranker_name == RerankerModelType.LLM:
            if not llm_client:
                 # Default to ClaudeLLM if no client is provided
                llm_client = ClaudeLLM(model_name="claude-3-7-sonnet-20240708") # Use ClaudeLLM
            return LLMReranker(llm_client=llm_client)
        else:
            raise ValueError(f"Unsupported reranker: {reranker_name}")