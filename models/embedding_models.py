from typing import List, Dict, Any, Type
import os
import time
import random
import logging
from abc import ABC, abstractmethod
from utils.enums import EmbeddingModelType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_api_key(env_var: str) -> str:
    """Retrieve API key from environment variables with a standardized error."""
    api_key = os.environ.get(env_var)
    if not api_key:
        raise ValueError(f"{env_var} not found in environment variables.")
    return api_key

class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    
    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Convert a query string to an embedding vector."""
        pass
    
    @abstractmethod
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Convert a list of document strings to embedding vectors."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the embedding vectors."""
        pass

class OpenAIEmbedding(EmbeddingModel):
    """OpenAI embedding model implementation using langchain_openai."""
    MODEL_NAME = "text-embedding-3-large"
    DIMENSION = 1536

    def __init__(self):
        from langchain_openai import OpenAIEmbeddings
        self._model = OpenAIEmbeddings(model=self.MODEL_NAME, openai_api_key=get_api_key("OPENAI_API_KEY"))
    
    def embed_query(self, query: str) -> List[float]:
        return self._model.embed_query(query)
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        return self._model.embed_documents(documents)
    
    @property
    def dimension(self) -> int:
        return self.DIMENSION

class CohereEmbedding(EmbeddingModel):
    """Cohere embedding model implementation using the official cohere SDK."""
    MODEL_NAME = "embed-v4.0"  # Keep original model version
    DIMENSION = 1024

    def __init__(self):
        import cohere
        self._client = cohere.Client(api_key=get_api_key("COHERE_API_KEY"))

    def embed_query(self, query: str) -> List[float]:
        response = self._client.embed(texts=[query], model=self.MODEL_NAME, input_type="search_query")
        return response.embeddings[0]

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        response = self._client.embed(texts=documents, model=self.MODEL_NAME, input_type="search_document")
        return response.embeddings

    @property
    def dimension(self) -> int:
        return self.DIMENSION

class GeminiEmbedding(EmbeddingModel):
    """Gemini embedding model implementation using langchain_google_genai."""
    MODEL_NAME = "models/gemini-embedding-exp-03-07"
    DIMENSION = 768

    def __init__(self):
        import google.generativeai as genai
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        
        api_key = get_api_key("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        
        self._model = GoogleGenerativeAIEmbeddings(
            model=self.MODEL_NAME,
            google_api_key=api_key
        )

    def embed_query(self, query: str) -> List[float]:
        return self._model.embed_query(query)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        return self._model.embed_documents(documents)

    @property
    def dimension(self) -> int:
        return self.DIMENSION

class MistralEmbedding(EmbeddingModel):
    """Mistral embedding model implementation with batching and rate limiting."""
    MODEL_NAME = "mistral-embed"
    DIMENSION = 1024
    MAX_TOKENS_PER_BATCH = 8192

    def __init__(self, batch_size=32, initial_delay=1.0, max_retries=5, max_delay=60.0):
        from mistralai import Mistral
        import tiktoken
        self._client = Mistral(api_key=get_api_key("MISTRAL_API_KEY"))
        self._batch_size = batch_size
        self._initial_delay = initial_delay
        self._max_retries = max_retries
        self._max_delay = max_delay
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string"""
        return len(self._tokenizer.encode(text))
    
    def _batch_texts(self, texts: List[str], max_tokens_per_batch: int = 8192) -> List[List[str]]:
        """Split texts into batches based on token count"""
        batches = []
        current_batch = []
        current_batch_tokens = 0
        
        for text in texts:
            text_tokens = self._count_tokens(text)
            
            if text_tokens > max_tokens_per_batch:
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_batch_tokens = 0
                batches.append([text])
                continue
            
            if current_batch_tokens + text_tokens > max_tokens_per_batch:
                batches.append(current_batch)
                current_batch = [text]
                current_batch_tokens = text_tokens
            else:
                current_batch.append(text)
                current_batch_tokens += text_tokens
        
        if current_batch:
            batches.append(current_batch)
        
        return batches

    def _call_api_with_backoff(self, inputs):
        """Call the Mistral API with exponential backoff - keeping original method."""
        retry_count = 0
        delay = self._initial_delay
        
        while True:
            try:
                # Keep original API calling method
                response = self._client.embeddings.create(
                    model=self.MODEL_NAME,
                    inputs=inputs
                )
                return response
            except Exception as e:
                error_message = str(e)
                
                if "429" in error_message and "rate limit" in error_message.lower():
                    retry_count += 1
                    
                    if retry_count > self._max_retries:
                        logger.error(f"Maximum retries ({self._max_retries}) exceeded. Giving up.")
                        raise e
                    
                    jitter = random.uniform(0, 0.1 * delay)
                    actual_delay = min(delay + jitter, self._max_delay)
                    
                    logger.warning(f"Rate limit exceeded. Retrying in {actual_delay:.2f} seconds (retry {retry_count}/{self._max_retries})")
                    time.sleep(actual_delay)
                    
                    delay = min(delay * 2, self._max_delay)
                else:
                    raise e

    def embed_query(self, query: str) -> List[float]:
        response = self._call_api_with_backoff(query)
        return response.data[0].embedding

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        all_embeddings = []
        batches = self._batch_texts(documents)
        
        for i, batch in enumerate(batches):
            logger.info(f"Processing batch {i+1}/{len(batches)} with {len(batch)} documents")
            
            if i > 0:
                time.sleep(1)
            
            response = self._call_api_with_backoff(batch)
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings

    @property
    def dimension(self) -> int:
        return self.DIMENSION

class VoyageEmbedding(EmbeddingModel):
    """Voyage AI embedding model implementation using the official Voyage SDK."""
    MODEL_DIMENSIONS = {
        "voyage-3": 1024,  # Keep original default model
        "voyage-2": 1024,
        "voyage-large-2": 1536,
        "voyage-code-2": 1536,
        "voyage-large-2-instruct": 1536,
    }
    DEFAULT_MODEL = "voyage-3"  # Keep original default

    def __init__(self, model_name=DEFAULT_MODEL, batch_size=128):
        import voyageai
        self._client = voyageai.Client(api_key=get_api_key("VOYAGE_API_KEY"))
        self._model_name = model_name
        self._batch_size = batch_size
        if model_name not in self.MODEL_DIMENSIONS:
            logger.warning(f"Model '{model_name}' not in known list. Assuming dimension 1024.")

    def embed_query(self, query: str) -> List[float]:
        result = self._client.embed(texts=[query], model=self._model_name, input_type="query")
        return result.embeddings[0]

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        all_embeddings = []
        for i in range(0, len(documents), self._batch_size):
            batch = documents[i:i + self._batch_size]
            result = self._client.embed(texts=batch, model=self._model_name, input_type="document")
            all_embeddings.extend(result.embeddings)
        return all_embeddings

    @property
    def dimension(self) -> int:
        return self.MODEL_DIMENSIONS.get(self._model_name, 1024)


class EmbeddingModelFactory:
    """Factory for creating embedding models."""
    _model_map: Dict[EmbeddingModelType, Type[EmbeddingModel]] = {
        EmbeddingModelType.OPENAI: OpenAIEmbedding,
        EmbeddingModelType.COHERE: CohereEmbedding,
        EmbeddingModelType.GEMINI: GeminiEmbedding,
        EmbeddingModelType.MISTRAL: MistralEmbedding,
        EmbeddingModelType.VOYAGE: VoyageEmbedding
    }

    @classmethod
    def create_model(cls, model_type: EmbeddingModelType) -> EmbeddingModel:
        """Create an embedding model instance from an enum type."""
        model_class = cls._model_map.get(model_type)
        if not model_class:
            raise ValueError(f"Unknown embedding model type: {model_type}")
        return model_class()