from typing import List, Dict, Any
import os
import time
import random
import logging
from abc import ABC, abstractmethod
from mistralai import Mistral
import tiktoken  # For token counting
from enums import EmbeddingModelType
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingModel(ABC):
    """Abstract base class for embedding models following Interface Segregation Principle"""
    
    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Convert a query string to an embedding vector"""
        pass
    
    @abstractmethod
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Convert a list of document strings to embedding vectors"""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the embedding vectors"""
        pass

class OpenAIEmbedding(EmbeddingModel):
    """OpenAI embedding model implementation"""
    
    def __init__(self):
        """Initialize the OpenAI embedding model"""
        from langchain_openai import OpenAIEmbeddings
        
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key not found in environment variables")
        
        self._model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    def embed_query(self, query: str) -> List[float]:
        """Convert a query string to an embedding vector"""
        return self._model.embed_query(query)
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Convert a list of document strings to embedding vectors"""
        return self._model.embed_documents(documents)
    
    @property
    def dimension(self) -> int:
        """Return the dimension of the embedding vectors"""
        return 1536  # text-embedding-3-small dimension

class CohereEmbedding(EmbeddingModel):
    """Cohere embedding model implementation"""
    
    def __init__(self):
        """Initialize the Cohere embedding model"""
        from langchain_cohere import CohereEmbeddings
        
        if not os.environ.get("COHERE_API_KEY"):
            raise ValueError("Cohere API key not found in environment variables")
        
        self._model = CohereEmbeddings(model="embed-english-v3.0")
    
    def embed_query(self, query: str) -> List[float]:
        """Convert a query string to an embedding vector"""
        return self._model.embed_query(query)
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Convert a list of document strings to embedding vectors"""
        return self._model.embed_documents(documents)
    
    @property
    def dimension(self) -> int:
        """Return the dimension of the embedding vectors"""
        return 1024  # Cohere embed-english-v3.0 dimension

class GeminiEmbedding(EmbeddingModel):
    """Gemini embedding model implementation"""
    
    def __init__(self):
        """Initialize the Gemini embedding model"""
        import google.generativeai as genai
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key not found in environment variables")
        
        # Configure the Google Generative AI library with the API key
        genai.configure(api_key=api_key)
        
        # Initialize the embeddings with the model name
        self._model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-exp-03-07",
            google_api_key=api_key
        )
    
    def embed_query(self, query: str) -> List[float]:
        """Convert a query string to an embedding vector"""
        return self._model.embed_query(query)
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Convert a list of document strings to embedding vectors"""
        return self._model.embed_documents(documents)
    
    @property
    def dimension(self) -> int:
        """Return the dimension of the embedding vectors"""
        return 768  # Gemini embedding-001 dimension

class MistralEmbedding(EmbeddingModel):
    """Mistral embedding model implementation with batching and rate limiting"""
    
    def __init__(self, model_name="mistral-embed", batch_size=20, 
                 initial_delay=1, max_retries=5, max_delay=60):
        """Initialize the Mistral embedding model"""
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("Mistral API key not found in environment variables")
        
        self._client = Mistral(api_key=api_key)
        self._model_name = model_name
        self._batch_size = batch_size
        self._initial_delay = initial_delay
        self._max_retries = max_retries
        self._max_delay = max_delay
        
        # Initialize tokenizer for token counting
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
            
            # If this single text is too large for a batch, we need special handling
            if text_tokens > max_tokens_per_batch:
                # If we have anything in the current batch, add it to batches
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_batch_tokens = 0
                
                # This is a big text that needs special handling
                # For simplicity, we'll just put it in its own batch
                batches.append([text])
                continue
            
            # If adding this text would exceed batch token limit, create a new batch
            if current_batch_tokens + text_tokens > max_tokens_per_batch:
                batches.append(current_batch)
                current_batch = [text]
                current_batch_tokens = text_tokens
            else:
                current_batch.append(text)
                current_batch_tokens += text_tokens
        
        # Don't forget the last batch
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _call_api_with_backoff(self, inputs):
        """Call the Mistral API with exponential backoff for rate limiting"""
        retry_count = 0
        delay = self._initial_delay
        
        while True:
            try:
                response = self._client.embeddings.create(
                    model=self._model_name,
                    inputs=inputs
                )
                # If we get here, the request was successful
                return response
            except Exception as e:
                error_message = str(e)
                
                # Check if it's a rate limit error
                if "429" in error_message and "rate limit" in error_message.lower():
                    retry_count += 1
                    
                    if retry_count > self._max_retries:
                        logger.error(f"Maximum retries ({self._max_retries}) exceeded. Giving up.")
                        raise e
                    
                    # Add jitter to avoid all clients retrying at the same time
                    jitter = random.uniform(0, 0.1 * delay)
                    actual_delay = min(delay + jitter, self._max_delay)
                    
                    logger.warning(f"Rate limit exceeded. Retrying in {actual_delay:.2f} seconds (retry {retry_count}/{self._max_retries})")
                    time.sleep(actual_delay)
                    
                    # Exponential backoff
                    delay = min(delay * 2, self._max_delay)
                else:
                    # If it's not a rate limit error, re-raise
                    raise e
    
    def embed_query(self, query: str) -> List[float]:
        """Convert a query string to an embedding vector with retry logic"""
        response = self._call_api_with_backoff(query)
        return response.data[0].embedding
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Convert a list of document strings to embedding vectors with automatic batching and rate limiting"""
        all_embeddings = []
        
        # Create batches based on token count
        batches = self._batch_texts(documents)
        
        for i, batch in enumerate(batches):
            logger.info(f"Processing batch {i+1}/{len(batches)} with {len(batch)} documents")
            
            # Add delay between batches to avoid rate limiting
            if i > 0:
                time.sleep(1)  # 1 second delay between batches
            
            response = self._call_api_with_backoff(batch)
            
            # Extract embeddings from the response and add to results
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    @property
    def dimension(self) -> int:
        """Return the dimension of the embedding vectors"""
        return 1024  # Dimension for Mistral embeddings

class BGEEmbedding(EmbeddingModel):
    """BGE embedding model implementation using sentence-transformers"""
    
    def __init__(self, model_name="BAAI/bge-large-en-v1.5"):
        """Initialize the BGE embedding model with proper error handling"""
        try:
            # Try importing here to provide better error messages
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "Could not import sentence_transformers. "
                    "Please install it with: pip install -U sentence-transformers"
                )
                
            # Check transformers version
            try:
                import transformers
                logging.info(f"Using transformers version: {transformers.__version__}")
            except ImportError:
                raise ImportError(
                    "Could not import transformers. "
                    "Please install it with: pip install -U transformers"
                )
                
            # Initialize the model
            try:
                self._model = SentenceTransformer(model_name)
            except Exception as e:
                # If we hit the specific 'init_empty_weights' error
                if "init_empty_weights" in str(e):
                    logging.warning(
                        "Encountered 'init_empty_weights' error. "
                        "This is likely due to a version mismatch. "
                        "Attempting to fix by updating dependencies..."
                    )
                    raise ImportError(
                        f"Version compatibility issue detected: {str(e)}. "
                        "Please run: pip install -U transformers torch accelerate safetensors"
                    )
                else:
                    raise
                
        except Exception as e:
            logging.error(f"Failed to initialize BGE model: {str(e)}")
            raise ValueError(f"Failed to initialize BGE model: {str(e)}")
    
    def embed_query(self, query: str) -> List[float]:
        """Convert a query string to an embedding vector
        
        For BGE models, we add a prefix for better retrieval performance
        """
        # BGE models work better with a prefix for queries
        enhanced_query = f"Represent this sentence for retrieval: {query}"
        # Get embeddings and convert to list
        embeddings = self._model.encode(enhanced_query)
        return embeddings.tolist()
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Convert a list of document strings to embedding vectors"""
        # For documents, we don't use a prefix
        embeddings = self._model.encode(documents)
        # Convert numpy arrays to lists
        return embeddings.tolist()
    
    @property
    def dimension(self) -> int:
        """Return the dimension of the embedding vectors"""
        return 1024  # BGE-large-en-v1.5 dimension

class VoyageEmbedding(EmbeddingModel):
    """Voyage AI embedding model implementation with batching and rate limiting"""
    
    def __init__(self, model_name="voyage-3", batch_size=10, 
                 initial_delay=1, max_retries=5, max_delay=60):
        """Initialize the Voyage AI embedding model"""
        api_key = os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            raise ValueError("Voyage API key not found in environment variables")
        
        self._api_key = api_key
        self._model_name = model_name
        self._batch_size = batch_size
        self._initial_delay = initial_delay
        self._max_retries = max_retries
        self._max_delay = max_delay
        self._api_url = "https://api.voyageai.com/v1/embeddings"
    
    def _call_api_with_backoff(self, texts: List[str]) -> List[List[float]]:
        """Call the Voyage AI API with exponential backoff for rate limiting"""
        retry_count = 0
        delay = self._initial_delay
        
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self._model_name,
            "input": texts
        }
        
        while True:
            try:
                response = requests.post(self._api_url, headers=headers, json=payload)
                
                if response.status_code == 200:
                    return response.json()["data"]
                elif response.status_code == 429:
                    # Rate limit error, implement backoff
                    retry_count += 1
                    
                    if retry_count > self._max_retries:
                        logger.error(f"Maximum retries ({self._max_retries}) exceeded. Giving up.")
                        response.raise_for_status()
                    
                    # Add jitter to avoid all clients retrying at the same time
                    jitter = random.uniform(0, 0.1 * delay)
                    actual_delay = min(delay + jitter, self._max_delay)
                    
                    logger.warning(f"Rate limit exceeded. Retrying in {actual_delay:.2f} seconds (retry {retry_count}/{self._max_retries})")
                    time.sleep(actual_delay)
                    
                    # Exponential backoff
                    delay = min(delay * 2, self._max_delay)
                else:
                    # For other errors, raise the exception
                    response.raise_for_status()
            except Exception as e:
                if retry_count >= self._max_retries:
                    logger.error(f"Failed to get embeddings after {retry_count} retries: {str(e)}")
                    raise e
                
                retry_count += 1
                actual_delay = min(delay * (1 + 0.1 * random.random()), self._max_delay)
                logger.warning(f"API call failed: {str(e)}. Retrying in {actual_delay:.2f} seconds (retry {retry_count}/{self._max_retries})")
                time.sleep(actual_delay)
                delay = min(delay * 2, self._max_delay)
    
    def _batch_documents(self, documents: List[str]) -> List[List[str]]:
        """Split documents into batches of specified size"""
        return [documents[i:i + self._batch_size] for i in range(0, len(documents), self._batch_size)]
    
    def embed_query(self, query: str) -> List[float]:
        """Convert a query string to an embedding vector"""
        response_data = self._call_api_with_backoff([query])
        return response_data[0]["embedding"]
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Convert a list of document strings to embedding vectors with automatic batching"""
        all_embeddings = []
        
        # Create batches
        batches = self._batch_documents(documents)
        
        for i, batch in enumerate(batches):
            logger.info(f"Processing batch {i+1}/{len(batches)} with {len(batch)} documents")
            
            # Add delay between batches to avoid rate limiting
            if i > 0:
                time.sleep(1)  # 1 second delay between batches
            
            response_data = self._call_api_with_backoff(batch)
            
            # Extract embeddings from the response and add to results
            batch_embeddings = [item["embedding"] for item in response_data]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    @property
    def dimension(self) -> int:
        """Return the dimension of the embedding vectors"""
        # Voyage-2 has 1024 dimensions
        return 1024

class EmbeddingModelFactory:
    """Factory for creating embedding models using enum type"""
    
    @staticmethod
    def create_model(model_type: EmbeddingModelType) -> EmbeddingModel:
        """Create an embedding model based on the enum type"""
        if model_type == EmbeddingModelType.OPENAI:
            return OpenAIEmbedding()
        elif model_type == EmbeddingModelType.COHERE:
            return CohereEmbedding()
        elif model_type == EmbeddingModelType.GEMINI:
            return GeminiEmbedding()
        elif model_type == EmbeddingModelType.MISTRAL:
            return MistralEmbedding()
        elif model_type == EmbeddingModelType.VOYAGE: 
            return VoyageEmbedding()
        else:
            raise ValueError(f"Unsupported embedding model type: {model_type}")