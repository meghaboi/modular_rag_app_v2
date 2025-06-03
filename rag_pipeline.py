from typing import List, Dict, Any, Optional, Tuple, Callable
from embedding_models import EmbeddingModel
from rerankers import Reranker
from vector_stores import VectorStore
from llm_models import StreamingLLM
import re
from abc import ABC, abstractmethod

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import numpy as np  
from token_utils import TokenCounter, TokenCostManager
import logging
import time
from evaluator import BaseEvaluator # Added import
from chunking_strategies import ChunkingStrategy # Import from new file

# HybridSearch class moved to vector_stores.py
# ChunkingStrategy and related classes (ParagraphChunking, etc.) and ChunkingStrategyFactory moved to chunking_strategies.py

class RAGPipeline:
    """RAG Pipeline that combines all components with streaming support"""
    # This class itself does not need to be here if it's just a container for Indexer, Retriever, etc.
    # However, it's kept for now as per the current structure.
    
class Indexer:
    """Handles document indexing including chunking, embedding, and storage."""

    def __init__(self, chunking_strategy: ChunkingStrategy,
                 embedding_model: EmbeddingModel,
                 vector_store: VectorStore):
        """
        Initialize the Indexer.

        Args:
            chunking_strategy: The strategy to use for chunking text.
            embedding_model: The model to use for creating embeddings.
            vector_store: The store to save document chunks and embeddings.
        """
        # Type check is good, ensures the passed object is of the expected abstract type
        if not isinstance(chunking_strategy, ChunkingStrategy):
            raise TypeError("chunking_strategy must be an instance of ChunkingStrategy from chunking_strategies.py")

        self.chunking_strategy = chunking_strategy
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.documents_indexed = [] # Keep track of indexed document chunks

    def index_documents(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        """
        Index documents from a file.

        Args:
            file_path: Path to the document file.
            chunk_size: Maximum number of tokens per chunk.
            chunk_overlap: Number of tokens to overlap between chunks.
        """
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split text into chunks using the selected strategy
        logging.info(f"Starting chunking with strategy: {self.chunking_strategy.name}, size: {chunk_size}, overlap: {chunk_overlap}")
        chunks = self.chunking_strategy.chunk_text(text, chunk_size, chunk_overlap)
        self.documents_indexed = chunks # Store the actual text chunks
        logging.info(f"Created {len(chunks)} chunks.")
        
        # Get embeddings for chunks
        logging.info("Starting embedding of document chunks...")
        embeddings = self.embedding_model.embed_documents(chunks)
        logging.info(f"Created {len(embeddings)} embeddings.")
        
        # Add chunks to vector store
        logging.info("Adding documents and embeddings to vector store...")
        self.vector_store.add_documents(chunks, embeddings)
        logging.info("Document indexing complete.")

class Retriever:
    """Handles context retrieval including query embedding, vector search, and reranking."""

    def __init__(self, embedding_model: EmbeddingModel,
                 vector_store: VectorStore,
                 top_k: int,
                 reranker: Optional[Reranker] = None):
        """
        Initialize the Retriever.

        Args:
            embedding_model: The model to use for creating query embeddings.
            vector_store: The store to search for relevant document chunks.
            top_k: The number of top documents to retrieve from the vector store.
            reranker: Optional reranker to re-score and sort documents.
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.reranker = reranker
        self.top_k = top_k

    def retrieve_context(self, query: str) -> List[str]:
        """
        Retrieve relevant contexts for a given query.

        Args:
            query: The user's query string.

        Returns:
            A list of context strings.
        """
        logging.info(f"Retrieving context for query: '{query}' with top_k={self.top_k}")
        # Get query embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Retrieve documents - check if vector store supports hybrid search
        if hasattr(self.vector_store, 'search') and 'query' in self.vector_store.search.__code__.co_varnames:
            # Vector store supports hybrid search
            retrieved_docs_with_scores = self.vector_store.search(query_embedding, self.top_k, query=query)
        else:
            # Standard vector search
            retrieved_docs_with_scores = self.vector_store.search(query_embedding, self.top_k)
        
        retrieved_texts = [doc_tuple[0] for doc_tuple in retrieved_docs_with_scores]
        logging.info(f"Retrieved {len(retrieved_texts)} documents from vector store.")
        
        # Apply reranking if available
        if self.reranker and retrieved_texts:
            logging.info(f"Reranking {len(retrieved_texts)} documents.")
            reranked_docs_with_scores = self.reranker.rerank(query, retrieved_texts)
            # Consistent with previous logic, reranker might return more, but we take top N (e.g. 5 or self.top_k)
            # Let's make this configurable or stick to a constant like 5 for now after reranking.
            # For this refactor, sticking to previous `min(5, ...)`
            top_n_reranked = reranked_docs_with_scores[:min(5, len(reranked_docs_with_scores))]
            retrieved_texts = [doc_tuple[0] for doc_tuple in top_n_reranked]
            logging.info(f"Number of documents after reranking: {len(retrieved_texts)}")
        else:
            logging.info(f"No reranker applied or no initial documents to rerank. Number of documents: {len(retrieved_texts)}")

        return retrieved_texts

class RAGPipeline:
    """RAG Pipeline that combines all components with streaming support"""

    def __init__(self,
                 llm: StreamingLLM,
                 indexer: Indexer,
                 retriever: Retriever,
                 evaluator: Optional[BaseEvaluator] = None, # Added evaluator
                 evaluation_mode: bool = False):
        """Initialize the RAG pipeline with the selected components"""
        self.llm = llm
        self.indexer = indexer
        self.retriever = retriever
        self.evaluator = evaluator # Store the evaluator instance
        self.evaluation_mode = evaluation_mode
        self.last_evaluation_scores = None
        self.last_metrics = {}
        self.last_llm_usage = None

    def initialize(self, file_path: str, chunk_size: int, chunk_overlap: int) -> None:
        """Initialize the pipeline by indexing documents via the Indexer."""
        logging.info(f"RAGPipeline: Initializing and indexing file: {file_path}")
        self.indexer.index_documents(file_path, chunk_size, chunk_overlap)

    # retrieve_context method removed
    
    def run(self, query: str) -> Tuple[str, List[str], Dict[str, Any]]:
        """Process a query and return the response, contexts, and metrics (non-streaming)"""
        start_time = time.time()

        # Get context using the Retriever
        retrieved_texts = self.retriever.retrieve_context(query)

        # Combine retrieved documents
        self.token_counter = TokenCounter()
    
    @abstractmethod
    def chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """Split text into chunks with the specified strategy
        
        Args:
            text: Text to chunk
            chunk_size: Maximum number of tokens per chunk
            chunk_overlap: Number of tokens to overlap between chunks
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the chunking strategy"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Return a brief description of the chunking strategy"""
        pass

class ParagraphChunking(ChunkingStrategy):
    """Paragraph-based chunking strategy that respects paragraph boundaries"""
    
    def chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """Split text into chunks based on paragraphs"""
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            paragraph_tokens = self.token_counter.count_tokens(paragraph)
            
            if current_tokens + paragraph_tokens <= chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                current_tokens += paragraph_tokens
            else:
                chunks.append(current_chunk)
                # Start new chunk with overlap
                overlap_text = ""
                overlap_tokens = 0
                current_chunk_paragraphs = re.split(r'\n\s*\n', current_chunk)
                
                # Calculate overlap
                for para in reversed(current_chunk_paragraphs):
                    para_tokens = self.token_counter.count_tokens(para)
                    if overlap_tokens + para_tokens <= chunk_overlap:
                        overlap_text = para + "\n\n" + overlap_text if overlap_text else para
                        overlap_tokens += para_tokens
                    else:
                        break
                
                current_chunk = overlap_text + paragraph if overlap_text else paragraph
                current_tokens = self.token_counter.count_tokens(current_chunk)
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    @property
    def name(self) -> str:
        return "Paragraph-based"
    
    @property
    def description(self) -> str:
        return "Splits text at paragraph boundaries. Good for preserving logical content structure."

class SlidingWindowChunking(ChunkingStrategy):
    """Sliding window chunking strategy that uses fixed-size chunks with overlap"""
    
    def chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """Split text into chunks using a sliding window approach"""
        return self.token_counter.split_into_chunks(text, chunk_size, chunk_overlap)
    
    @property
    def name(self) -> str:
        return "Sliding Window"
    
    @property
    def description(self) -> str:
        return "Uses fixed-size windows with overlap. Better for dense text where topics span multiple paragraphs."

class HierarchicalChunking(ChunkingStrategy):
    """Hierarchical chunking strategy that creates multi-level chunks"""
    
    def __init__(self, levels: int = 2):
        """
        Initialize hierarchical chunking with specified number of levels
        
        Args:
            levels: Number of hierarchical levels (default: 2)
        """
        super().__init__()
        self.levels = max(2, min(levels, 4))  # Constrain between 2-4 levels
    
    def chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """
        Split text into hierarchical chunks of varying sizes
        
        For each level, chunks are created with progressively larger sizes:
        - Level 1: Base level chunks (chunk_size)
        - Level 2: 2x chunk_size with overlap
        - Level 3: 4x chunk_size with overlap
        - Level 4: 8x chunk_size with overlap
        
        Returns a combined list of all chunks from all levels
        """
        # First split text into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        all_chunks = []
        
        # Level 1: Create base chunks
        base_chunks = self._create_base_chunks(paragraphs, chunk_size, chunk_overlap)
        all_chunks.extend(base_chunks)
        
        # Level 2+: Create progressively larger chunks
        for level in range(2, self.levels + 1):
            level_chunk_size = chunk_size * (2 ** (level - 1))
            level_overlap = min(chunk_overlap * level, level_chunk_size // 4)
            
            level_chunks = self._create_level_chunks(
                text, base_chunks, level_chunk_size, level_overlap, level
            )
            all_chunks.extend(level_chunks)
        
        return all_chunks
    
    def _create_base_chunks(self, paragraphs: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
        """Create base-level chunks from paragraphs"""
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for paragraph in paragraphs:
            if not paragraph:
                continue
            
            paragraph_tokens = self.token_counter.count_tokens(paragraph)
            
            if current_tokens + paragraph_tokens <= chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                current_tokens += paragraph_tokens
            else:
                chunks.append(current_chunk)
                # Start new chunk with overlap
                overlap_text = ""
                overlap_tokens = 0
                current_chunk_paragraphs = re.split(r'\n\s*\n', current_chunk)
                
                for para in reversed(current_chunk_paragraphs):
                    para_tokens = self.token_counter.count_tokens(para)
                    if overlap_tokens + para_tokens <= chunk_overlap:
                        overlap_text = para + "\n\n" + overlap_text if overlap_text else para
                        overlap_tokens += para_tokens
                    else:
                        break
                
                current_chunk = overlap_text + paragraph if overlap_text else paragraph
                current_tokens = self.token_counter.count_tokens(current_chunk)
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _create_level_chunks(self, full_text: str, base_chunks: List[str], 
                           level_chunk_size: int, level_overlap: int, level: int) -> List[str]:
        """Create higher-level chunks from base chunks or full text"""
        chunks = []
        prefix = f"[L{level}] "
        
        # Split full text into sentences or paragraphs as atomic units
        if level == 2:
            units = re.split(r'\n\s*\n', full_text)
            units = [u.strip() for u in units if u.strip()]
        else:
            units = re.split(r'(?<=[.!?])\s+', full_text)
            units = [u.strip() for u in units if u.strip()]
        
        current_chunk = prefix
        current_tokens = self.token_counter.count_tokens(prefix)
        
        for unit in units:
            unit_tokens = self.token_counter.count_tokens(unit)
            
            if current_tokens + unit_tokens <= level_chunk_size:
                if current_tokens > self.token_counter.count_tokens(prefix):
                    current_chunk += "\n\n" if level == 2 else " "
                current_chunk += unit
                current_tokens += unit_tokens
            else:
                chunks.append(current_chunk)
                
                # Calculate overlap
                overlap_text = prefix
                overlap_tokens = self.token_counter.count_tokens(prefix)
                
                chunk_units = current_chunk[len(prefix):].split("\n\n" if level == 2 else " ")
                overlap_start_idx = 0
                
                for i in range(len(chunk_units) - 1, -1, -1):
                    unit_tokens = self.token_counter.count_tokens(chunk_units[i])
                    if overlap_tokens + unit_tokens <= level_overlap:
                        overlap_tokens += unit_tokens
                        overlap_start_idx = i
                    else:
                        break
                
                overlap_text = prefix
                if overlap_start_idx > 0:
                    overlap_units = chunk_units[overlap_start_idx:]
                    overlap_text += ("\n\n" if level == 2 else " ").join(overlap_units)
                
                current_chunk = overlap_text
                if current_tokens > self.token_counter.count_tokens(prefix):
                    current_chunk += "\n\n" if level == 2 else " "
                current_chunk += unit
                current_tokens = self.token_counter.count_tokens(current_chunk)
        
        if current_tokens > self.token_counter.count_tokens(prefix):
            chunks.append(current_chunk)
        
        return chunks
    
    @property
    def name(self) -> str:
        return "Hierarchical"
    
    @property
    def description(self) -> str:
        return f"Creates a {self.levels}-level hierarchy of chunks with different sizes. Combines small chunks for local context with larger chunks for broader context. Best for complex documents with nested structure."

class SemanticChunking(ChunkingStrategy):
    """Semantic chunking strategy that splits text based on topic changes"""
    
    def __init__(self, similarity_threshold: float = 0.5, min_chunk_size: int = 200):
        """
        Initialize semantic chunking
        
        Args:
            similarity_threshold: Threshold for determining topic change (0-1)
            min_chunk_size: Minimum size of chunks to avoid overly small chunks
        """
        super().__init__()
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.vectorizer = TfidfVectorizer(stop_words='english')
    
    def chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """Split text into chunks based on semantic similarity"""
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        if not paragraphs:
            return []
        
        if len(paragraphs) <= 3:
            return paragraphs
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform(paragraphs)
            similarity_matrix = cosine_similarity(tfidf_matrix)
        except ValueError:
            return self._fallback_chunking(paragraphs, chunk_size, chunk_overlap)
        
        chunks = []
        current_chunk_paragraphs = [paragraphs[0]]
        current_tokens = self.token_counter.count_tokens(paragraphs[0])
        
        for i in range(1, len(paragraphs)):
            current_paragraph = paragraphs[i]
            paragraph_tokens = self.token_counter.count_tokens(current_paragraph)
            
            similarities = [similarity_matrix[i][j] for j in range(i) 
                            if paragraphs[j] in current_chunk_paragraphs]
            avg_similarity = np.mean(similarities) if similarities else 0
            
            if (avg_similarity >= self.similarity_threshold and 
                current_tokens + paragraph_tokens <= chunk_size * 1.5) or \
               current_tokens < self.min_chunk_size:
                current_chunk_paragraphs.append(current_paragraph)
                current_tokens += paragraph_tokens
            else:
                chunks.append("\n\n".join(current_chunk_paragraphs))
                
                overlap_paragraphs = []
                overlap_tokens = 0
                
                for para in reversed(current_chunk_paragraphs):
                    para_tokens = self.token_counter.count_tokens(para)
                    if overlap_tokens + para_tokens <= chunk_overlap:
                        overlap_paragraphs.insert(0, para)
                        overlap_tokens += para_tokens
                    else:
                        break
                
                current_chunk_paragraphs = overlap_paragraphs + [current_paragraph]
                current_tokens = self.token_counter.count_tokens("\n\n".join(current_chunk_paragraphs))
        
        if current_chunk_paragraphs:
            chunks.append("\n\n".join(current_chunk_paragraphs))
        
        return self._ensure_chunk_constraints(chunks, chunk_size)
    
    def _fallback_chunking(self, paragraphs: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
        """Fallback to paragraph-based chunking if semantic analysis fails"""
        paragraph_chunker = ParagraphChunking()
        text = "\n\n".join(paragraphs)
        return paragraph_chunker.chunk_text(text, chunk_size, chunk_overlap)
    
    def _ensure_chunk_constraints(self, chunks: List[str], max_size: int) -> List[str]:
        """Ensure chunks don't exceed maximum size"""
        result = []
        for chunk in chunks:
            if self.token_counter.count_tokens(chunk) <= max_size * 1.5:
                result.append(chunk)
            else:
                paragraph_chunker = ParagraphChunking()
                split_chunks = paragraph_chunker.chunk_text(chunk, max_size)
                result.extend(split_chunks)
        return result
    
    @property
    def name(self) -> str:
        return "Semantic"
    
    @property
    def description(self) -> str:
        return "Divides text based on semantic similarity and topic shifts. Ideal for documents with varying topic structure and lengths."

class ChunkingStrategyFactory:
    """Factory for creating chunking strategies"""
    
    @staticmethod
    def get_strategy(strategy_name: str) -> ChunkingStrategy:
        """Get a chunking strategy by name"""
        strategies = {
            "Paragraph-based": ParagraphChunking(),
            "Sliding Window": SlidingWindowChunking(),
            "Hierarchical": HierarchicalChunking(),
            "Semantic": SemanticChunking()
        }
        
        if strategy_name not in strategies:
            raise ValueError(f"Unknown chunking strategy: {strategy_name}")
        
        return strategies[strategy_name]
    
    @staticmethod
    def get_all_strategies() -> Dict[str, ChunkingStrategy]:
        """Get all available chunking strategies"""
        return {
            "Paragraph-based": ParagraphChunking(),
            "Sliding Window": SlidingWindowChunking(),
            "Hierarchical": HierarchicalChunking(),
            "Semantic": SemanticChunking()
        }

class RAGPipeline:
    """RAG Pipeline that combines all components with streaming support"""
    
class Indexer:
    """Handles document indexing including chunking, embedding, and storage."""

    def __init__(self, chunking_strategy: ChunkingStrategy,
                 embedding_model: EmbeddingModel,
                 vector_store: VectorStore):
        """
        Initialize the Indexer.

        Args:
            chunking_strategy: The strategy to use for chunking text.
            embedding_model: The model to use for creating embeddings.
            vector_store: The store to save document chunks and embeddings.
        """
        self.chunking_strategy = chunking_strategy
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.documents_indexed = [] # Keep track of indexed document chunks

    def index_documents(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        """
        Index documents from a file.

        Args:
            file_path: Path to the document file.
            chunk_size: Maximum number of tokens per chunk.
            chunk_overlap: Number of tokens to overlap between chunks.
        """
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split text into chunks using the selected strategy
        logging.info(f"Starting chunking with strategy: {self.chunking_strategy.name}, size: {chunk_size}, overlap: {chunk_overlap}")
        chunks = self.chunking_strategy.chunk_text(text, chunk_size, chunk_overlap)
        self.documents_indexed = chunks # Store the actual text chunks
        logging.info(f"Created {len(chunks)} chunks.")
        
        # Get embeddings for chunks
        logging.info("Starting embedding of document chunks...")
        embeddings = self.embedding_model.embed_documents(chunks)
        logging.info(f"Created {len(embeddings)} embeddings.")
        
        # Add chunks to vector store
        logging.info("Adding documents and embeddings to vector store...")
        self.vector_store.add_documents(chunks, embeddings)
        logging.info("Document indexing complete.")

class Retriever:
    """Handles context retrieval including query embedding, vector search, and reranking."""

    def __init__(self, embedding_model: EmbeddingModel,
                 vector_store: VectorStore,
                 top_k: int,
                 reranker: Optional[Reranker] = None):
        """
        Initialize the Retriever.

        Args:
            embedding_model: The model to use for creating query embeddings.
            vector_store: The store to search for relevant document chunks.
            top_k: The number of top documents to retrieve from the vector store.
            reranker: Optional reranker to re-score and sort documents.
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.reranker = reranker
        self.top_k = top_k

    def retrieve_context(self, query: str) -> List[str]:
        """
        Retrieve relevant contexts for a given query.

        Args:
            query: The user's query string.

        Returns:
            A list of context strings.
        """
        logging.info(f"Retrieving context for query: '{query}' with top_k={self.top_k}")
        # Get query embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Retrieve documents - check if vector store supports hybrid search
        if hasattr(self.vector_store, 'search') and 'query' in self.vector_store.search.__code__.co_varnames:
            # Vector store supports hybrid search
            retrieved_docs_with_scores = self.vector_store.search(query_embedding, self.top_k, query=query)
        else:
            # Standard vector search
            retrieved_docs_with_scores = self.vector_store.search(query_embedding, self.top_k)

        retrieved_texts = [doc_tuple[0] for doc_tuple in retrieved_docs_with_scores]
        logging.info(f"Retrieved {len(retrieved_texts)} documents from vector store.")
        
        # Apply reranking if available
        if self.reranker and retrieved_texts:
            logging.info(f"Reranking {len(retrieved_texts)} documents.")
            reranked_docs_with_scores = self.reranker.rerank(query, retrieved_texts)
            # Consistent with previous logic, reranker might return more, but we take top N (e.g. 5 or self.top_k)
            # Let's make this configurable or stick to a constant like 5 for now after reranking.
            # For this refactor, sticking to previous `min(5, ...)`
            top_n_reranked = reranked_docs_with_scores[:min(5, len(reranked_docs_with_scores))]
            retrieved_texts = [doc_tuple[0] for doc_tuple in top_n_reranked]
            logging.info(f"Number of documents after reranking: {len(retrieved_texts)}")
        else:
            logging.info(f"No reranker applied or no initial documents to rerank. Number of documents: {len(retrieved_texts)}")

        return retrieved_texts

class RAGPipeline:
    """RAG Pipeline that combines all components with streaming support"""

    def __init__(self,
                 llm: StreamingLLM,
                 indexer: Indexer,
                 retriever: Retriever,
                 evaluator: Optional[BaseEvaluator] = None, # Added evaluator
                 evaluation_mode: bool = False):
        """Initialize the RAG pipeline with the selected components"""
        self.llm = llm
        self.indexer = indexer
        self.retriever = retriever
        self.evaluator = evaluator # Store the evaluator instance
        self.evaluation_mode = evaluation_mode
        self.last_evaluation_scores = None
        self.last_metrics = {}
        self.last_llm_usage = None

    def initialize(self, file_path: str, chunk_size: int, chunk_overlap: int) -> None:
        """Initialize the pipeline by indexing documents via the Indexer."""
        logging.info(f"RAGPipeline: Initializing and indexing file: {file_path}")
        self.indexer.index_documents(file_path, chunk_size, chunk_overlap)

    # retrieve_context method removed
    
    def run(self, query: str) -> Tuple[str, List[str], Dict[str, Any]]:
        """Process a query and return the response, contexts, and metrics (non-streaming)"""
        start_time = time.time()
        
        # Get context using the Retriever
        retrieved_texts = self.retriever.retrieve_context(query)
        
        # Combine retrieved documents
        context_str = "\n\n".join(retrieved_texts)
        
        # Generate response
        response_text, usage_info = self.llm.generate(query, context_str, evaluation_mode=self.evaluation_mode)
        self.last_llm_usage = usage_info
        
        total_time = time.time() - start_time
        
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        
        if usage_info:
            prompt_tokens = usage_info.get('prompt_tokens', 0)
            completion_tokens = usage_info.get('completion_tokens', 0)
            total_tokens = usage_info.get('total_tokens', 0)
        else: # Fallback if usage_info is None, though unlikely with new changes
            logging.warning("LLM usage_info was None in RAGPipeline.run. Token counts may be estimated.")
            # Fallback to TokenCounter if usage_info is not available
            try:
                token_counter = TokenCounter(model_name=self.llm.get_model_name())
                prompt_tokens = token_counter.count_tokens(query + context_str)
                completion_tokens = token_counter.count_tokens(response_text)
                total_tokens = prompt_tokens + completion_tokens
            except Exception as e:
                logging.error(f"TokenCounter fallback failed in RAGPipeline.run: {e}")
        model_name = self.llm.get_model_name()
        calculated_cost = TokenCostManager.calculate_cost(model_name, prompt_tokens, completion_tokens)
        
        self.last_metrics = {
            "total_time": total_time,
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "llm_cost": calculated_cost if calculated_cost is not None else 0.0
        }
        
        return response_text, retrieved_texts, self.last_metrics
    
    def stream_run(self, query: str):
        """Process a query and stream the response
        
        In evaluation mode, this will use non-streaming to maintain consistency
        """
        # If we're in evaluation mode, use the non-streaming method instead
        if self.evaluation_mode:
            response_text, _, _ = self.run(query) # Discard contexts and metrics for streaming yield
            yield response_text
            return
            
        # Get context
        retrieved_texts = self.retrieve_context(query)
        
        # Combine retrieved documents
        context = "\n\n".join(retrieved_texts)
        
        # Stream response
        for chunk in self.llm.stream_generate(query, context, evaluation_mode=self.evaluation_mode):
            # Ensure we're not returning None values from our generator
            if chunk is not None:
                yield chunk
            else:
                logging.warning("LLM returned None chunk, skipping")
                
    # process_query method removed as its functionality is covered by run()

    def evaluate_response(self, query: str, response: str, contexts: List[str], ground_truth: str) -> Dict[str, float]:
        """Evaluate the response using the configured evaluator."""
        if self.evaluator is None:
            logging.warning("RAGPipeline.evaluate_response called but no evaluator is configured. Returning empty metrics.")
            return {}

        try:
            cost_to_pass = self.last_metrics.get("llm_cost")
            
            scores = self.evaluator.evaluate(
                query=query,
                response=response,
                contexts=contexts,
                ground_truth=ground_truth,
                cost=cost_to_pass
            )
            
            # Add performance metrics to scores (which might include RAGAS scores + cost)
            # self.last_metrics already contains total_time, token counts, and llm_cost
            # The evaluator's scores might also contain a 'cost' if it was passed.
            # We should ensure metrics are not duplicated or overwritten unintentionally.

            # Let's assume evaluator.evaluate() returns the RAG-specific scores (like faithfulness)
            # and we add the performance metrics to it.
            # If cost was passed to evaluator, it might also be in 'scores'.
            # We prioritize self.last_metrics for performance numbers.

            final_scores = scores.copy() # Start with what the evaluator returned
            if self.last_metrics:
                final_scores.update(self.last_metrics) # Add/overwrite with performance metrics
            
            self.last_evaluation_scores = final_scores
            return final_scores
            
        except Exception as e:
            logging.error(f"Error evaluating response with configured evaluator: {e}", exc_info=True)
            self.last_evaluation_scores = None
            # Return performance metrics even if RAG evaluation failed, if available
            return self.last_metrics if self.last_metrics else {}