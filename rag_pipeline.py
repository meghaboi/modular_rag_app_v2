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
from token_utils import TokenCounter
import logging
import time

class HybridSearch:
    """Combines dense vector search with sparse keyword search (BM25)"""
    
    def __init__(self, alpha: float = 0.5):
        """
        Initialize hybrid search
        
        Args:
            alpha: Weight for vector search scores (1-alpha = weight for BM25)
        """
        self.alpha = alpha
        self.documents = []
        self.bm25 = None
        self.doc_embeddings = None
        
    def index_documents(self, documents: List[str], embeddings: List[List[float]]) -> None:
        """Index documents for both vector search and BM25"""
        self.documents = documents
        self.doc_embeddings = np.array(embeddings)
        
        # Tokenize documents for BM25
        tokenized_docs = [self._tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25"""
        # Convert to lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\w+', text)
        return tokens
    
    def search(self, query: str, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Perform hybrid search using both vector similarity and BM25
        
        Args:
            query: Text query for keyword search
            query_embedding: Vector embedding of the query
            top_k: Number of results to return
            
        Returns:
            List of tuples with (document, score)
        """
        if not self.documents or len(self.documents) == 0:
            return []
        
        # Vector search scores
        vector_scores = self._vector_search(query_embedding)
        
        # BM25 search scores
        bm25_scores = self._bm25_search(query)
        
        # Normalize scores to [0, 1] range
        vector_scores_norm = self._normalize_scores(vector_scores)
        bm25_scores_norm = self._normalize_scores(bm25_scores)
        
        # Combine scores with alpha weighting
        combined_scores = self.alpha * vector_scores_norm + (1 - self.alpha) * bm25_scores_norm
        
        # Get top k results
        top_indices = np.argsort(-combined_scores)[:top_k]
        
        results = [(self.documents[i], combined_scores[i]) for i in top_indices]
        return results
    
    def _vector_search(self, query_embedding: List[float]) -> np.ndarray:
        """Calculate vector similarity scores for all documents"""
        query_embedding = np.array(query_embedding)
        
        # Calculate cosine similarity
        # Normalize vectors for cosine similarity
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm
            
        # Calculate dot product for normalized vectors (equal to cosine similarity)
        doc_norms = np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True)
        normalized_embeddings = np.divide(self.doc_embeddings, doc_norms, 
                                         where=doc_norms != 0)
        
        similarities = np.dot(normalized_embeddings, query_embedding)
        return similarities
    
    def _bm25_search(self, query: str) -> np.ndarray:
        """Calculate BM25 scores for all documents"""
        query_tokens = self._tokenize(query)
        scores = np.array(self.bm25.get_scores(query_tokens))
        return scores
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range"""
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score == min_score:
            return np.ones_like(scores)
            
        normalized = (scores - min_score) / (max_score - min_score)
        return normalized

class ChunkingStrategy(ABC):
    """Abstract class for text chunking strategies"""
    
    def __init__(self):
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
    
    def __init__(self, embedding_model, vector_store, 
                 llm, reranker=None, top_k=3,
                 chunking_strategy=None, chunk_size=1000, 
                 chunk_overlap=200, evaluation_mode=False):
        """Initialize the RAG pipeline with the selected components"""
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.reranker = reranker
        self.llm = llm
        self.top_k = top_k
        self.documents = []
        self.chunking_strategy = chunking_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.evaluation_mode = evaluation_mode
        self.last_evaluation_scores = None  # Store the last evaluation scores
        self.last_metrics = {}  # Store the last performance metrics
    
    def initialize(self, file_path: str) -> None:
        """Initialize the pipeline with a document file"""
        self.index_documents(file_path, self.chunk_size, self.chunk_overlap)
    
    def index_documents(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        """Index documents from a file"""
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split text into chunks using the selected strategy
        chunks = self.chunking_strategy.chunk_text(text, chunk_size, chunk_overlap)
        self.documents = chunks
        
        # Get embeddings for chunks
        embeddings = self.embedding_model.embed_documents(chunks)
        
        # Add chunks to vector store
        self.vector_store.add_documents(chunks, embeddings)
    
    def retrieve_context(self, query: str) -> list:
        """Retrieve relevant contexts for a given query"""
        # Get query embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Retrieve documents - check if vector store supports hybrid search
        if hasattr(self.vector_store, 'search') and 'query' in self.vector_store.search.__code__.co_varnames:
            # Vector store supports hybrid search
            retrieved_docs = self.vector_store.search(query_embedding, self.top_k, query=query)
        else:
            # Standard vector search
            retrieved_docs = self.vector_store.search(query_embedding, self.top_k)
            
        retrieved_texts = [doc[0] for doc in retrieved_docs]
        
        # Apply reranking if available
        if self.reranker and retrieved_texts:
            reranked_docs = self.reranker.rerank(query, retrieved_texts)
            # Select top 5 chunks after reranking
            reranked_docs = reranked_docs[:5]
            retrieved_texts = [doc[0] for doc in reranked_docs]
        
        return retrieved_texts
    
    def run(self, query: str) -> str:
        """Process a query and return the response (non-streaming)"""
        # Get context
        retrieved_texts = self.retrieve_context(query)
        
        # Combine retrieved documents
        context = "\n\n".join(retrieved_texts)
        
        # Generate response
        response = self.llm.generate(query, context, evaluation_mode=self.evaluation_mode)
        
        return response
    
    def stream_run(self, query: str):
        """Process a query and stream the response
        
        In evaluation mode, this will use non-streaming to maintain consistency
        """
        # If we're in evaluation mode, use the non-streaming method instead
        if self.evaluation_mode:
            yield self.run(query)
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
                
    def process_query(self, query: str) -> Tuple[str, List[str], Dict[str, Any]]:
        """Process a query and return the response, retrieved contexts, and metrics"""
        try:
            start_time = time.time()
            
            # Get contexts from vector store
            contexts = self.retrieve_context(query)
            
            # Generate response
            response = self.llm.generate(query, context="\n".join(contexts), evaluation_mode=self.evaluation_mode)
            
            # Calculate total time
            total_time = time.time() - start_time
            
            # Count tokens
            token_counter = TokenCounter(model_name=self.llm.get_model_name())
            input_tokens = token_counter.count_tokens(query + "\n".join(contexts))
            output_tokens = token_counter.count_tokens(response)
            
            # Store metrics
            self.last_metrics = {
                "total_time": total_time,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            }
            
            return response, contexts, self.last_metrics
            
        except Exception as e:
            logging.error(f"Error processing query: {e}")
            raise

    def evaluate_response(self, query: str, response: str, contexts: List[str], ground_truth: str) -> Dict[str, float]:
        """Evaluate the response using RAGAS metrics"""
        try:
            from evaluator import EvaluatorFactory
            from enums import EvaluationBackendType, EvaluationMetricType
            
            # Use RAGAS_V2 for consistency with permutation evaluations
            evaluator = EvaluatorFactory.create_evaluator(
                EvaluationBackendType.RAGAS_V2, 
                EvaluationMetricType.get_metrics_for_backend(EvaluationBackendType.RAGAS_V2)
            )
            
            scores = evaluator.evaluate(
                query=query,
                response=response,
                contexts=contexts,
                ground_truth=ground_truth
            )
            
            # Add performance metrics to scores
            if self.last_metrics:
                scores.update(self.last_metrics)
            
            # Store the scores
            self.last_evaluation_scores = scores
            
            return scores
        except Exception as e:
            logging.error(f"Error evaluating response: {e}")
            self.last_evaluation_scores = None
            raise