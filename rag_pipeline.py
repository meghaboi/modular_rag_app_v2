from typing import List, Dict, Any, Optional, Tuple, Callable
from embedding_models import EmbeddingModel
from rerankers import Reranker
from vector_stores import VectorStore
from llm_models import LLM
import re
from abc import ABC, abstractmethod

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import numpy as np  

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
    
    @abstractmethod
    def chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """Split text into chunks with the specified strategy"""
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
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            if len(current_chunk) + len(paragraph) <= chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                chunks.append(current_chunk)
                # Start new chunk with overlap
                overlap_text = ""
                current_chunk_paragraphs = re.split(r'\n\s*\n', current_chunk)
                
                # Calculate overlap
                remaining_size = chunk_overlap
                for para in reversed(current_chunk_paragraphs):
                    if len(para) <= remaining_size:
                        overlap_text = para + "\n\n" + overlap_text if overlap_text else para
                        remaining_size -= len(para)
                    else:
                        break
                
                current_chunk = overlap_text + paragraph if overlap_text else paragraph
        
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
        chunks = []
        text = text.replace('\n', ' ').replace('\r', '')
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Use tokens (words) as the unit for chunking
        tokens = text.split(' ')
        
        # Approximate chunk_size and chunk_overlap in terms of tokens
        # Assuming average word length of 5 characters + 1 for space
        avg_token_size = 6
        token_chunk_size = max(1, chunk_size // avg_token_size)
        token_overlap = max(1, chunk_overlap // avg_token_size)
        
        # Create chunks with sliding window
        for i in range(0, len(tokens), token_chunk_size - token_overlap):
            chunk_tokens = tokens[i:i + token_chunk_size]
            if chunk_tokens:
                chunk = ' '.join(chunk_tokens)
                chunks.append(chunk)
                
            # Break if this is the last chunk
            if i + token_chunk_size >= len(tokens):
                break
        
        return chunks
    
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
        
        # Level 1: Create base chunks (similar to paragraph chunking)
        base_chunks = self._create_base_chunks(paragraphs, chunk_size, chunk_overlap)
        all_chunks.extend(base_chunks)
        
        # Level 2+: Create progressively larger chunks
        for level in range(2, self.levels + 1):
            # Increase chunk size for each level (2x, 4x, 8x)
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
        
        for paragraph in paragraphs:
            if not paragraph:
                continue
                
            if len(current_chunk) + len(paragraph) <= chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                chunks.append(current_chunk)
                # Start new chunk with overlap
                overlap_text = ""
                current_chunk_paragraphs = re.split(r'\n\s*\n', current_chunk)
                
                # Calculate overlap
                remaining_size = chunk_overlap
                for para in reversed(current_chunk_paragraphs):
                    if len(para) <= remaining_size:
                        overlap_text = para + "\n\n" + overlap_text if overlap_text else para
                        remaining_size -= len(para)
                    else:
                        break
                
                current_chunk = overlap_text + paragraph if overlap_text else paragraph
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _create_level_chunks(self, full_text: str, base_chunks: List[str], 
                           level_chunk_size: int, level_overlap: int, level: int) -> List[str]:
        """Create higher-level chunks from base chunks or full text"""
        # For higher levels, we'll create sliding windows over the whole text
        chunks = []
        
        # Add level prefix to each chunk for identification
        prefix = f"[L{level}] "
        
        # Split full text into sentences or paragraphs as atomic units
        if level == 2:
            # Use paragraphs as units for level 2
            units = re.split(r'\n\s*\n', full_text)
            units = [u.strip() for u in units if u.strip()]
        else:
            # Use sentences as units for higher levels
            units = re.split(r'(?<=[.!?])\s+', full_text)
            units = [u.strip() for u in units if u.strip()]
        
        current_chunk = prefix
        current_length = 0
        
        for unit in units:
            unit_length = len(unit)
            
            if current_length + unit_length <= level_chunk_size:
                if current_length > len(prefix):
                    current_chunk += "\n\n" if level == 2 else " "
                current_chunk += unit
                current_length += unit_length
            else:
                chunks.append(current_chunk)
                
                # Calculate overlap by picking units from the end
                overlap_text = prefix
                overlap_length = 0
                
                # Find where we should start for overlap
                chunk_units = current_chunk[len(prefix):].split("\n\n" if level == 2 else " ")
                overlap_start_idx = 0
                
                for i in range(len(chunk_units) - 1, -1, -1):
                    unit_len = len(chunk_units[i])
                    if overlap_length + unit_len <= level_overlap:
                        overlap_length += unit_len
                        overlap_start_idx = i
                    else:
                        break
                
                # Build overlap text
                overlap_text = prefix
                if overlap_start_idx > 0:
                    overlap_units = chunk_units[overlap_start_idx:]
                    overlap_text += ("\n\n" if level == 2 else " ").join(overlap_units)
                
                # Start new chunk with overlap content
                current_chunk = overlap_text
                if current_length > len(prefix):
                    current_chunk += "\n\n" if level == 2 else " "
                current_chunk += unit
                current_length = len(current_chunk)
        
        if current_length > len(prefix):
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
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.vectorizer = TfidfVectorizer(stop_words='english')
    
    def chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """Split text into chunks based on semantic similarity"""
        # First split text into paragraphs as our base units
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        if not paragraphs:
            return []
        
        # If we have very few paragraphs, use them directly
        if len(paragraphs) <= 3:
            return paragraphs
        
        # Create TF-IDF vectors for each paragraph
        try:
            tfidf_matrix = self.vectorizer.fit_transform(paragraphs)
            # Compute similarity matrix
            similarity_matrix = cosine_similarity(tfidf_matrix)
        except ValueError:
            # Fallback if vectorization fails (e.g., with very short paragraphs)
            return self._fallback_chunking(paragraphs, chunk_size, chunk_overlap)
        
        # Group paragraphs into semantic chunks
        chunks = []
        current_chunk_paragraphs = [paragraphs[0]]
        current_chunk_size = len(paragraphs[0])
        
        for i in range(1, len(paragraphs)):
            current_paragraph = paragraphs[i]
            paragraph_length = len(current_paragraph)
            
            # Calculate average similarity with paragraphs in current chunk
            similarities = [similarity_matrix[i][j] for j in range(i) 
                            if paragraphs[j] in current_chunk_paragraphs]
            avg_similarity = np.mean(similarities) if similarities else 0
            
            # Check if paragraph is semantically similar to current chunk
            # OR if current chunk is too small
            if (avg_similarity >= self.similarity_threshold and 
                current_chunk_size + paragraph_length <= chunk_size * 1.5) or \
               current_chunk_size < self.min_chunk_size:
                # Add to current chunk
                current_chunk_paragraphs.append(current_paragraph)
                current_chunk_size += paragraph_length
            else:
                # Create a new chunk
                chunks.append("\n\n".join(current_chunk_paragraphs))
                
                # Start new chunk with overlap
                # Find paragraphs to include for overlap
                overlap_paragraphs = []
                overlap_size = 0
                
                for para in reversed(current_chunk_paragraphs):
                    if overlap_size + len(para) <= chunk_overlap:
                        overlap_paragraphs.insert(0, para)
                        overlap_size += len(para)
                    else:
                        break
                
                # Start new chunk with overlap paragraphs and current paragraph
                current_chunk_paragraphs = overlap_paragraphs + [current_paragraph]
                current_chunk_size = sum(len(p) for p in current_chunk_paragraphs)
        
        # Add the last chunk if not empty
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
            if len(chunk) <= max_size * 1.5:  # Allow some flexibility
                result.append(chunk)
            else:
                # Split oversized chunks using paragraph chunking
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
    """RAG Pipeline that combines all components"""
    
    def __init__(self, embedding_model: EmbeddingModel, vector_store: VectorStore, 
                 llm: LLM, reranker: Optional[Reranker] = None, top_k: int = 3,
                 chunking_strategy: Optional[ChunkingStrategy] = None):
        """Initialize the RAG pipeline with the selected components"""
        self._embedding_model = embedding_model
        self._vector_store = vector_store
        self._reranker = reranker
        self._llm = llm
        self._top_k = top_k
        self._documents = []
        self._chunking_strategy = chunking_strategy or ParagraphChunking()
    
    def index_documents(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        """Index documents from a file"""
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split text into chunks using the selected strategy
        chunks = self._chunking_strategy.chunk_text(text, chunk_size, chunk_overlap)
        self._documents = chunks
        
        # Get embeddings for chunks
        embeddings = self._embedding_model.embed_documents(chunks)
        
        # Add chunks to vector store
        self._vector_store.add_documents(chunks, embeddings)
    
    def process_query(self, query: str) -> Tuple[str, List[str]]:
        """Process a query and return the response and retrieved contexts"""
        # Get query embedding
        query_embedding = self._embedding_model.embed_query(query)
        
        # Retrieve documents - check if vector store supports hybrid search
        if hasattr(self._vector_store, 'search') and 'query' in self._vector_store.search.__code__.co_varnames:
            # Vector store supports hybrid search
            retrieved_docs = self._vector_store.search(query_embedding, self._top_k, query=query)
        else:
            # Standard vector search
            retrieved_docs = self._vector_store.search(query_embedding, self._top_k)
            
        retrieved_texts = [doc[0] for doc in retrieved_docs]
        
        # Apply reranking if available
        if self._reranker and retrieved_texts:
            reranked_docs = self._reranker.rerank(query, retrieved_texts)
            retrieved_texts = [doc[0] for doc in reranked_docs]
        
        # Combine retrieved documents
        context = "\n\n".join(retrieved_texts)
        
        # Generate response
        response = self._llm.generate(query, context)
        
        return response, retrieved_texts