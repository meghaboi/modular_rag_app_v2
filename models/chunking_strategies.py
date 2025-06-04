from typing import List, Dict, Any
from abc import ABC, abstractmethod
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.token_utils import TokenCounter

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