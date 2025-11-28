from typing import List, Dict, Any, Tuple
from abc import ABC, abstractmethod
import re
import os
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.token_utils import TokenCounter
from prompts import get_provider


class ChunkingStrategy(ABC):
    """Abstract class for text chunking strategies."""

    def __init__(self):
        self.token_counter = TokenCounter()

    @abstractmethod
    def chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """Split text into chunks with the specified strategy.

        Args:
            text: Text to chunk.
            chunk_size: Maximum number of tokens per chunk.
            chunk_overlap: Number of tokens to overlap between chunks.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the chunking strategy."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a brief description of the chunking strategy."""
        pass

    def _chunk_units(self, units: List[str], chunk_size: int, chunk_overlap: int, unit_separator: str) -> List[str]:
        """Generic helper to chunk a list of text units (like paragraphs or sentences)."""
        chunks = []
        current_chunk_units = []
        current_tokens = 0

        for unit in units:
            unit_tokens = self.token_counter.count_tokens(unit)

            if current_tokens + unit_tokens <= chunk_size:
                current_chunk_units.append(unit)
                current_tokens += unit_tokens
            else:
                chunks.append(unit_separator.join(current_chunk_units))
                
                overlap_units = self._get_overlap_units(current_chunk_units, chunk_overlap, unit_separator)
                current_chunk_units = overlap_units + [unit]
                current_tokens = self.token_counter.count_tokens(unit_separator.join(current_chunk_units))

        if current_chunk_units:
            chunks.append(unit_separator.join(current_chunk_units))

        return chunks

    def _get_overlap_units(self, units: List[str], chunk_overlap: int, unit_separator: str) -> List[str]:
        """Calculates the units from the end of a list that fit within the overlap token limit."""
        overlap_units = []
        overlap_tokens = 0
        for unit in reversed(units):
            unit_tokens = self.token_counter.count_tokens(unit)
            # Add separator tokens to all but the last unit in the overlap
            separator_tokens = self.token_counter.count_tokens(unit_separator) if overlap_units else 0

            if overlap_tokens + unit_tokens + separator_tokens <= chunk_overlap:
                overlap_units.insert(0, unit)
                overlap_tokens += unit_tokens + separator_tokens
            else:
                break
        return overlap_units

class ContextualChunking(ChunkingStrategy):
    """Contextual chunking: adds succinct context to each chunk using Claude Haiku and the whole document.

    Note: LLM dependency is created lazily to avoid import-time failures when API keys are missing.
    """
    def __init__(self, base_chunker=None, llm_model=None):
        super().__init__()
        self.base_chunker = base_chunker or SlidingWindowChunking()
        self.llm = llm_model  # may be None; created lazily when needed
        self.prompt_provider = get_provider('contextual_chunking')
        self.logger = logging.getLogger(__name__)

    def _ensure_llm(self):
        """Instantiate a default Claude LLM lazily if not provided and if possible."""
        if self.llm is not None:
            return
        # Only attempt creation if Anthropic API key appears available
        if not os.environ.get("ANTHROPIC_API_KEY"):
            self.logger.warning("ContextualChunking: ANTHROPIC_API_KEY not set; falling back to non-contextual chunks.")
            self.llm = None
            return
        try:
            from models.llm_models import ClaudeLLM  # local import to avoid heavy dependencies at module import
            self.llm = ClaudeLLM(model_name="claude-3-5-haiku-20241022")
        except Exception as e:
            self.logger.warning(f"ContextualChunking: Failed to initialize ClaudeLLM ({e}); proceeding without context.")
            self.llm = None

    def chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        chunks = self.base_chunker.chunk_text(text, chunk_size, chunk_overlap)
        results: List[str] = []

        # Ensure LLM is ready (or decide to skip contextualization)
        self._ensure_llm()

        for chunk in chunks:
            if self.llm is None:
                # No LLM available; return chunk without added context but keep structure
                context = "[Context unavailable]"
            else:
                prompt = self.prompt_provider.get_prompt(
                    'contextual_chunking',
                    WHOLE_DOCUMENT=text,
                    CHUNK_CONTENT=chunk
                )
                try:
                    context, _ = self.llm.generate(prompt)
                except Exception as e:
                    context = f"[Context generation failed: {e}]"
            # Combine context and chunk as a single string to keep downstream expectations (List[str])
            combined = f"[CONTEXT]\n{context}\n\n[CHUNK]\n{chunk}"
            results.append(combined)
        return results

    @property
    def name(self) -> str:
        return "Contextual"

    @property
    def description(self) -> str:
        return "Adds succinct context to each chunk using Claude Haiku and the whole document."

class ParagraphChunking(ChunkingStrategy):
    """Paragraph-based chunking strategy that respects paragraph boundaries."""

    def chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """Split text into chunks based on paragraphs."""
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        if not paragraphs:
            return []
        return self._chunk_units(paragraphs, chunk_size, chunk_overlap, "\n\n")

    @property
    def name(self) -> str:
        return "Paragraph-based"

    @property
    def description(self) -> str:
        return "Splits text at paragraph boundaries. Good for preserving logical content structure."


class SlidingWindowChunking(ChunkingStrategy):
    """Sliding window chunking strategy that uses fixed-size chunks with overlap."""

    def chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """Split text into chunks using a sliding window approach."""
        return self.token_counter.split_into_chunks(text, chunk_size, chunk_overlap)

    @property
    def name(self) -> str:
        return "Sliding Window"

    @property
    def description(self) -> str:
        return "Uses fixed-size windows with overlap. Better for dense text where topics span multiple paragraphs."


class HierarchicalChunking(ChunkingStrategy):
    """Hierarchical chunking strategy that creates multi-level chunks."""

    def __init__(self, levels: int = 2):
        """
        Initialize hierarchical chunking with specified number of levels.

        Args:
            levels: Number of hierarchical levels (default: 2), constrained to 2-4.
        """
        super().__init__()
        self.levels = max(2, min(levels, 4))

    def chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """Split text into a hierarchy of chunks with varying sizes."""
        all_chunks = []

        # Level 1: Base chunks are paragraph-based
        paragraph_chunker = ParagraphChunking()
        base_chunks = paragraph_chunker.chunk_text(text, chunk_size, chunk_overlap)
        all_chunks.extend(base_chunks)

        # Levels 2+: Create progressively larger chunks
        for level in range(2, self.levels + 1):
            level_chunk_size = chunk_size * (2 ** (level - 1))
            level_overlap = min(chunk_overlap * level, level_chunk_size // 4)
            
            level_chunks = self._create_level_chunks(text, level_chunk_size, level_overlap, level)
            all_chunks.extend(level_chunks)

        return all_chunks

    def _create_level_chunks(self, full_text: str, level_chunk_size: int, level_overlap: int, level: int) -> List[str]:
        """Create higher-level chunks from the full text."""
        # Level 2 uses paragraphs as units, higher levels use sentences
        unit_separator = "\n\n" if level == 2 else " "
        regex_pattern = r'\n\s*\n' if level == 2 else r'(?<=[.!?])\s+'
        units = [u.strip() for u in re.split(regex_pattern, full_text) if u.strip()]

        if not units:
            return []

        # Prefix each chunk with its level for identification
        prefix = f"[L{level}] "
        prefixed_chunks = self._chunk_units(units, level_chunk_size - len(prefix), level_overlap, unit_separator)
        return [prefix + chunk for chunk in prefixed_chunks]

    @property
    def name(self) -> str:
        return "Hierarchical"

    @property
    def description(self) -> str:
        return (
            f"Creates a {self.levels}-level hierarchy of chunks with different sizes. "
            f"Combines small chunks for local context with larger chunks for broader context. "
            f"Best for complex documents with nested structure.")


class SemanticChunking(ChunkingStrategy):
    """Semantic chunking strategy that splits text based on topic changes."""
    # Allow chunks to exceed the target size by this factor to avoid tiny chunks
    _CHUNK_SIZE_FLEXIBILITY_FACTOR = 1.5

    def __init__(self, similarity_threshold: float = 0.5, min_chunk_size: int = 200):
        """
        Initialize semantic chunking.

        Args:
            similarity_threshold: Threshold for determining topic change (0-1).
            min_chunk_size: Minimum token size for a chunk.
        """
        super().__init__()
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """Split text into chunks based on semantic similarity."""
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        if not paragraphs:
            return []
        if len(paragraphs) <= 3: # Not enough content for semantic analysis
            return ["\n\n".join(paragraphs)]

        try:
            tfidf_matrix = self.vectorizer.fit_transform(paragraphs)
            similarity_matrix = cosine_similarity(tfidf_matrix)
        except ValueError: # Fallback if TF-IDF fails (e.g., all stop words)
            return self._fallback_chunking(text, chunk_size, chunk_overlap)

        chunks = []
        current_chunk_paragraphs = [paragraphs[0]]
        current_tokens = self.token_counter.count_tokens(paragraphs[0])

        for i in range(1, len(paragraphs)):
            paragraph_tokens = self.token_counter.count_tokens(paragraphs[i])

            if self._should_merge_paragraph(i, paragraphs, similarity_matrix, current_chunk_paragraphs, current_tokens, paragraph_tokens, chunk_size):
                current_chunk_paragraphs.append(paragraphs[i])
                current_tokens += paragraph_tokens
            else:
                chunks.append("\n\n".join(current_chunk_paragraphs))
                
                overlap_paras = self._get_overlap_units(current_chunk_paragraphs, chunk_overlap, "\n\n")
                current_chunk_paragraphs = overlap_paras + [paragraphs[i]]
                current_tokens = self.token_counter.count_tokens("\n\n".join(current_chunk_paragraphs))

        if current_chunk_paragraphs:
            chunks.append("\n\n".join(current_chunk_paragraphs))

        return self._ensure_chunk_constraints(chunks, chunk_size)

    def _should_merge_paragraph(self, para_idx: int, all_paras: List[str], sim_matrix: np.ndarray, 
                                  chunk_paras: List[str], chunk_tokens: int, para_tokens: int, chunk_size: int) -> bool:
        """Determines if the next paragraph should be merged into the current chunk."""
        # Calculate average similarity of the new paragraph to the paragraphs already in the chunk
        similarities = [sim_matrix[para_idx][all_paras.index(p)] for p in chunk_paras]
        avg_similarity = np.mean(similarities) if similarities else 0

        is_similar_enough = avg_similarity >= self.similarity_threshold
        is_under_flex_size = chunk_tokens + para_tokens <= chunk_size * self._CHUNK_SIZE_FLEXIBILITY_FACTOR
        is_too_small = chunk_tokens < self.min_chunk_size

        return is_too_small or (is_similar_enough and is_under_flex_size)

    def _fallback_chunking(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Fallback to paragraph-based chunking if semantic analysis fails."""
        return ParagraphChunking().chunk_text(text, chunk_size, chunk_overlap)

    def _ensure_chunk_constraints(self, chunks: List[str], max_size: int) -> List[str]:
        """Ensure chunks don't exceed the flexible maximum size."""
        final_chunks = []
        for chunk in chunks:
            if self.token_counter.count_tokens(chunk) <= max_size * self._CHUNK_SIZE_FLEXIBILITY_FACTOR:
                final_chunks.append(chunk)
            else:
                # If a chunk is still too big, split it forcefully
                split_chunks = ParagraphChunking().chunk_text(chunk, max_size, max_size // 5)
                final_chunks.extend(split_chunks)
        return final_chunks

    @property
    def name(self) -> str:
        return "Semantic"

    @property
    def description(self) -> str:
        return "Divides text based on semantic similarity and topic shifts. Ideal for documents with varying topic structure and lengths."


class ChunkingStrategyFactory:
    """Factory for creating chunking strategies."""
    _strategies = None

    @classmethod
    def _get_strategies(cls) -> Dict[str, ChunkingStrategy]:
        """Initializes and returns the dictionary of strategies."""
        if cls._strategies is None:
            cls._strategies = {
                "Paragraph-based": ParagraphChunking(),
                "Sliding Window": SlidingWindowChunking(),
                "Hierarchical": HierarchicalChunking(),
                "Semantic": SemanticChunking(),
                "Contextual": ContextualChunking()
            }
        return cls._strategies

    @classmethod
    def get_strategy(cls, strategy_name: str) -> ChunkingStrategy:
        """Get a chunking strategy by name."""
        strategies = cls._get_strategies()
        strategy = strategies.get(strategy_name)
        if not strategy:
            raise ValueError(f"Unknown chunking strategy: {strategy_name}")
        return strategy

    @classmethod
    def get_all_strategies(cls) -> Dict[str, ChunkingStrategy]:
        """Get all available chunking strategies."""
        return cls._get_strategies()