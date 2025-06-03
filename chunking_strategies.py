from abc import ABC, abstractmethod
from typing import List, Dict
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from token_utils import TokenCounter # Assuming TokenCounter is in token_utils.py at root
import logging # Added for SemanticChunking fallback logging

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
                if current_chunk: # Add the current_chunk before starting a new one with overlap
                    chunks.append(current_chunk)

                # Start new chunk with overlap
                overlap_text = ""
                if chunk_overlap > 0 and current_chunk: # Ensure there's a chunk to overlap from
                    overlap_tokens = 0
                    current_chunk_paragraphs = re.split(r'\n\s*\n', current_chunk)

                    for para_idx in range(len(current_chunk_paragraphs) -1, -1, -1):
                        para_to_add = current_chunk_paragraphs[para_idx]
                        para_tokens = self.token_counter.count_tokens(para_to_add)
                        if overlap_tokens + para_tokens <= chunk_overlap:
                            overlap_text = para_to_add + ("\n\n" + overlap_text if overlap_text else "")
                            overlap_tokens += para_tokens
                        else:
                            break # Stop if adding this paragraph exceeds overlap

                current_chunk = (overlap_text + "\n\n" + paragraph) if overlap_text else paragraph
                current_tokens = self.token_counter.count_tokens(current_chunk)

        if current_chunk: # Add the last remaining chunk
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
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        all_chunks = []

        base_chunks = self._create_base_chunks(paragraphs, chunk_size, chunk_overlap)
        all_chunks.extend(base_chunks)

        for level in range(2, self.levels + 1):
            level_chunk_size = chunk_size * (2 ** (level - 1))
            level_overlap = min(chunk_overlap * level, level_chunk_size // 4) # Ensure overlap is reasonable

            level_specific_chunks = self._create_level_chunks(
                text, base_chunks, level_chunk_size, level_overlap, level
            )
            all_chunks.extend(level_specific_chunks)

        return list(set(all_chunks)) # Remove duplicates that might arise from different levels

    def _create_base_chunks(self, paragraphs: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
        # This is essentially the ParagraphChunking logic, slightly adapted
        chunks = []
        current_chunk = ""
        current_tokens = 0

        for paragraph in paragraphs:
            if not paragraph:
                continue

            paragraph_tokens = self.token_counter.count_tokens(paragraph)

            if current_tokens == 0: # Starting a new chunk
                current_chunk = paragraph
                current_tokens = paragraph_tokens
            elif current_tokens + self.token_counter.count_tokens("\n\n") + paragraph_tokens <= chunk_size:
                current_chunk += "\n\n" + paragraph
                current_tokens += self.token_counter.count_tokens("\n\n") + paragraph_tokens
            else: # Current chunk is full
                chunks.append(current_chunk)
                # Start new chunk with overlap
                overlap_text = ""
                if chunk_overlap > 0:
                    overlap_tokens = 0
                    # Simple overlap: take last few tokens from previous chunk
                    # More sophisticated: take last few sentences/paragraphs of current_chunk
                    # For simplicity, let's use a simpler token-based overlap from the end of current_chunk
                    # This can be improved to be more context-aware (e.g. sentence boundary)

                    # A simple way to get overlap: take the end of the current_chunk
                    # This might not be ideal as it can cut words/sentences.
                    # A better way would be to reconstruct from paragraphs as in ParagraphChunking
                    temp_overlap_chunker = ParagraphChunking()
                    temp_chunks_for_overlap = temp_overlap_chunker.chunk_text(current_chunk, chunk_size, chunk_overlap)
                    if temp_chunks_for_overlap and len(temp_chunks_for_overlap)>1: # check if any overlap was created
                        #this is a crude way to get an overlapping part.
                        #The last part of previous chunk could be used as overlap
                        #This needs more robust logic for hierarchical overlap
                        pass # placeholder for better overlap logic for base chunks in hierarchy

                current_chunk = paragraph # Start new chunk with current paragraph
                current_tokens = paragraph_tokens
                # If overlap_text was generated, it should be prepended to current_chunk here.
                # current_chunk = overlap_text + ("\n\n" if overlap_text else "") + paragraph
                # current_tokens = self.token_counter.count_tokens(current_chunk)

        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def _create_level_chunks(self, full_text: str, base_chunks: List[str],
                           level_chunk_size: int, level_overlap: int, level: int) -> List[str]:
        # This method should combine base_chunks or re-chunk full_text for higher levels
        # For simplicity, let's re-chunk full_text using a sliding window for higher levels
        # This is a simplification; true hierarchical might combine smaller chunks.

        # Prefix to identify chunk level, can be useful for debugging or special processing
        # prefix = f"[L{level}] "

        # Use SlidingWindowChunking for higher levels with adjusted sizes
        # The `text_to_chunk` should ideally be the full_text for broader context
        higher_level_chunker = SlidingWindowChunking()
        level_chunks_text = higher_level_chunker.chunk_text(full_text, level_chunk_size, level_overlap)

        # Optionally, add prefix to each chunk
        # level_chunks_text = [prefix + chunk for chunk in level_chunks_text]
        return level_chunks_text

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
        self.min_chunk_size = min_chunk_size # In tokens
        self.vectorizer = TfidfVectorizer(stop_words='english', min_df=1) # Ensure min_df is at least 1

    def chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """Split text into chunks based on semantic similarity"""
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return []

        if len(paragraphs) <= 3: # Not enough paragraphs for meaningful semantic splitting
            # Fallback to paragraph chunking if too few paragraphs
            return ParagraphChunking().chunk_text(text, chunk_size, chunk_overlap)

        try:
            # Ensure all paragraphs are non-empty before TF-IDF
            valid_paragraphs = [p for p in paragraphs if p]
            if not valid_paragraphs: # If all paragraphs were empty or whitespace
                 return []
            tfidf_matrix = self.vectorizer.fit_transform(valid_paragraphs)
            similarity_matrix = cosine_similarity(tfidf_matrix)
        except ValueError as e:
            logging.warning(f"Semantic chunking TF-IDF error: {e}. Falling back to ParagraphChunking.")
            return self._fallback_chunking(text, chunk_size, chunk_overlap) # Pass original text

        chunks = []
        current_chunk_paragraphs = [valid_paragraphs[0]]
        current_tokens = self.token_counter.count_tokens(valid_paragraphs[0])

        for i in range(1, len(valid_paragraphs)):
            current_paragraph = valid_paragraphs[i]
            paragraph_tokens = self.token_counter.count_tokens(current_paragraph)

            # Compare current_paragraph with the last paragraph of the current_chunk_paragraphs
            # The similarity_matrix is based on valid_paragraphs indexing
            # We need to find the index of the last paragraph of current_chunk_paragraphs in valid_paragraphs
            # This is simplified: compare current_paragraph (i) with previous (i-1)
            similarity_to_previous = similarity_matrix[i, i-1]

            # Condition to merge:
            # 1. Similarity is high OR
            # 2. Current chunk is too small (below min_chunk_size)
            # AND adding current paragraph doesn't exceed max_size (with some leeway, e.g., 1.5*chunk_size)

            should_merge = (similarity_to_previous >= self.similarity_threshold or \
                            current_tokens < self.min_chunk_size) and \
                           (current_tokens + paragraph_tokens <= chunk_size * 1.5)

            if should_merge:
                current_chunk_paragraphs.append(current_paragraph)
                current_tokens += paragraph_tokens + self.token_counter.count_tokens("\n\n") # account for joiner
            else:
                # Finalize current chunk
                chunks.append("\n\n".join(current_chunk_paragraphs))

                # Start new chunk, considering overlap
                # For overlap, take the last part of the just-finalized chunk
                overlap_text = ""
                if chunk_overlap > 0 and chunks:
                    last_finalized_chunk = chunks[-1]
                    # Simple token-based overlap from the end
                    # This could be improved to respect sentence boundaries
                    temp_tokens = []
                    words = last_finalized_chunk.split()
                    temp_overlap_tokens = 0
                    for word_idx in range(len(words) -1, -1, -1):
                        word = words[word_idx]
                        word_tok_count = self.token_counter.count_tokens(word + " ") # Approx with space
                        if temp_overlap_tokens + word_tok_count <= chunk_overlap:
                            temp_tokens.insert(0, word)
                            temp_overlap_tokens += word_tok_count
                        else:
                            break
                    overlap_text = " ".join(temp_tokens)

                current_chunk_paragraphs = [current_paragraph]
                current_tokens = paragraph_tokens
                if overlap_text:
                     # Prepend overlap, ensuring no double newlines if paragraph is empty
                    current_chunk_paragraphs.insert(0, overlap_text)
                    current_tokens += self.token_counter.count_tokens(overlap_text + "\n\n")


        if current_chunk_paragraphs: # Add the last remaining chunk
            chunks.append("\n\n".join(current_chunk_paragraphs))

        return self._ensure_chunk_constraints(chunks, chunk_size) # Ensure constraints as a final step

    def _fallback_chunking(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Fallback to paragraph-based chunking if semantic analysis fails"""
        # Pass the original full text to the fallback chunker
        paragraph_chunker = ParagraphChunking()
        return paragraph_chunker.chunk_text(text, chunk_size, chunk_overlap)

    def _ensure_chunk_constraints(self, chunks: List[str], max_size: int) -> List[str]:
        """Ensure chunks don't exceed maximum size, and handle min_size if needed"""
        # This method might further split chunks if they are too large.
        # For simplicity, the current semantic chunker tries to adhere to max_size * 1.5 during creation.
        # A more robust version would re-chunk oversized ones here.
        # For now, just return them, assuming the creation logic is mostly sufficient.
        # A proper implementation would use a basic chunker (e.g. SlidingWindow) to split oversized chunks.

        final_chunks = []
        for chunk in chunks:
            if self.token_counter.count_tokens(chunk) > max_size * 1.5: # If still too large
                # Fallback to split the oversized chunk
                 logging.warning(f"Semantic chunking produced an oversized chunk ({self.token_counter.count_tokens(chunk)} tokens). Splitting further.")
                 split_chunks = ParagraphChunking().chunk_text(chunk, max_size, max_size // 5) # Use smaller overlap for splits
                 final_chunks.extend(split_chunks)
            else:
                final_chunks.append(chunk)
        return final_chunks

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
        # Ensure names match keys exactly if used elsewhere (e.g. enums)
        strategies = {
            "Paragraph-based": ParagraphChunking(),
            "Sliding Window": SlidingWindowChunking(),
            "Hierarchical": HierarchicalChunking(),
            "Semantic": SemanticChunking()
        }

        if strategy_name not in strategies:
            # Fallback or error
            logging.warning(f"Unknown chunking strategy: {strategy_name}. Defaulting to Paragraph-based.")
            return ParagraphChunking() # Default to a safe strategy
            # raise ValueError(f"Unknown chunking strategy: {strategy_name}")

        return strategies[strategy_name]

    @staticmethod
    def get_all_strategies() -> Dict[str, ChunkingStrategy]:
        """Get all available chunking strategies"""
        return {
            "Paragraph-based": ParagraphChunking(),
            "Sliding Window": SlidingWindowChunking(),
            "Hierarchical": HierarchicalChunking(), # Default levels
            "Semantic": SemanticChunking() # Default threshold/min_size
        }
