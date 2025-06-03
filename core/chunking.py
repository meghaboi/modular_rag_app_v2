from abc import ABC, abstractmethod
from typing import List, Dict
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# import numpy as np # Removed unused import
from ..token_utils import TokenCounter  # Relative import for token_utils

# It's good practice to handle potential ImportError for rank_bm25 if it were used here,
# but it's used in HybridSearch, not directly in chunking strategies.
# However, if any chunking strategy started using it, that would be the place.


class ChunkingStrategy(ABC):
    """Abstract class for text chunking strategies"""

    def __init__(self):
        self.token_counter = TokenCounter()

    @abstractmethod
    def chunk_text(
        self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> List[str]:
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

    def chunk_text(
        self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> List[str]:
        """Split text into chunks based on paragraphs"""
        paragraphs = re.split(r"\n\s*\n", text)
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
                # Finalize current chunk if it's not empty
                if current_chunk:
                    chunks.append(current_chunk)

                # Start new chunk, considering overlap from the previous one
                overlap_text = ""
                if chunks:  # Check if there's a previous chunk to overlap with
                    last_chunk_paragraphs = re.split(r"\n\s*\n", chunks[-1])
                    overlap_tokens_calculated = 0
                    temp_overlap_paras = []
                    for para_idx in range(len(last_chunk_paragraphs) - 1, -1, -1):
                        para = last_chunk_paragraphs[para_idx]
                        para_tokens = self.token_counter.count_tokens(para)
                        if overlap_tokens_calculated + para_tokens <= chunk_overlap:
                            temp_overlap_paras.insert(0, para)
                            overlap_tokens_calculated += para_tokens
                        else:
                            break
                    overlap_text = "\n\n".join(temp_overlap_paras)

                current_chunk = overlap_text
                # Add current paragraph to new chunk, ensuring not to duplicate if it was part of overlap
                if not overlap_text or not overlap_text.endswith(paragraph):
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
                current_tokens = self.token_counter.count_tokens(current_chunk)

        if current_chunk:  # Add the last assembled chunk
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

    def chunk_text(
        self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> List[str]:
        """Split text into chunks using a sliding window approach"""
        # This method relies on TokenCounter's split_into_chunks, ensure it handles tokenization.
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
        super().__init__()
        self.levels = max(2, min(levels, 4))

    def chunk_text(
        self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> List[str]:
        paragraphs = re.split(r"\n\s*\n", text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        all_chunks = []

        base_chunks = self._create_base_chunks(paragraphs, chunk_size, chunk_overlap)
        all_chunks.extend(base_chunks)

        for level in range(2, self.levels + 1):
            level_chunk_size = chunk_size * (2 ** (level - 1))
            level_overlap = min(
                chunk_overlap * 2, level_chunk_size // 4
            )  # Adjusted overlap logic slightly for higher levels

            level_chunks = self._create_level_chunks(
                text, base_chunks, level_chunk_size, level_overlap, level
            )
            all_chunks.extend(level_chunks)

        return list(
            dict.fromkeys(all_chunks)
        )  # Remove duplicates while preserving order

    def _create_base_chunks(
        self, paragraphs: List[str], chunk_size: int, chunk_overlap: int
    ) -> List[str]:
        # Re-using ParagraphChunking logic for base chunks might be cleaner if it fits
        # For now, direct implementation:
        chunks = []
        current_chunk = ""
        current_tokens = 0

        for i, paragraph in enumerate(paragraphs):
            paragraph_tokens = self.token_counter.count_tokens(paragraph)

            if current_tokens + paragraph_tokens <= chunk_size or not current_chunk:
                current_chunk = f"{current_chunk}\n\n{paragraph}".strip()
                current_tokens += paragraph_tokens
            else:
                chunks.append(current_chunk)
                # Create overlap
                overlapped_text = ""
                overlapped_tokens = 0
                # Iterate backwards through the paragraphs of the just-added chunk
                # This is a simplified way to get some trailing text for overlap
                temp_paras_for_overlap = re.split(r"\n\s*\n", current_chunk)
                for para_in_overlap in reversed(temp_paras_for_overlap):
                    para_tokens = self.token_counter.count_tokens(para_in_overlap)
                    if overlapped_tokens + para_tokens <= chunk_overlap:
                        overlapped_text = (
                            f"{para_in_overlap}\n\n{overlapped_text}".strip()
                        )
                        overlapped_tokens += para_tokens
                    else:
                        break
                current_chunk = f"{overlapped_text}\n\n{paragraph}".strip()
                current_tokens = self.token_counter.count_tokens(current_chunk)

        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def _create_level_chunks(
        self,
        full_text: str,
        base_chunks: List[str],
        level_chunk_size: int,
        level_overlap: int,
        level: int,
    ) -> List[str]:
        chunks = []
        prefix = f"[L{level}] "  # Not used in current token counting, but good for identification

        # Use sentences for higher levels to allow more granular combination
        # This is a simple sentence split, might need refinement.
        units = re.split(r"(?<=[.!?])\s+", full_text)
        units = [u.strip() for u in units if u.strip()]

        current_chunk_text = ""
        current_tokens = 0

        for i, unit in enumerate(units):
            unit_tokens = self.token_counter.count_tokens(unit)

            if (
                current_tokens + unit_tokens <= level_chunk_size
                or not current_chunk_text
            ):
                current_chunk_text = f"{current_chunk_text} {unit}".strip()
                current_tokens += unit_tokens
            else:
                chunks.append(f"{prefix}{current_chunk_text}")
                # Create overlap for level chunks
                overlapped_text = ""
                overlapped_tokens = 0
                temp_units_for_overlap = current_chunk_text.split(
                    " "
                )  # very basic split for overlap
                for unit_in_overlap in reversed(temp_units_for_overlap):
                    # This is a simplification; ideally, re-tokenize for accuracy
                    unit_ov_tokens = self.token_counter.count_tokens(unit_in_overlap)
                    if overlapped_tokens + unit_ov_tokens <= level_overlap:
                        overlapped_text = f"{unit_in_overlap} {overlapped_text}".strip()
                        overlapped_tokens += unit_ov_tokens
                    else:
                        break
                current_chunk_text = f"{overlapped_text} {unit}".strip()
                current_tokens = self.token_counter.count_tokens(current_chunk_text)

        if current_chunk_text:
            chunks.append(f"{prefix}{current_chunk_text}")
        return chunks

    @property
    def name(self) -> str:
        return "Hierarchical"

    @property
    def description(self) -> str:
        return f"Creates a {self.levels}-level hierarchy of chunks. Combines small chunks for local context with larger chunks for broader context."


class SemanticChunking(ChunkingStrategy):
    """Semantic chunking strategy that splits text based on topic changes"""

    def __init__(
        self,
        similarity_threshold: float = 0.4,
        min_chunk_size: int = 150,
        max_chunk_size_multiplier: float = 1.5,
    ):
        super().__init__()
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size  # Min tokens for a chunk
        # Max chunk size relative to target (e.g. chunk_size * 1.5)
        self.max_chunk_size_multiplier = max_chunk_size_multiplier
        self.vectorizer = TfidfVectorizer(stop_words="english")

    def chunk_text(
        self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> List[str]:
        # Split into sentences as the base unit for semantic chunking
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return []

        if (
            len(sentences) <= 3
        ):  # Not enough sentences for meaningful semantic splitting
            return [" ".join(sentences)]

        try:
            # Compute TF-IDF only for sentences that are not empty or too short
            valid_sentences = [
                s for s in sentences if len(s.split()) > 2
            ]  # Min 3 words
            if (
                not valid_sentences or len(valid_sentences) < 2
            ):  # Need at least 2 valid sentences to compare
                return self._fallback_chunking(text, chunk_size, chunk_overlap)

            tfidf_matrix = self.vectorizer.fit_transform(valid_sentences)
            # Map valid_sentences indices back to original sentences if necessary, or work with valid_sentences
        except ValueError:  # Handles cases where TF-IDF fails (e.g. all stop words)
            return self._fallback_chunking(text, chunk_size, chunk_overlap)

        chunks = []
        current_chunk_sentences = []
        current_chunk_tokens = 0

        # Iterate through original sentences, but use TF-IDF from valid_sentences
        # This requires careful index mapping if not all sentences are valid for TF-IDF
        # For simplicity, this example assumes all sentences are processed by TF-IDF
        # or that TF-IDF is on `sentences` directly if they are all substantial.
        # The code below implies tfidf_matrix corresponds to `sentences`.

        # Re-adjusting to use `sentences` directly with TF-IDF, assuming they are substantial enough
        # If not, pre-filtering and index mapping would be needed.
        try:
            tfidf_matrix = self.vectorizer.fit_transform(sentences)  # Use all sentences
        except ValueError:
            return self._fallback_chunking(text, chunk_size, chunk_overlap)

        for i, sentence in enumerate(sentences):
            sentence_tokens = self.token_counter.count_tokens(sentence)

            if not current_chunk_sentences:  # First sentence for a new chunk
                current_chunk_sentences.append(sentence)
                current_chunk_tokens += sentence_tokens
                continue

            # Compare current sentence with the last sentence of the current chunk
            # More sophisticated: compare with centroid of current_chunk_sentences
            # For simplicity: compare with the previous sentence or the whole current chunk's text

            # Create text from current_chunk_sentences for comparison
            # current_chunk_text_for_comparison = " ".join(current_chunk_sentences) # F841: Unused
            # This comparison is tricky: vectorizer is fit on all sentences.
            # We need similarity between sentence[i] and sentence[i-1] (or chunk average)
            # similarity_matrix[i, i-1] gives similarity between sentence i and sentence i-1
            similarity_to_previous = 0
            if i > 0:  # Can only compare if not the first sentence
                sim_matrix = cosine_similarity(tfidf_matrix[i], tfidf_matrix[i - 1])
                similarity_to_previous = sim_matrix[0, 0]

            # Decision to add to current chunk or start a new one
            # Condition 1: Semantic similarity is high
            # Condition 2: Adding sentence does not exceed max_chunk_size (chunk_size * multiplier)
            # Condition 3: Current chunk is not yet at min_chunk_size

            max_allowed_tokens = chunk_size * self.max_chunk_size_multiplier

            if (
                similarity_to_previous >= self.similarity_threshold
                and current_chunk_tokens + sentence_tokens <= max_allowed_tokens
            ) or current_chunk_tokens < self.min_chunk_size:

                current_chunk_sentences.append(sentence)
                current_chunk_tokens += sentence_tokens
            else:
                # Finalize current chunk
                chunks.append(" ".join(current_chunk_sentences))

                # Start new chunk with overlap and current sentence
                # Simple overlap: take last few sentences from the finalized chunk
                num_overlap_sentences = 2  # Example: 2 sentences for overlap
                # overlap_sentences_text = " ".join( # F841: Unused
                #     current_chunk_sentences[-num_overlap_sentences:]
                # )

                # Ensure overlap does not make the new chunk too big with the current sentence
                # This part of overlap logic needs careful token counting

                # Reset for new chunk
                # current_chunk_sentences = [sentence] # Original: No overlap
                # current_chunk_tokens = sentence_tokens

                # New chunk starts with overlap + current sentence
                # This needs to be careful to avoid re-adding sentences if overlap is complex
                # A simpler overlap: the current sentence begins the new chunk, overlap is handled by `chunk_overlap` at a higher level if desired
                # For now, new chunk starts with current sentence, no explicit sentence-level overlap from previous chunk here.
                # The `chunk_overlap` parameter is not directly used in this semantic split logic per se,
                # it's more of a target for other strategies or a post-processing step.

                # For semantic chunking, "overlap" is more about ensuring context,
                # which is implicitly handled by trying to keep semantically related sentences together.
                # If explicit token overlap is needed, it would typically be applied *after* semantic splitting,
                # or by adjusting split points.

                # Simplification: new chunk starts with current sentence.
                current_chunk_sentences = [sentence]
                # current_tokens = sentence_tokens # F841: Unused as current_chunk_tokens is immediately recalculated
                current_chunk_tokens = sentence_tokens  # Re-initialize current_chunk_tokens for the new chunk

        if current_chunk_sentences:  # Add the last chunk
            chunks.append(" ".join(current_chunk_sentences))

        # Post-process to ensure chunk sizes are within limits (especially if min_chunk_size forced small semantic groups together)
        return self._ensure_chunk_constraints(chunks, chunk_size, chunk_overlap)

    def _fallback_chunking(
        self, text: str, chunk_size: int, chunk_overlap: int
    ) -> List[str]:
        """Fallback to paragraph-based chunking if semantic analysis fails or is not applicable."""
        # Using SlidingWindow as a robust fallback might be better than Paragraph if text has no paragraphs
        fallback_strategy = SlidingWindowChunking()
        return fallback_strategy.chunk_text(text, chunk_size, chunk_overlap)

    def _ensure_chunk_constraints(
        self, chunks: List[str], target_chunk_size: int, target_overlap: int
    ) -> List[str]:
        """Ensure chunks adhere to size constraints, potentially re-chunking or merging."""
        processed_chunks = []
        temp_merged_chunk = ""

        for chunk in chunks:
            current_merged_tokens = self.token_counter.count_tokens(temp_merged_chunk)
            chunk_tokens = self.token_counter.count_tokens(chunk)

            # If a chunk is too large, split it using fallback
            if chunk_tokens > target_chunk_size * self.max_chunk_size_multiplier:
                if temp_merged_chunk:  # Add any existing merged content first
                    processed_chunks.append(temp_merged_chunk)
                    temp_merged_chunk = ""
                # Split the oversized chunk
                split_sub_chunks = self._fallback_chunking(
                    chunk, target_chunk_size, target_overlap
                )
                processed_chunks.extend(split_sub_chunks)
                continue

            # If current chunk can be added to temp_merged_chunk without exceeding max size
            if (
                temp_merged_chunk
                and current_merged_tokens + chunk_tokens
                <= target_chunk_size * self.max_chunk_size_multiplier
            ):
                temp_merged_chunk += " " + chunk  # Merge with space
            # If temp_merged_chunk is too small and current chunk isn't massive, try merging
            elif (
                current_merged_tokens < self.min_chunk_size
                and (current_merged_tokens + chunk_tokens)
                <= target_chunk_size * self.max_chunk_size_multiplier
            ):
                temp_merged_chunk = (
                    f"{temp_merged_chunk} {chunk}".strip()
                    if temp_merged_chunk
                    else chunk
                )
            else:  # Cannot merge or current chunk starts a new segment
                if temp_merged_chunk:  # Finalize the previous merged chunk
                    processed_chunks.append(temp_merged_chunk)
                temp_merged_chunk = (
                    chunk  # Start new temp_merged_chunk with current chunk
                )

        if temp_merged_chunk:  # Add any remaining merged chunk
            processed_chunks.append(temp_merged_chunk)

        # A final pass to ensure no chunk is too small (unless it's the only chunk)
        final_chunks = []
        i = 0
        while i < len(processed_chunks):
            chk = processed_chunks[i]
            chk_tokens = self.token_counter.count_tokens(chk)
            if chk_tokens < self.min_chunk_size and i + 1 < len(processed_chunks):
                # Try to merge with next chunk if this one is too small
                next_chk = processed_chunks[i + 1]
                next_chk_tokens = self.token_counter.count_tokens(next_chk)
                if (
                    chk_tokens + next_chk_tokens
                    <= target_chunk_size * self.max_chunk_size_multiplier
                ):
                    final_chunks.append(f"{chk} {next_chk}")
                    i += 1  # Skip next chunk as it's merged
                else:
                    final_chunks.append(chk)  # Add small chunk as is if cannot merge
            else:
                final_chunks.append(chk)
            i += 1

        return final_chunks if final_chunks else processed_chunks

    @property
    def name(self) -> str:
        return "Semantic"

    @property
    def description(self) -> str:
        return "Divides text based on semantic similarity. Ideal for documents with varying topic structure."


class ChunkingStrategyFactory:
    """Factory for creating chunking strategies"""

    # Store instances to avoid re-creating them if not necessary
    _strategies = {
        "Paragraph-based": ParagraphChunking(),
        "Sliding Window": SlidingWindowChunking(),
        "Hierarchical": HierarchicalChunking(),  # Default levels=2
        "Semantic": SemanticChunking(),  # Default threshold
    }

    @staticmethod
    def get_strategy(strategy_name: str, **kwargs) -> ChunkingStrategy:
        """Get a chunking strategy by name, with optional configuration"""
        if strategy_name == "Hierarchical" and "levels" in kwargs:
            return HierarchicalChunking(levels=kwargs["levels"])
        if strategy_name == "Semantic" and "similarity_threshold" in kwargs:
            return SemanticChunking(
                similarity_threshold=kwargs["similarity_threshold"],
                min_chunk_size=kwargs.get("min_chunk_size", 150),
            )  # Allow min_chunk_size override

        if strategy_name not in ChunkingStrategyFactory._strategies:
            # Fallback or error for unknown strategy
            # For robustness, could default to Sliding Window or raise error
            # raise ValueError(f"Unknown chunking strategy: {strategy_name}")
            return ChunkingStrategyFactory._strategies["Sliding Window"]  # Default

        return ChunkingStrategyFactory._strategies[strategy_name]

    @staticmethod
    def get_all_strategies() -> Dict[str, ChunkingStrategy]:
        """Get all available chunking strategies (instances with default settings)"""
        return ChunkingStrategyFactory._strategies.copy()

    @staticmethod
    def get_available_strategy_names() -> List[str]:
        """Get names of all available chunking strategies"""
        return list(ChunkingStrategyFactory._strategies.keys())
