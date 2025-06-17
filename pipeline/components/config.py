import os
import json
from typing import Dict, Any, Type, ClassVar
from dataclasses import dataclass, field

from utils.enums import (
    EmbeddingModelType, RerankerModelType, LLMModelType, VectorStoreType,
    ChunkingStrategyType
)

@dataclass
class PipelineConfig:
    """Configuration class for RAG pipeline settings."""

    # --- Constants for default values ---
    DEFAULT_HYBRID_ALPHA: ClassVar[float] = 0.5
    DEFAULT_CHUNK_SIZE: ClassVar[int] = 1000
    DEFAULT_CHUNK_OVERLAP: ClassVar[int] = 200
    DEFAULT_TOP_K: ClassVar[int] = 3
    DEFAULT_EVALUATION_MODE: ClassVar[bool] = False

    # --- Constants for dictionary keys (for serialization) ---
    class Keys:
        FILE_PATH = "file_path"
        EMBEDDING_MODEL = "embedding_model"
        VECTOR_STORE = "vector_store"
        RERANKER = "reranker"
        LLM = "llm"
        CHUNKING = "chunking"
        HYBRID_ALPHA = "hybrid_alpha"
        CHUNK_SIZE = "chunk_size"
        CHUNK_OVERLAP = "chunk_overlap"
        TOP_K = "top_k"
        EVALUATION_MODE = "evaluation_mode"

    # --- Configuration fields ---
    file_path: str
    embedding_model_type: EmbeddingModelType
    vector_store_type: VectorStoreType
    reranker_type: RerankerModelType
    llm_type: LLMModelType
    chunking_strategy_type: ChunkingStrategyType
    hybrid_alpha: float = field(default=DEFAULT_HYBRID_ALPHA)
    chunk_size: int = field(default=DEFAULT_CHUNK_SIZE)
    chunk_overlap: int = field(default=DEFAULT_CHUNK_OVERLAP)
    top_k: int = field(default=DEFAULT_TOP_K)
    evaluation_mode: bool = field(default=DEFAULT_EVALUATION_MODE)

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate configuration values."""
        if not os.path.exists(self.file_path):
            raise ValueError(f"File path does not exist: {self.file_path}")
        if not 0 <= self.hybrid_alpha <= 1:
            raise ValueError(f"Hybrid alpha must be between 0 and 1, got {self.hybrid_alpha}")
        if self.chunk_size <= 0:
            raise ValueError(f"Chunk size must be positive, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"Chunk overlap must be non-negative, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"Chunk overlap ({self.chunk_overlap}) must be less than chunk size ({self.chunk_size})")
        if self.top_k <= 0:
            raise ValueError(f"Top k must be positive, got {self.top_k}")

    @classmethod
    def create_default(cls: Type['PipelineConfig'], file_path: str) -> 'PipelineConfig':
        """Create a default configuration with the given file path."""
        return cls(
            file_path=file_path,
            embedding_model_type=EmbeddingModelType.OPENAI_EMBEDDINGS,
            vector_store_type=VectorStoreType.CHROMA,
            reranker_type=RerankerModelType.NONE,
            llm_type=LLMModelType.CLAUDE_3_SONNET,
            chunking_strategy_type=ChunkingStrategyType.PARAGRAPH
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to a dictionary for logging and storage."""
        return {
            self.Keys.FILE_PATH: self.file_path,
            self.Keys.EMBEDDING_MODEL: self.embedding_model_type.value,
            self.Keys.VECTOR_STORE: self.vector_store_type.value,
            self.Keys.RERANKER: self.reranker_type.value,
            self.Keys.LLM: self.llm_type.value,
            self.Keys.CHUNKING: self.chunking_strategy_type.value,
            self.Keys.HYBRID_ALPHA: self.hybrid_alpha,
            self.Keys.CHUNK_SIZE: self.chunk_size,
            self.Keys.CHUNK_OVERLAP: self.chunk_overlap,
            self.Keys.TOP_K: self.top_k,
            self.Keys.EVALUATION_MODE: self.evaluation_mode
        }

    @classmethod
    def from_dict(cls: Type['PipelineConfig'], config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create a PipelineConfig from a dictionary."""
        return cls(
            file_path=config_dict[cls.Keys.FILE_PATH],
            embedding_model_type=EmbeddingModelType(config_dict[cls.Keys.EMBEDDING_MODEL]),
            vector_store_type=VectorStoreType(config_dict[cls.Keys.VECTOR_STORE]),
            reranker_type=RerankerModelType(config_dict[cls.Keys.RERANKER]),
            llm_type=LLMModelType(config_dict[cls.Keys.LLM]),
            chunking_strategy_type=ChunkingStrategyType(config_dict[cls.Keys.CHUNKING]),
            hybrid_alpha=config_dict.get(cls.Keys.HYBRID_ALPHA, cls.DEFAULT_HYBRID_ALPHA),
            chunk_size=config_dict.get(cls.Keys.CHUNK_SIZE, cls.DEFAULT_CHUNK_SIZE),
            chunk_overlap=config_dict.get(cls.Keys.CHUNK_OVERLAP, cls.DEFAULT_CHUNK_OVERLAP),
            top_k=config_dict.get(cls.Keys.TOP_K, cls.DEFAULT_TOP_K),
            evaluation_mode=config_dict.get(cls.Keys.EVALUATION_MODE, cls.DEFAULT_EVALUATION_MODE)
        )

    def save_to_file(self, file_path: str) -> None:
        """Save configuration to a JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls: Type['PipelineConfig'], file_path: str) -> 'PipelineConfig':
        """Load configuration from a JSON file."""
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def copy(self) -> 'PipelineConfig':
        """Create a copy of the configuration"""
        return PipelineConfig(
            file_path=self.file_path,
            vector_store_type=self.vector_store_type,
            reranker_type=self.reranker_type,
            llm_type=self.llm_type,
            chunking_strategy_type=self.chunking_strategy_type,
            hybrid_alpha=self.hybrid_alpha,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            top_k=self.top_k,
            evaluation_mode=self.evaluation_mode
        ) 