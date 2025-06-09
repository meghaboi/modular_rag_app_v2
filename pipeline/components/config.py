import os
import json
from typing import Dict, Any
from dataclasses import dataclass

from utils.enums import (
    EmbeddingModelType, RerankerModelType, LLMModelType, VectorStoreType,
    ChunkingStrategyType
)

@dataclass
class PipelineConfig:
    """Configuration class for RAG pipeline settings"""
    file_path: str
    vector_store_type: VectorStoreType
    reranker_type: RerankerModelType
    llm_type: LLMModelType
    chunking_strategy_type: ChunkingStrategyType
    hybrid_alpha: float = 0.5
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 3
    evaluation_mode: bool = False

    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate()

    def _validate(self) -> None:
        """Validate configuration values"""
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
    def create_default(cls, file_path: str) -> 'PipelineConfig':
        """Create a default configuration with the given file path"""
        return cls(
            file_path=file_path,
            vector_store_type=VectorStoreType.CHROMA,
            reranker_type=RerankerModelType.NONE,
            llm_type=LLMModelType.CLAUDE_37_SONNET,
            chunking_strategy_type=ChunkingStrategyType.PARAGRAPH
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging and storage"""
        return {
            "file_path": self.file_path,
            "vector_store": self.vector_store_type.value,
            "reranker": self.reranker_type.value,
            "llm": self.llm_type.value,
            "chunking": self.chunking_strategy_type.value,
            "hybrid_alpha": self.hybrid_alpha,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "evaluation_mode": self.evaluation_mode
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create a PipelineConfig from a dictionary"""
        return cls(
            file_path=config_dict["file_path"],
            vector_store_type=VectorStoreType(config_dict["vector_store"]),
            reranker_type=RerankerModelType(config_dict["reranker"]),
            llm_type=LLMModelType(config_dict["llm"]),
            chunking_strategy_type=ChunkingStrategyType(config_dict["chunking"]),
            hybrid_alpha=config_dict.get("hybrid_alpha", 0.5),
            chunk_size=config_dict.get("chunk_size", 1000),
            chunk_overlap=config_dict.get("chunk_overlap", 200),
            top_k=config_dict.get("top_k", 3),
            evaluation_mode=config_dict.get("evaluation_mode", False)
        )

    def save_to_file(self, file_path: str) -> None:
        """Save configuration to a JSON file"""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, file_path: str) -> 'PipelineConfig':
        """Load configuration from a JSON file"""
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