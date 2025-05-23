from typing import Dict, Any
from dataclasses import dataclass
from enums import (
    EmbeddingModelType,
    RerankerModelType,
    LLMModelType,
    VectorStoreType,
    ChunkingStrategyType
)

@dataclass
class SubjectConfig:
    chunk_size: int
    chunk_overlap: int
    top_k: int
    hybrid_alpha: float = 0.5

# Default configuration values
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_TOP_K = 3
DEFAULT_HYBRID_ALPHA = 0.5

DEFAULT_EMBEDDING_MODEL = EmbeddingModelType.MISTRAL
DEFAULT_VECTOR_STORE = VectorStoreType.CHROMA
DEFAULT_RERANKER_MODEL = RerankerModelType.VOYAGE_2
DEFAULT_LLM_MODEL = LLMModelType.CLAUDE_37_SONNET
DEFAULT_CHUNKING_STRATEGY = ChunkingStrategyType.HIERARCHICAL

# Subject-specific configurations
SUBJECT_CONFIGS: Dict[str, SubjectConfig] = {
    "general": SubjectConfig(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        top_k=DEFAULT_TOP_K,
        hybrid_alpha=DEFAULT_HYBRID_ALPHA
    ),
    "mathematics": SubjectConfig(
        chunk_size=300,  # Smaller chunks for precise mathematical content
        chunk_overlap=100,  # Higher overlap to maintain context
        top_k=5,  # More documents for comprehensive coverage
        hybrid_alpha=0.7  # Higher vector weight for mathematical precision
    ),
    "science": SubjectConfig(
        chunk_size=400,
        chunk_overlap=75,
        top_k=4,
        hybrid_alpha=0.6
    ),
    "history": SubjectConfig(
        chunk_size=600,  # Larger chunks for narrative context
        chunk_overlap=100,
        top_k=5,
        hybrid_alpha=0.4  # Lower vector weight for more keyword matching
    ),
    "literature": SubjectConfig(
        chunk_size=500,
        chunk_overlap=100,
        top_k=4,
        hybrid_alpha=0.5
    ),
    "computer_science": SubjectConfig(
        chunk_size=350,  # Smaller chunks for code and technical content
        chunk_overlap=75,
        top_k=4,
        hybrid_alpha=0.7  # Higher vector weight for technical precision
    ),
    "medicine": SubjectConfig(
        chunk_size=450,
        chunk_overlap=100,
        top_k=5,
        hybrid_alpha=0.6
    ),
    "law": SubjectConfig(
        chunk_size=550,  # Larger chunks for legal context
        chunk_overlap=100,
        top_k=5,
        hybrid_alpha=0.4  # Lower vector weight for more keyword matching
    )
}

def get_subject_config(subject: str) -> SubjectConfig:
    """
    Get the configuration for a specific subject.
    Falls back to general configuration if subject not found.
    """
    return SUBJECT_CONFIGS.get(subject.lower(), SUBJECT_CONFIGS["general"]) 