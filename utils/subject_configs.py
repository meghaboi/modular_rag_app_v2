from typing import Dict, Any
from dataclasses import dataclass
from utils.enums import (
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
DEFAULT_LLM_MODEL = LLMModelType.CLAUDE_4_SONNET
DEFAULT_CHUNKING_STRATEGY = ChunkingStrategyType.HIERARCHICAL

SUBJECT_CONFIGS: Dict[str, SubjectConfig] = {
    "general": SubjectConfig(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        top_k=DEFAULT_TOP_K,
        hybrid_alpha=DEFAULT_HYBRID_ALPHA
    ),
    "mathematics": SubjectConfig(
        chunk_size=200,
        chunk_overlap=50,
        top_k=2,
        hybrid_alpha=0.7
    ),
    "science": SubjectConfig(
        chunk_size=400,
        chunk_overlap=75,
        top_k=4,
        hybrid_alpha=0.6
    ),
    "history": SubjectConfig(
        chunk_size=600,
        chunk_overlap=100,
        top_k=5,
        hybrid_alpha=0.4
    ),
    "literature": SubjectConfig(
        chunk_size=500,
        chunk_overlap=100,
        top_k=4,
        hybrid_alpha=0.5
    ),
    "computer_science": SubjectConfig(
        chunk_size=350,
        chunk_overlap=75,
        top_k=4,
        hybrid_alpha=0.7
    ),
    "medicine": SubjectConfig(
        chunk_size=450,
        chunk_overlap=100,
        top_k=5,
        hybrid_alpha=0.6
    ),
    "law": SubjectConfig(
        chunk_size=550,
        chunk_overlap=100,
        top_k=5,
        hybrid_alpha=0.4
    ),
    "question_answering": SubjectConfig(
        chunk_size=256,
        chunk_overlap=50,
        top_k=5,
        hybrid_alpha=0.75
    ),
    "summarization": SubjectConfig(
        chunk_size=768,
        chunk_overlap=150,
        top_k=3,
        hybrid_alpha=0.4
    ),
    "comparison": SubjectConfig(
        chunk_size=512,
        chunk_overlap=100,
        top_k=4,
        hybrid_alpha=0.6
    ),
    "code_generation": SubjectConfig(
        chunk_size=384,
        chunk_overlap=75,
        top_k=4,
        hybrid_alpha=0.7
    ),
    "general_discussion": SubjectConfig(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        top_k=DEFAULT_TOP_K,
        hybrid_alpha=DEFAULT_HYBRID_ALPHA
    )
}

def get_subject_config(subject: str) -> SubjectConfig:
    """
    Get the configuration for a specific subject.
    Falls back to general configuration if subject not found.
    """
    return SUBJECT_CONFIGS.get(subject.lower(), SUBJECT_CONFIGS["general"]) 