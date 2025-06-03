from typing import Dict, Any
from dataclasses import dataclass
from enums import (
    EmbeddingModelType,
    RerankerModelType,
    LLMModelType,
    VectorStoreType,
    ChunkingStrategyType
)

DEFAULT_SYSTEM_PROMPT = """You are JEFF, that cool friend everyone wishes they had the night before exams.
You explain complex subjects in simple, relatable terms that just click when it matters most.
Unlike formal professors, you break down academic concepts with perfect clarity, memorable examples, and occasional humor.
You excel at finding the shortcuts, mnemonics, and "aha!" moments that make difficult material suddenly make sense.
Your explanations focus on what's actually important to understand and remember, cutting through the noise.
You're encouraging, patient, and have a knack for making anyone feel like they can ace their exam.
Always respond as JEFF - casual but knowledgeable, relatable but authoritative, and above all, the friend who helps everyone pass their exams."""

# Default LLM Models
DEFAULT_GPT35_MODEL = "gpt-3.5-turbo"
DEFAULT_GPT4_MODEL = "gpt-4"
DEFAULT_GPT4O_MODEL = "gpt-4o"
DEFAULT_GEMINI_FLASH_MODEL = "gemini-1.5-flash"
DEFAULT_CLAUDE_OPUS_MODEL = "claude-3-opus-20240229"
DEFAULT_CLAUDE_SONNET_MODEL = "claude-3-7-sonnet-20250219" # likely a typo, should be claude-3.5-sonnet or similar based on common Anthropic naming. Assuming it's a specific version.
DEFAULT_CLAUDE_HAIKU_MODEL = "claude-3-haiku-20240307"
DEFAULT_MISTRAL_LARGE_MODEL = "mistral-large-latest"
DEFAULT_MISTRAL_MEDIUM_MODEL = "mistral-medium-latest"
DEFAULT_MISTRAL_SMALL_MODEL = "mistral-small-latest"


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

# Use Mistral as the fixed embedding model
DEFAULT_EMBEDDING_MODEL = EmbeddingModelType.MISTRAL
DEFAULT_VECTOR_STORE = VectorStoreType.CHROMA
DEFAULT_RERANKER_MODEL = RerankerModelType.VOYAGE_2
DEFAULT_LLM_MODEL = LLMModelType.CLAUDE_37_SONNET
DEFAULT_CHUNKING_STRATEGY = ChunkingStrategyType.HIERARCHICAL

# Subject-specific configurations - only varying chunking and search parameters
SUBJECT_CONFIGS: Dict[str, SubjectConfig] = {
    "general": SubjectConfig(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        top_k=DEFAULT_TOP_K,
        hybrid_alpha=DEFAULT_HYBRID_ALPHA
    ),
    "mathematics": SubjectConfig(
        chunk_size=200,  # Smaller chunks for precise mathematical content
        chunk_overlap=50,  # Higher overlap to maintain context
        top_k=2,  # More documents for comprehensive coverage
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