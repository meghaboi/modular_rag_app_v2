from enum import Enum, auto
from typing import Dict, Any, List, Optional, Type

# Enum for Embedding Models
class EmbeddingModelType(Enum):
    OPENAI = "OpenAI"
    COHERE = "Cohere"
    GEMINI = "Gemini"
    MISTRAL = "Mistral"
    VOYAGE = "Voyage"  
    
    @classmethod
    def list(cls) -> List[str]:
        """Return a list of all enum values as strings"""
        return [e.value for e in cls]
    
    @classmethod
    def from_string(cls, value: str) -> "EmbeddingModelType":
        """Get enum from string value"""
        for e in cls:
            if e.value == value:
                return e
        raise ValueError(f"Unknown embedding model: {value}")

# Enum for Reranker Models
class RerankerModelType(Enum):
    NONE = "None"
    COHERE_V2 = "Cohere-V2"
    COHERE_V3 = "Cohere-V3"
    COHERE_MULTILINGUAL = "Cohere-Multilingual"
    VOYAGE = "Voyage"
    VOYAGE_2 = "Voyage-2"
    JINA = "Jina"
    JINA_V2 = "Jina-v2"
    LLM = "LLM"
    
    @classmethod
    def list(cls) -> List[str]:
        return [e.value for e in cls]
    
    @classmethod
    def from_string(cls, value: str) -> "RerankerModelType":
        for e in cls:
            if e.value == value:
                return e
        raise ValueError(f"Unknown reranker model: {value}")

# Enum for LLM Models
class LLMModelType(Enum):
    OPENAI_GPT35 = "OpenAI GPT-3.5"
    OPENAI_GPT4 = "OpenAI GPT-4"
    GEMINI = "Gemini"
    CLAUDE_3_OPUS = "Claude-3-Opus"
    CLAUDE_37_SONNET = "Claude-3.7-Sonnet"
    MISTRAL_LARGE = "Mistral-Large"
    MISTRAL_MEDIUM = "Mistral-Medium"
    MISTRAL_SMALL = "Mistral-Small"
    
    @classmethod
    def list(cls) -> List[str]:
        return [e.value for e in cls]
    
    @classmethod
    def from_string(cls, value: str) -> "LLMModelType":
        for e in cls:
            if e.value == value:
                return e
        raise ValueError(f"Unknown LLM model: {value}")

# Enum for Vector Stores
class VectorStoreType(Enum):
    FAISS = "FAISS"
    CHROMA = "Chroma"
    MILVUS = "Milvus"
    HYBRID = "Hybrid"  # New hybrid option
    
    @classmethod
    def list(cls) -> List[str]:
        return [e.value for e in cls]
    
    @classmethod
    def from_string(cls, value: str) -> "VectorStoreType":
        for e in cls:
            if e.value == value:
                return e
        raise ValueError(f"Unknown vector store: {value}")

# Enum for Chunking Strategies
class ChunkingStrategyType(Enum):
    PARAGRAPH = "Paragraph-based"
    SLIDING_WINDOW = "Sliding Window"
    HIERARCHICAL = "Hierarchical"
    SEMANTIC = "Semantic"
    
    @classmethod
    def list(cls) -> List[str]:
        return [e.value for e in cls]
    
    @classmethod
    def from_string(cls, value: str) -> "ChunkingStrategyType":
        for e in cls:
            if e.value == value:
                return e
        raise ValueError(f"Unknown chunking strategy: {value}")

# Enum for Evaluation Methods
class EvaluationMethodType(Enum):
    BUILTIN = "Built-in Evaluator"
    LANGSMITH = "LangSmith Evaluator"
    
    @classmethod
    def list(cls) -> List[str]:
        return [e.value for e in cls]
    
    @classmethod
    def from_string(cls, value: str) -> "EvaluationMethodType":
        for e in cls:
            if e.value == value:
                return e
        raise ValueError(f"Unknown evaluation method: {value}")

class EvaluationBackendType(Enum):
    """Enum for evaluation backend types"""
    BUILTIN = "Built-in Evaluator"
    RAGAS = "RAGAS"
    LANGSMITH = "LangSmith"
    DEEP_EVAL = "DeepEval"
    RAGAS_V2 = "RAGAS_V2"
    CUSTOM = "Custom"

    @classmethod
    def list(cls):
        """Return list of enum values as strings"""
        return [e.value for e in cls]
    
    @classmethod
    def from_string(cls, value):
        """Get enum from string value"""
        for e in cls:
            if e.value == value:
                return e
        raise ValueError(f"No enum value matches: {value}")

# Enum for Evaluation Metrics
class EvaluationMetricType(Enum):
    """Enum for evaluation metric types"""
    ANSWER_RELEVANCE = "answer_relevance"
    CONTEXT_RELEVANCE = "context_relevance"
    GROUNDEDNESS = "groundedness" 
    FAITHFULNESS = "faithfulness"
    CONTEXT_PRECISION = "context_precision"  # RAGAS specific
    CONTEXT_RECALL = "context_recall"        # RAGAS specific
    ANSWER_CONSISTENCY = "answer_consistency"  # Custom metric
    CONTEXT_COVERAGE = "context_coverage"    # Custom metric
    ANSWER_CORRECTNESS = "answer_correctness"
    F1_SCORE = "f1_score"                   # Harmonic mean of context recall and relevance
    
    @classmethod
    def list(cls):
        """Return list of enum values as strings"""
        return [e.value for e in cls]
    
    @classmethod
    def from_string(cls, value):
        """Get enum from string value"""
        for e in cls:
            if e.value == value:
                return e
        raise ValueError(f"No enum value matches: {value}")
    
    @classmethod
    def get_metrics_for_backend(cls, backend_type: EvaluationBackendType):
        """Get available metrics for a specific backend"""
        if backend_type == EvaluationBackendType.BUILTIN:
            return [
                cls.ANSWER_RELEVANCE.value,
                cls.CONTEXT_RELEVANCE.value,
                cls.GROUNDEDNESS.value,
                cls.FAITHFULNESS.value
            ]
        elif backend_type == EvaluationBackendType.RAGAS:
            return [
                cls.CONTEXT_PRECISION.value,
                cls.CONTEXT_RECALL.value,
                cls.FAITHFULNESS.value,
                cls.ANSWER_CORRECTNESS.value  # Add correctness here
            ]
        elif backend_type == EvaluationBackendType.DEEP_EVAL:
            return [
                cls.ANSWER_RELEVANCE.value,
                cls.CONTEXT_RELEVANCE.value,
                cls.GROUNDEDNESS.value,
                cls.FAITHFULNESS.value,
                cls.ANSWER_CONSISTENCY.value,
                cls.CONTEXT_COVERAGE.value
            ]
        elif backend_type == EvaluationBackendType.RAGAS_V2:
            return [
                cls.FAITHFULNESS.value,
                cls.CONTEXT_PRECISION.value,
                cls.CONTEXT_RECALL.value,
                cls.ANSWER_CORRECTNESS.value,
            ]
        elif backend_type == EvaluationBackendType.CUSTOM:
            return [
                cls.CONTEXT_RECALL.value,
                cls.CONTEXT_PRECISION.value,
                cls.ANSWER_RELEVANCE.value,
                cls.FAITHFULNESS.value,
                cls.ANSWER_CORRECTNESS.value
            ]
        else:
            return []