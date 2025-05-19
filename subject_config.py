from typing import Dict, Any, Optional
from enums import (
    EmbeddingModelType, RerankerModelType, LLMModelType, VectorStoreType,
    ChunkingStrategyType
)

class SubjectConfig:
    """Manages subject-specific configurations for the RAG pipeline"""
    
    # Default configurations for different subjects
    SUBJECT_CONFIGS = {
        "math": {
            "embedding_model": EmbeddingModelType.MISTRAL,
            "vector_store": VectorStoreType.CHROMA,
            "reranker": RerankerModelType.COHERE_V3,
            "llm_model": LLMModelType.CLAUDE_37_SONNET,
            "chunking_strategy": ChunkingStrategyType.HIERARCHICAL,
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "top_k": 4,
            "hybrid_alpha": 0.5
        },
        "science": {
            "embedding_model": EmbeddingModelType.MISTRAL,
            "vector_store": VectorStoreType.CHROMA,
            "reranker": RerankerModelType.COHERE_V3,
            "llm_model": LLMModelType.CLAUDE_37_SONNET,
            "chunking_strategy": ChunkingStrategyType.HIERARCHICAL,
            "chunk_size": 1500,
            "chunk_overlap": 300,
            "top_k": 5,
            "hybrid_alpha": 0.6
        },
        "history": {
            "embedding_model": EmbeddingModelType.MISTRAL,
            "vector_store": VectorStoreType.CHROMA,
            "reranker": RerankerModelType.COHERE_V3,
            "llm_model": LLMModelType.CLAUDE_37_SONNET,
            "chunking_strategy": ChunkingStrategyType.HIERARCHICAL,
            "chunk_size": 2000,
            "chunk_overlap": 400,
            "top_k": 6,
            "hybrid_alpha": 0.7
        },
        "literature": {
            "embedding_model": EmbeddingModelType.MISTRAL,
            "vector_store": VectorStoreType.CHROMA,
            "reranker": RerankerModelType.COHERE_V3,
            "llm_model": LLMModelType.CLAUDE_37_SONNET,
            "chunking_strategy": ChunkingStrategyType.HIERARCHICAL,
            "chunk_size": 2500,
            "chunk_overlap": 500,
            "top_k": 7,
            "hybrid_alpha": 0.8
        }
    }
    
    @classmethod
    def get_subject_config(cls, subject: str) -> Dict[str, Any]:
        """
        Get the configuration for a specific subject.
        
        Args:
            subject: The subject name (case-insensitive)
            
        Returns:
            Dict containing the configuration for the subject
        """
        subject = subject.lower()
        if subject in cls.SUBJECT_CONFIGS:
            return cls.SUBJECT_CONFIGS[subject]
        else:
            # Return default configuration for unknown subjects
            return cls.SUBJECT_CONFIGS["math"]
    
    @classmethod
    def get_available_subjects(cls) -> list:
        """Get list of available subjects"""
        return list(cls.SUBJECT_CONFIGS.keys())
    
    @classmethod
    def add_custom_subject(cls, subject: str, config: Dict[str, Any]) -> None:
        """
        Add a custom subject configuration.
        
        Args:
            subject: The subject name
            config: The configuration dictionary
        """
        cls.SUBJECT_CONFIGS[subject.lower()] = config 