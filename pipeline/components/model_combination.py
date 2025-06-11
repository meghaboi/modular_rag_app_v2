import itertools
from dataclasses import dataclass
from typing import List

from utils.enums import (
    EmbeddingModelType, RerankerModelType, LLMModelType, VectorStoreType
)
from utils.subject_configs import DEFAULT_EMBEDDING_MODEL
from pipeline.components.config import PipelineConfig

@dataclass
class ModelCombination:
    """Represents a combination of models for testing"""
    embedding_model: EmbeddingModelType
    vector_store: VectorStoreType
    reranker: RerankerModelType
    llm: LLMModelType

    def to_config(self, base_config: PipelineConfig) -> PipelineConfig:
        """Convert to PipelineConfig using base configuration"""
        return PipelineConfig(
            file_path=base_config.file_path,
            vector_store_type=self.vector_store,
            reranker_type=self.reranker,
            llm_type=self.llm,
            chunking_strategy_type=base_config.chunking_strategy_type,
            hybrid_alpha=base_config.hybrid_alpha,
            chunk_size=base_config.chunk_size,
            chunk_overlap=base_config.chunk_overlap,
            top_k=base_config.top_k,
            evaluation_mode=base_config.evaluation_mode
        )

    @classmethod
    def get_default_combinations(cls) -> List['ModelCombination']:
        """Get default model combinations for testing"""
        return [
            cls(
                embedding_model=emb,
                vector_store=vec,
                reranker=rer,
                llm=llm
            )
            for emb, vec, rer, llm in itertools.product(
                [EmbeddingModelType.MISTRAL, EmbeddingModelType.OPENAI],
                [VectorStoreType.CHROMA, VectorStoreType.HYBRID],
                [RerankerModelType.NONE, RerankerModelType.COHERE_V3],
                [LLMModelType.CLAUDE_4_SONNET, LLMModelType.MISTRAL_LARGE]
            )
        ]

    @classmethod
    def from_config(cls, config: PipelineConfig) -> 'ModelCombination':
        """Create a ModelCombination from a PipelineConfig"""
        return cls(
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            vector_store=config.vector_store_type,
            reranker=config.reranker_type,
            llm=config.llm_type
        ) 