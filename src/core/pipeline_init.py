"""
Pipeline initialization functionality for the ModularRAG application.
"""

import logging
from typing import Optional
from ..config.enums import (
    EmbeddingModelType,
    RerankerModelType,
    LLMModelType,
    VectorStoreType,
    ChunkingStrategyType
)

def initialize_pipeline(
    file_path: str,
    embedding_model_enum: EmbeddingModelType,
    vector_store_enum: VectorStoreType,
    reranker_enum: RerankerModelType,
    llm_enum: LLMModelType,
    chunking_strategy_enum: ChunkingStrategyType,
    hybrid_alpha: float = 0.5,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    top_k: int = 3
) -> Optional[Any]:
    """Initialize RAG pipeline with selected configuration"""
    logging.info(f"Attempting to initialize RAG pipeline with config:")
    logging.info(f"  Embedding: {embedding_model_enum.value}, Vector Store: {vector_store_enum.value}, Reranker: {reranker_enum.value}, LLM: {llm_enum.value}")
    logging.info(f"  Chunking: {chunking_strategy_enum.value}, Size: {chunk_size}, Overlap: {chunk_overlap}, Top K: {top_k}, Hybrid Alpha: {hybrid_alpha}")

    try:
        # Initialize components
        from ..models.embedding_models import EmbeddingModelFactory
        from ..models.vector_stores import VectorStoreFactory
        from ..models.rerankers import RerankerFactory
        from ..models.llm_models import LLMFactory
        from ..rag.chunking import ChunkingStrategyFactory
        from ..rag.pipeline import RAGPipeline

        embedding_model_instance = EmbeddingModelFactory.create_model(embedding_model_enum)
        vector_store_instance = VectorStoreFactory.create_store(vector_store_enum, alpha=hybrid_alpha)
        reranker_instance = RerankerFactory.create_reranker(reranker_enum) if reranker_enum != RerankerModelType.NONE else None
        llm_instance = LLMFactory.create_llm(llm_enum)
        chunking_strategy_instance = ChunkingStrategyFactory.get_strategy(chunking_strategy_enum.value)

        # Create RAG pipeline
        pipeline = RAGPipeline(
            embedding_model=embedding_model_instance,
            vector_store=vector_store_instance,
            reranker=reranker_instance,
            llm=llm_instance,
            top_k=top_k,
            chunking_strategy=chunking_strategy_instance
        )

        # Index documents
        pipeline.index_documents(file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return pipeline

    except Exception as e:
        logging.error(f"Error initializing RAG pipeline: {e}", exc_info=True)
        return None 