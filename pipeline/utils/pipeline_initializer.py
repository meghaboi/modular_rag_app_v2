import logging
import time
import os
import streamlit as st
from typing import Optional
from utils.enums import (
    EmbeddingModelType,
    RerankerModelType,
    LLMModelType,
    VectorStoreType,
    ChunkingStrategyType
)
from models.embedding_models import EmbeddingModelFactory
from models.vector_stores import VectorStoreFactory
from models.rerankers import RerankerFactory
from models.llm_models import LLMFactory
from models.chunking_strategies import ChunkingStrategyFactory
from pipeline.rag_pipeline import RAGPipeline

class PipelineInitializer:
    @staticmethod
    def initialize_pipeline(
        file_path: str,
        embedding_model_enum: EmbeddingModelType,
        vector_store_enum: VectorStoreType,
        reranker_enum: RerankerModelType,
        llm_enum: LLMModelType,
        chunking_strategy_enum: ChunkingStrategyType,
        hybrid_alpha: float,
        chunk_size: int,
        chunk_overlap: int,
        top_k: int
    ) -> Optional[RAGPipeline]:
        """Initialize RAG pipeline with selected configuration"""
        logging.info(f"Attempting to initialize RAG pipeline with config:")
        logging.info(f"  Embedding: {embedding_model_enum.value}, Vector Store: {vector_store_enum.value}, Reranker: {reranker_enum.value}, LLM: {llm_enum.value}")
        logging.info(f"  Chunking: {chunking_strategy_enum.value}, Size: {chunk_size}, Overlap: {chunk_overlap}, Top K: {top_k}, Hybrid Alpha: {hybrid_alpha}")

        if not file_path or not os.path.exists(file_path):
            logging.error("Pipeline initialization failed: Invalid file path.")
            return None

        try:
            # Initialize components
            embedding_model_instance = EmbeddingModelFactory.create_model(embedding_model_enum)

            if vector_store_enum == VectorStoreType.HYBRID:
                vector_store_instance = VectorStoreFactory.create_store(vector_store_enum, alpha=hybrid_alpha)
            else:
                vector_store_instance = VectorStoreFactory.create_store(vector_store_enum)

            reranker_instance = None
            if reranker_enum != RerankerModelType.NONE:
                if reranker_enum == RerankerModelType.LLM:
                    llm_instance = LLMFactory.create_llm(llm_enum)
                    reranker_instance = RerankerFactory.create_reranker(reranker_enum, llm_client=llm_instance)
                else:
                    reranker_instance = RerankerFactory.create_reranker(reranker_enum)

            if 'llm_instance' not in locals() or llm_instance is None:
                llm_instance = LLMFactory.create_llm(llm_enum)
            
            chunking_strategy_instance = ChunkingStrategyFactory.get_strategy(chunking_strategy_enum.value)

            is_in_evaluation_mode = st.session_state.mode == "evaluation"

            # Create RAG pipeline
            pipeline = RAGPipeline(
                embedding_model=embedding_model_instance,
                vector_store=vector_store_instance,
                reranker=reranker_instance,
                llm=llm_instance,
                top_k=top_k,
                chunking_strategy=chunking_strategy_instance,
                evaluation_mode=is_in_evaluation_mode 
            )

            # Index documents
            logging.info(f"Indexing documents from: {file_path}")
            index_start_time = time.time()
            try:
                pipeline.index_documents(file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            except Exception as index_e:
                logging.error(f"Error during document indexing: {index_e}", exc_info=True)
                return None
            index_end_time = time.time()
            logging.info(f"Document indexing completed in {index_end_time - index_start_time:.2f} seconds.")

            return pipeline

        except Exception as e:
            logging.error(f"Error initializing RAG pipeline: {e}", exc_info=True)
            return None