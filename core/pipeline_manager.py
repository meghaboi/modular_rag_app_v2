import os
import logging

# import pandas as pd # Removed: No longer used in this file
import streamlit as st  # Still used by initialize_pipeline and run_pipeline_with_config
from typing import Optional, Dict, Any  # Removed List, Tuple

# import itertools # Removed: No longer used in this file

from ..utils.enums import (  # Updated path to enums
    EmbeddingModelType,
    RerankerModelType,
    LLMModelType,
    VectorStoreType,
    ChunkingStrategyType,
    # EvaluationBackendType, # Removed: Not directly used here
    # EvaluationMetricType,  # Removed: Not directly used here
)
from ..utils.evaluator import (
    evaluate_rag_response,
)  # Import the new evaluation function
from .embedding_models import EmbeddingModelFactory
from .rerankers import RerankerFactory
from .vector_stores import VectorStoreFactory
from .llm_models import LLMFactory
from .rag_pipeline import RAGPipeline
from .chunking import ChunkingStrategyFactory  # Moved to .chunking

# from ..config import check_api_keys  # Removed: Not used in this file
from ..subject_configs import (
    DEFAULT_EMBEDDING_MODEL,
)  # Assuming subject_configs is one level up


def initialize_pipeline(
    file_path: str,
    vector_store_enum: VectorStoreType,
    reranker_enum: RerankerModelType,
    llm_enum: LLMModelType,
    chunking_strategy_enum: ChunkingStrategyType,
    hybrid_alpha: float,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
) -> Optional[RAGPipeline]:
    """
    Initialize RAG pipeline with selected configuration.

    Args:
        file_path (str): Path to the document file
        vector_store_enum (VectorStoreType): Selected vector store
        reranker_enum (RerankerModelType): Selected reranker
        llm_enum (LLMModelType): Selected LLM
        chunking_strategy_enum (ChunkingStrategyType): Selected chunking strategy
        hybrid_alpha (float): Hybrid search alpha parameter
        chunk_size (int): Size of text chunks
        chunk_overlap (int): Overlap between chunks
        top_k (int): Number of top results to retrieve

    Returns:
        Optional[RAGPipeline]: Initialized pipeline or None if initialization fails
    """
    logging.info(f"Attempting to initialize RAG pipeline with config:")
    logging.info(
        f"  Embedding: {DEFAULT_EMBEDDING_MODEL.value}, Vector Store: {vector_store_enum.value}, "
        f"Reranker: {reranker_enum.value}, LLM: {llm_enum.value}"
    )

    try:
        # Initialize components with fixed embedding model
        embedding_model_instance = EmbeddingModelFactory.create_model(
            DEFAULT_EMBEDDING_MODEL
        )
        vector_store_instance = (
            VectorStoreFactory.create_store(vector_store_enum, alpha=hybrid_alpha)
            if vector_store_enum == VectorStoreType.HYBRID
            else VectorStoreFactory.create_store(vector_store_enum)
        )
        reranker_instance = (
            RerankerFactory.create_reranker(reranker_enum)
            if reranker_enum != RerankerModelType.NONE
            else None
        )
        llm_instance = LLMFactory.create_llm(llm_enum)
        chunking_strategy_instance = ChunkingStrategyFactory.get_strategy(
            chunking_strategy_enum.value
        )

        IsInEvaluationMode = False
        if st.session_state.mode == "evaluation":
            IsInEvaluationMode = True

        pipeline = RAGPipeline(
            embedding_model=embedding_model_instance,
            vector_store=vector_store_instance,
            reranker=reranker_instance,
            llm=llm_instance,
            top_k=top_k,
            chunking_strategy=chunking_strategy_instance,
            evaluation_mode=IsInEvaluationMode,
        )

        # Indexing
        pipeline.index_documents(
            file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        return pipeline

    except Exception as e:
        logging.error(f"Failed to initialize pipeline: {str(e)}")
        return None


def run_pipeline_with_config(
    file_path: str,
    user_query: str,
    ground_truth: str,
    embedding_model_enum: EmbeddingModelType,
    vector_store_enum: VectorStoreType,
    reranker_enum: RerankerModelType,
    llm_enum: LLMModelType,
    chunking_strategy_enum: ChunkingStrategyType,
    hybrid_alpha: float = 0.5,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Run a single pipeline configuration and return results.

    Args:
        file_path (str): Path to the document file
        user_query (str): User's question
        ground_truth (str): Expected answer for evaluation
        embedding_model_enum (EmbeddingModelType): Embedding model to use
        vector_store_enum (VectorStoreType): Vector store to use
        reranker_enum (RerankerModelType): Reranker to use
        llm_enum (LLMModelType): LLM to use
        chunking_strategy_enum (ChunkingStrategyType): Chunking strategy to use
        hybrid_alpha (float): Hybrid search alpha parameter
        chunk_size (int): Size of text chunks
        chunk_overlap (int): Overlap between chunks
        top_k (int): Number of top results to retrieve

    Returns:
        Dict[str, Any]: Results including response, metrics, and configuration
    """
    try:
        pipeline = initialize_pipeline(
            file_path,
            vector_store_enum,
            reranker_enum,
            llm_enum,
            chunking_strategy_enum,
            hybrid_alpha,
            chunk_size,
            chunk_overlap,
            top_k,
        )

        if not pipeline:
            return {
                "status": "error",
                "error": "Failed to initialize pipeline",
                "config": {
                    "embedding": embedding_model_enum.value,
                    "vector_store": vector_store_enum.value,
                    "reranker": reranker_enum.value,
                    "llm": llm_enum.value,
                    "chunking": chunking_strategy_enum.value,
                },
            }

        # Run query and get non-streaming response
        response_text, contexts, metrics_from_run = pipeline.run(user_query)

        # Get evaluation metrics if ground truth is provided
        evaluation_metrics = {}
        if ground_truth:
            # Call the new evaluate_rag_response function
            # metrics_from_run contains performance data like cost, tokens, time
            evaluation_metrics = evaluate_rag_response(
                query=user_query,
                response=response_text,
                contexts=contexts,
                ground_truth=ground_truth,
                cost=metrics_from_run.get("llm_cost"),
                metrics_to_include=metrics_from_run,
                # backend_type can be parameterized if needed, defaulting to RAGAS_V2
                # backend_type=EvaluationBackendType.RAGAS_V2
            )
        else:
            # If no ground truth, evaluation_metrics will primarily be performance metrics
            evaluation_metrics = metrics_from_run

        return {
            "status": "success",
            "response": response_text,
            "metrics": evaluation_metrics,  # This now contains RAGAS scores + performance metrics
            "contexts": contexts,  # Adding contexts to the output
            "config": {
                "embedding": embedding_model_enum.value,
                "vector_store": vector_store_enum.value,
                "reranker": reranker_enum.value,
                "llm": llm_enum.value,
                "chunking": chunking_strategy_enum.value,
            },
        }

    except Exception as e:
        logging.error(f"Error running pipeline: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "config": {
                "embedding": embedding_model_enum.value,
                "vector_store": vector_store_enum.value,
                "reranker": reranker_enum.value,
                "llm": llm_enum.value,
                "chunking": chunking_strategy_enum.value,
            },
        }


# run_all_permutations function has been moved to experiment_runner.py
