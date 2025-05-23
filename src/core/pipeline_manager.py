import os
import logging
import pandas as pd
import streamlit as st
from typing import Optional, Dict, Any, List, Tuple
import itertools

from enums import (
    EmbeddingModelType, RerankerModelType, LLMModelType, VectorStoreType,
    ChunkingStrategyType
)
from embedding_models import EmbeddingModelFactory
from rerankers import RerankerFactory
from vector_stores import VectorStoreFactory
from llm_models import LLMFactory
from rag_pipeline import RAGPipeline, ChunkingStrategyFactory
from config import check_api_keys

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
    """
    Initialize RAG pipeline with selected configuration.
    
    Args:
        file_path (str): Path to the document file
        embedding_model_enum (EmbeddingModelType): Selected embedding model
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
    logging.info(f"  Embedding: {embedding_model_enum.value}, Vector Store: {vector_store_enum.value}, "
                f"Reranker: {reranker_enum.value}, LLM: {llm_enum.value}")
    logging.info(f"  Chunking: {chunking_strategy_enum.value}, Size: {chunk_size}, "
                f"Overlap: {chunk_overlap}, Top K: {top_k}, Hybrid Alpha: {hybrid_alpha}")

    if not file_path or not os.path.exists(file_path):
        st.error("Cannot initialize pipeline: Document file path is invalid or missing.")
        logging.error("Pipeline initialization failed: Invalid file path.")
        return None

    try:
        # Check for required API keys
        missing_keys = check_api_keys(embedding_model_enum, vector_store_enum, reranker_enum, llm_enum)
        if missing_keys:
            st.error(f"Missing required API keys: {', '.join(missing_keys)}")
            return None

        # Initialize components
        embedding_model = EmbeddingModelFactory.create_model(embedding_model_enum)
        
        if vector_store_enum == VectorStoreType.HYBRID:
            vector_store = VectorStoreFactory.create_store(vector_store_enum, alpha=hybrid_alpha)
        else:
            vector_store = VectorStoreFactory.create_store(vector_store_enum)

        reranker = None
        if reranker_enum != RerankerModelType.NONE:
            reranker = RerankerFactory.create_reranker(reranker_enum)

        llm = LLMFactory.create_llm(llm_enum)
        chunking_strategy = ChunkingStrategyFactory.get_strategy(chunking_strategy_enum.value)

        # Create and initialize pipeline
        pipeline = RAGPipeline(
            embedding_model=embedding_model,
            vector_store=vector_store,
            reranker=reranker,
            llm=llm,
            chunking_strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k
        )

        # Initialize with document
        pipeline.initialize(file_path)
        logging.info("Pipeline initialized successfully")
        return pipeline

    except Exception as e:
        st.error(f"Failed to initialize pipeline: {str(e)}")
        logging.error(f"Pipeline initialization failed: {e}", exc_info=True)
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
    top_k: int = 3
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
            file_path, embedding_model_enum, vector_store_enum, reranker_enum,
            llm_enum, chunking_strategy_enum, hybrid_alpha, chunk_size,
            chunk_overlap, top_k
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
                    "chunking": chunking_strategy_enum.value
                }
            }

        # Run query and get non-streaming response
        response = pipeline.run(user_query)
        
        # Get evaluation metrics if ground truth is provided
        metrics = {}
        if ground_truth:
            metrics = pipeline.evaluate_response(response, ground_truth)

        return {
            "status": "success",
            "response": response,
            "metrics": metrics,
            "config": {
                "embedding": embedding_model_enum.value,
                "vector_store": vector_store_enum.value,
                "reranker": reranker_enum.value,
                "llm": llm_enum.value,
                "chunking": chunking_strategy_enum.value
            }
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
                "chunking": chunking_strategy_enum.value
            }
        }

def run_all_permutations(
    file_path: str,
    user_query: str,
    ground_truth: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    hybrid_alpha: float,
    chunking_strategy_enum: ChunkingStrategyType
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """
    Run all permutations of models and return results as a dataframe.
    
    Args:
        file_path (str): Path to the document file
        user_query (str): User's question
        ground_truth (str): Expected answer for evaluation
        chunk_size (int): Size of text chunks
        chunk_overlap (int): Overlap between chunks
        top_k (int): Number of top results to retrieve
        hybrid_alpha (float): Hybrid search alpha parameter
        chunking_strategy_enum (ChunkingStrategyType): Chunking strategy to use
        
    Returns:
        Tuple[List[Dict[str, Any]], pd.DataFrame]: List of results and summary dataframe
    """
    results = []
    
    # Define model combinations to test
    embedding_models = [EmbeddingModelType.MISTRAL, EmbeddingModelType.OPENAI]
    vector_stores = [VectorStoreType.CHROMA, VectorStoreType.HYBRID]
    rerankers = [RerankerModelType.NONE, RerankerModelType.COHERE_V3]
    llms = [LLMModelType.CLAUDE_37_SONNET, LLMModelType.MISTRAL_LARGE]

    # Generate all combinations
    combinations = list(itertools.product(
        embedding_models, vector_stores, rerankers, llms
    ))

    total_combinations = len(combinations)
    progress_bar = st.progress(0)
    
    for i, (emb, vec, rer, llm) in enumerate(combinations, 1):
        st.write(f"Testing combination {i}/{total_combinations}:")
        st.write(f"Embedding: {emb.value}, Vector Store: {vec.value}, "
                f"Reranker: {rer.value}, LLM: {llm.value}")
        
        result = run_pipeline_with_config(
            file_path=file_path,
            user_query=user_query,
            ground_truth=ground_truth,
            embedding_model_enum=emb,
            vector_store_enum=vec,
            reranker_enum=rer,
            llm_enum=llm,
            chunking_strategy_enum=chunking_strategy_enum,
            hybrid_alpha=hybrid_alpha,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k
        )
        
        results.append(result)
        progress_bar.progress(i / total_combinations)

    # Convert results to DataFrame
    df_data = []
    for result in results:
        if result["status"] == "success":
            row = {
                "Embedding": result["config"]["embedding"],
                "Vector Store": result["config"]["vector_store"],
                "Reranker": result["config"]["reranker"],
                "LLM": result["config"]["llm"],
                **result.get("metrics", {})
            }
            df_data.append(row)

    df = pd.DataFrame(df_data)
    return results, df