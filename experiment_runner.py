import pandas as pd
import streamlit as st  # For progress bar and st.write, might be refactored if this moves to a non-UI context
from typing import List, Dict, Any, Tuple
import itertools

# Assuming core.pipeline_manager and utils.enums will be in the Python path
from core.pipeline_manager import run_pipeline_with_config
from utils.enums import (
    EmbeddingModelType,
    RerankerModelType,
    LLMModelType,
    VectorStoreType,
    ChunkingStrategyType,
)


def run_all_permutations(
    file_path: str,
    user_query: str,
    ground_truth: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    hybrid_alpha: float,
    chunking_strategy_enum: ChunkingStrategyType,
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
    combinations = list(
        itertools.product(embedding_models, vector_stores, rerankers, llms)
    )

    total_combinations = len(combinations)
    # Note: st.progress and st.write are UI elements. If this function is meant to be
    # purely backend, these should be replaced with logging or other non-UI feedback.
    # For now, keeping them as per original function.
    progress_bar = st.progress(0)

    for i, (emb, vec, rer, llm) in enumerate(combinations, 1):
        # st.write is a UI element. Consider replacing with logging if this moves to backend.
        st.write(f"Testing combination {i}/{total_combinations}:")
        st.write(
            f"Embedding: {emb.value}, Vector Store: {vec.value}, "
            f"Reranker: {rer.value}, LLM: {llm.value}"
        )

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
            top_k=top_k,
        )

        results.append(result)
        progress_bar.progress(i / total_combinations)

    # Convert results to DataFrame
    df_data = []
    for result in results:
        if result["status"] == "success":
            # Ensure all expected keys from 'metrics' are handled, even if missing
            metrics = result.get("metrics", {})
            row = {
                "Embedding": result.get("config", {}).get("embedding", "N/A"),
                "Vector Store": result.get("config", {}).get("vector_store", "N/A"),
                "Reranker": result.get("config", {}).get("reranker", "N/A"),
                "LLM": result.get("config", {}).get("llm", "N/A"),
                **metrics,  # Spread the metrics dictionary
            }
            df_data.append(row)

    df = pd.DataFrame(df_data)
    return results, df
