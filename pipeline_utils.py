import logging
import time
import itertools
import pandas as pd
import os
import streamlit as st
from typing import Dict, Any, List, Tuple
from enums import (
    EmbeddingModelType,
    RerankerModelType,
    LLMModelType,
    VectorStoreType,
    ChunkingStrategyType
)
from utils import check_api_keys
from embedding_models import EmbeddingModelFactory
from vector_stores import VectorStoreFactory
from rerankers import RerankerFactory
from llm_models import LLMFactory
# Assuming ChunkingStrategyFactory might be in rag_pipeline or needs to be created/found.
# For now, I will comment it out if it's not directly available.
# from chunking_factory import ChunkingStrategyFactory # Placeholder
from rag_pipeline import RAGPipeline, ChunkingStrategyFactory # If ChunkingStrategyFactory is in rag_pipeline.py

def initialize_pipeline(file_path, embedding_model_enum, vector_store_enum, reranker_enum, llm_enum,
                        chunking_strategy_enum, hybrid_alpha, chunk_size, chunk_overlap, top_k):
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
            reranker_instance = RerankerFactory.create_reranker(reranker_enum)

        llm_instance = LLMFactory.create_llm(llm_enum)
        chunking_strategy_instance = ChunkingStrategyFactory.get_strategy(chunking_strategy_enum.value)

        IsInEvaluationMode = False
        if st.session_state.mode == "evaluation":
            IsInEvaluationMode = True

        # Create RAG pipeline
        pipeline = RAGPipeline(
            embedding_model=embedding_model_instance,
            vector_store=vector_store_instance,
            reranker=reranker_instance,
            llm=llm_instance,
            top_k=top_k,
            chunking_strategy=chunking_strategy_instance,
            evaluation_mode=IsInEvaluationMode 
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
    """Run a single pipeline configuration and return results"""
    config_str = f"{embedding_model_enum.value}, {vector_store_enum.value}, {reranker_enum.value}, {llm_enum.value}, {chunking_strategy_enum.value}"
    logging.info(f"Running pipeline with config: {config_str}")
    start_run_time = time.time()
    try:
        # Initialize components
        embedding_model_instance = EmbeddingModelFactory.create_model(embedding_model_enum)
        vector_store_instance = VectorStoreFactory.create_store(vector_store_enum, alpha=hybrid_alpha) if vector_store_enum == VectorStoreType.HYBRID else VectorStoreFactory.create_store(vector_store_enum)
        reranker_instance = RerankerFactory.create_reranker(reranker_enum) if reranker_enum != RerankerModelType.NONE else None
        llm_instance = LLMFactory.create_llm(llm_enum)
        chunking_strategy_instance = ChunkingStrategyFactory.get_strategy(chunking_strategy_enum.value)

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
            evaluation_mode=IsInEvaluationMode 
        )

        # Indexing (re-index per config for isolation in eval)
        pipeline.index_documents(file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Process query
        start_query_time = time.time()
        response, contexts, metrics = pipeline.process_query(user_query)
        query_elapsed_time = time.time() - start_query_time
        logging.info(f"Query processed in {query_elapsed_time:.2f}s. Response length: {len(response)}")

        # Run evaluation
        evaluation_results = {}
        avg_score = 0
        if ground_truth:
            try:
                evaluator = EvaluatorFactory.create_evaluator(
                    EvaluationBackendType.CUSTOM,
                    EvaluationMetricType.get_metrics_for_backend(EvaluationBackendType.CUSTOM)
                )
                evaluation_results = evaluator.evaluate(
                    query=user_query, response=response, contexts=contexts, ground_truth=ground_truth
                )
                if evaluation_results and isinstance(evaluation_results, dict):
                    valid_scores = [v for v in evaluation_results.values() if isinstance(v, (int, float))]
                    if valid_scores:
                        avg_score = sum(valid_scores) / len(valid_scores)
                logging.info(f"Evaluation scores: {evaluation_results}")
            except Exception as eval_e:
                logging.error(f"Error during evaluation for config {config_str}: {eval_e}", exc_info=True)
                evaluation_results = {"error": str(eval_e)}
        else:
             logging.warning("No ground truth provided, skipping RAGAS evaluation.")

        total_elapsed_time = time.time() - start_run_time
        logging.info(f"Total run time for config {config_str}: {total_elapsed_time:.2f}s")

        # Combine all results
        results = {
            "config": config_str,
            "response": response,
            "contexts": contexts,
            "evaluation_scores": evaluation_results,
            "avg_score": avg_score,
            "metrics": metrics,
            "total_time": total_elapsed_time
        }

        return results

    except Exception as e:
        logging.error(f"Error running pipeline with config {config_str}: {e}", exc_info=True)
        return {
            "config": config_str,
            "error": str(e),
            "total_time": time.time() - start_run_time
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
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Run all permutations of models and return results as a dataframe"""
    logging.info("Starting 'Run All Permutations'")
    embedding_models = [
        EmbeddingModelType.VOYAGE, EmbeddingModelType.GEMINI, EmbeddingModelType.MISTRAL
    ]
    vector_stores = [
        VectorStoreType.FAISS, VectorStoreType.CHROMA
    ]
    rerankers = list(RerankerModelType)
    llm_models = [
        LLMModelType.CLAUDE_37_SONNET, LLMModelType.GEMINI
    ]

    all_permutations = list(itertools.product(embedding_models, vector_stores, rerankers, llm_models))
    num_permutations = len(all_permutations)
    logging.info(f"Total permutations to run: {num_permutations}")

    progress_bar = st.progress(0, text="Starting permutations...")
    results = []
    start_permutations_time = time.time()

    for i, (embedding_model, vector_store, reranker, llm_model) in enumerate(all_permutations):
        current_config_str = f"{embedding_model.value}, {vector_store.value}, {reranker.value}, {llm_model.value}"
        logging.info(f"Running permutation {i+1}/{num_permutations}: {current_config_str}")
        progress_text = f"Running permutation {i+1}/{num_permutations}: {current_config_str}"
        try:
            progress_bar.progress((i + 1) / num_permutations, text=progress_text)
        except Exception as pb_e:
            logging.warning(f"Could not update progress bar: {pb_e}")

        missing_keys = check_api_keys(embedding_model, vector_store, reranker, llm_model)
        if missing_keys:
            st.warning(f"Skipping permutation {current_config_str} due to missing keys: {', '.join(missing_keys)}")
            result = {
                "embedding_model": embedding_model.value, "vector_store": vector_store.value,
                "reranker": reranker.value, "llm_model": llm_model.value,
                "chunking_strategy": chunking_strategy_enum.value, "response": "SKIPPED - Missing API Keys",
                "evaluation_scores": {}, "avg_score": 0, "elapsed_time": 0, "contexts": []
            }
        else:
             result = run_pipeline_with_config(
                 file_path=file_path, user_query=user_query, ground_truth=ground_truth,
                 embedding_model_enum=embedding_model, vector_store_enum=vector_store,
                 reranker_enum=reranker, llm_enum=llm_model,
                 chunking_strategy_enum=chunking_strategy_enum, hybrid_alpha=hybrid_alpha,
                 chunk_size=chunk_size, chunk_overlap=chunk_overlap, top_k=top_k
             )

        if "evaluation_scores" in result and isinstance(result["evaluation_scores"], dict):
             for metric, score in result["evaluation_scores"].items():
                 if isinstance(score, (int, float)):
                     result[f"metric_{metric}"] = score

        results.append(result)

    end_permutations_time = time.time()
    total_time = end_permutations_time - start_permutations_time
    logging.info(f"All {num_permutations} permutations completed in {total_time:.2f} seconds.")
    try:
        progress_bar.progress(1.0, text="Permutations complete!")
        time.sleep(1)
        progress_bar.empty()
    except Exception as pb_e:
        logging.warning(f"Could not update/empty progress bar: {pb_e}")

    results_df = pd.DataFrame(results)

    base_columns = ["embedding_model", "vector_store", "reranker", "llm_model", "chunking_strategy", "avg_score", "elapsed_time"]
    metric_columns = sorted([col for col in results_df.columns if col.startswith("metric_")])
    csv_columns = base_columns + metric_columns + ["response"]

    for col in csv_columns:
        if col not in results_df.columns:
            results_df[col] = pd.NA

    results_df['avg_score'] = pd.to_numeric(results_df['avg_score'], errors='coerce')
    results_df['elapsed_time'] = pd.to_numeric(results_df['elapsed_time'], errors='coerce')
    for col in metric_columns:
        results_df[col] = pd.to_numeric(results_df[col], errors='coerce')

    display_df = results_df[csv_columns].copy()
    return display_df, results 