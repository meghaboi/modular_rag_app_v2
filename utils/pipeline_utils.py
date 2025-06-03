import logging
import time
import itertools
import pandas as pd
import os
import csv
import streamlit as st
from typing import Dict, Any, List, Tuple
from .enums import (  # Relative import
    EmbeddingModelType,
    RerankerModelType,
    LLMModelType,
    VectorStoreType,
    ChunkingStrategyType,
    EvaluationBackendType,
    EvaluationMetricType,
)
from .utils import check_api_keys  # Relative import
from ..core.embedding_models import EmbeddingModelFactory  # Core import
from ..core.vector_stores import VectorStoreFactory  # Core import
from ..core.rerankers import RerankerFactory  # Core import
from ..core.llm_models import LLMFactory  # Core import
from ..core.evaluator import (
    EvaluatorFactory,
)  # Core import (assuming evaluator.py is in core)

# Assuming ChunkingStrategyFactory might be in rag_pipeline or needs to be created/found.
# For now, I will comment it out if it's not directly available.
# from chunking_factory import ChunkingStrategyFactory # Placeholder
from ..core.rag_pipeline import RAGPipeline  # Core import
from ..core.chunking import (
    ChunkingStrategyFactory,
)  # Core import (ChunkingStrategyFactory is in chunking.py)


def initialize_pipeline(
    file_path,
    embedding_model_enum,
    vector_store_enum,
    reranker_enum,
    llm_enum,
    chunking_strategy_enum,
    hybrid_alpha,
    chunk_size,
    chunk_overlap,
    top_k,
):
    """Initialize RAG pipeline with selected configuration"""
    logging.info(f"Attempting to initialize RAG pipeline with config:")
    logging.info(
        f"  Embedding: {embedding_model_enum.value}, Vector Store: {vector_store_enum.value}, Reranker: {reranker_enum.value}, LLM: {llm_enum.value}"
    )
    logging.info(
        f"  Chunking: {chunking_strategy_enum.value}, Size: {chunk_size}, Overlap: {chunk_overlap}, Top K: {top_k}, Hybrid Alpha: {hybrid_alpha}"
    )

    if not file_path or not os.path.exists(file_path):
        logging.error("Pipeline initialization failed: Invalid file path.")
        return None

    try:
        # Initialize components
        embedding_model_instance = EmbeddingModelFactory.create_model(
            embedding_model_enum
        )

        if vector_store_enum == VectorStoreType.HYBRID:
            vector_store_instance = VectorStoreFactory.create_store(
                vector_store_enum, alpha=hybrid_alpha
            )
        else:
            vector_store_instance = VectorStoreFactory.create_store(vector_store_enum)

        reranker_instance = None
        if reranker_enum != RerankerModelType.NONE:
            if reranker_enum == RerankerModelType.LLM:
                # Ensure llm_instance is created before reranker if LLM reranker is used
                llm_instance = LLMFactory.create_llm(
                    llm_enum
                )  # This line might be redundant if llm_instance is already created
                reranker_instance = RerankerFactory.create_reranker(
                    reranker_enum, llm_client=llm_instance
                )
            else:
                reranker_instance = RerankerFactory.create_reranker(reranker_enum)

        # llm_instance might be created here if not LLM reranker, or it might already exist.
        # Ensure llm_instance is defined before being used by RAGPipeline
        if "llm_instance" not in locals() or llm_instance is None:
            llm_instance = LLMFactory.create_llm(llm_enum)

        chunking_strategy_instance = ChunkingStrategyFactory.get_strategy(
            chunking_strategy_enum.value
        )

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
            evaluation_mode=IsInEvaluationMode,
        )

        # Index documents
        logging.info(f"Indexing documents from: {file_path}")
        index_start_time = time.time()
        try:
            pipeline.index_documents(
                file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        except Exception as index_e:
            logging.error(f"Error during document indexing: {index_e}", exc_info=True)
            return None
        index_end_time = time.time()
        logging.info(
            f"Document indexing completed in {index_end_time - index_start_time:.2f} seconds."
        )

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
    top_k: int = 3,
) -> Dict[str, Any]:
    """Run a single pipeline configuration and return results"""
    config_str = f"{embedding_model_enum.value}, {vector_store_enum.value}, {reranker_enum.value}, {llm_enum.value}, {chunking_strategy_enum.value}"
    logging.info(f"Running pipeline with config: {config_str}")
    start_run_time = time.time()
    try:
        # Initialize components
        embedding_model_instance = EmbeddingModelFactory.create_model(
            embedding_model_enum
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

        # Indexing (re-index per config for isolation in eval)
        pipeline.index_documents(
            file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        # Process query
        start_query_time = time.time()
        response, contexts, metrics = pipeline.process_query(user_query)
        query_elapsed_time = time.time() - start_query_time
        logging.info(
            f"Query processed in {query_elapsed_time:.2f}s. Response length: {len(response)}"
        )

        # Initialize evaluation results
        custom_evaluation_results = {}
        ragas_evaluation_results = {}
        avg_custom_score = 0

        if ground_truth:
            # Custom Evaluation - SKIPPED
            # try:
            #     custom_evaluator = EvaluatorFactory.create_evaluator(
            #         EvaluationBackendType.CUSTOM,
            #         EvaluationMetricType.get_metrics_for_backend(EvaluationBackendType.CUSTOM)
            #     )
            #     custom_evaluation_results = custom_evaluator.evaluate(
            #         query=user_query, response=response, contexts=contexts, ground_truth=ground_truth
            #     )
            #     if custom_evaluation_results and isinstance(custom_evaluation_results, dict):
            #         valid_scores = [v for v in custom_evaluation_results.values() if isinstance(v, (int, float))]
            #         if valid_scores:
            #             avg_custom_score = sum(valid_scores) / len(valid_scores)
            #     logging.info(f"Custom evaluation scores: {custom_evaluation_results}")
            # except Exception as eval_e:
            #     logging.error(f"Error during Custom evaluation for config {config_str}: {eval_e}", exc_info=True)
            #     custom_evaluation_results = {"error": str(eval_e)}

            # RAGAS Evaluation
            try:
                ragas_evaluator = EvaluatorFactory.create_evaluator(
                    EvaluationBackendType.RAGAS_V2,
                    EvaluationMetricType.get_metrics_for_backend(
                        EvaluationBackendType.RAGAS_V2
                    ),
                )
                ragas_evaluation_results = ragas_evaluator.evaluate(
                    query=user_query,
                    response=response,
                    contexts=contexts,
                    ground_truth=ground_truth,
                )
                logging.info(f"RAGAS evaluation scores: {ragas_evaluation_results}")
            except Exception as ragas_eval_e:
                logging.error(
                    f"Error during RAGAS evaluation for config {config_str}: {ragas_eval_e}",
                    exc_info=True,
                )
                ragas_evaluation_results = {"error": str(ragas_eval_e)}
        else:
            logging.warning(
                "No ground truth provided, skipping Custom and RAGAS evaluation."
            )

        total_elapsed_time = time.time() - start_run_time
        logging.info(
            f"Total run time for config {config_str}: {total_elapsed_time:.2f}s"
        )

        # Combine all results
        flat_custom_scores = {
            f"custom_{k}": v for k, v in custom_evaluation_results.items()
        }
        flat_ragas_scores = {
            f"ragas_{k}": v for k, v in ragas_evaluation_results.items()
        }

        results = {
            "embedding_model": embedding_model_enum.value,
            "vector_store": vector_store_enum.value,
            "reranker": reranker_enum.value,
            "llm_model": llm_enum.value,
            "chunking_strategy": chunking_strategy_enum.value,
            "response": response,
            "contexts": contexts,
            "custom_evaluation_scores": custom_evaluation_results,
            "ragas_evaluation_scores": ragas_evaluation_results,
            "avg_custom_score": avg_custom_score,
            "metrics": metrics,
            "elapsed_time": total_elapsed_time,
        }
        results.update(flat_custom_scores)
        results.update(flat_ragas_scores)

        return results

    except Exception as e:
        logging.error(
            f"Error running pipeline with config {config_str}: {e}", exc_info=True
        )
        return {
            "embedding_model": embedding_model_enum.value,
            "vector_store": vector_store_enum.value,
            "reranker": reranker_enum.value,
            "llm_model": llm_enum.value,
            "chunking_strategy": chunking_strategy_enum.value,
            "response": "ERROR",
            "contexts": [],
            "custom_evaluation_scores": {"error": str(e)},
            "ragas_evaluation_scores": {"error": "Pipeline Error"},
            "avg_custom_score": 0,
            "metrics": {},
            "elapsed_time": time.time() - start_run_time,
            "error": str(e),
        }


def run_all_permutations(
    file_path: str,
    user_query: str,
    ground_truth: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    hybrid_alpha: float,
    chunking_strategy_enum: ChunkingStrategyType,
    output_csv_file: str = "permutation_results.csv",
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Run all permutations of models and return results as a dataframe, writing to CSV incrementally"""
    logging.info(
        f"Starting 'Run All Permutations'. Results will be saved to {output_csv_file}"
    )
    embedding_models = [
        EmbeddingModelType.VOYAGE,
        EmbeddingModelType.GEMINI,
        EmbeddingModelType.MISTRAL,
    ]
    vector_stores = [VectorStoreType.FAISS, VectorStoreType.CHROMA]
    rerankers = [r for r in RerankerModelType if r != RerankerModelType.NONE] + [
        RerankerModelType.NONE
    ]
    llm_models = [LLMModelType.CLAUDE_37_SONNET, LLMModelType.GEMINI]

    all_permutations = list(
        itertools.product(embedding_models, vector_stores, rerankers, llm_models)
    )
    num_permutations = len(all_permutations)
    logging.info(f"Total permutations to run: {num_permutations}")

    progress_bar = st.progress(0, text="Starting permutations...")
    all_results_list = []
    start_permutations_time = time.time()

    csv_header_written = False

    with open(output_csv_file, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = None

        for i, (embedding_model, vector_store, reranker, llm_model) in enumerate(
            all_permutations
        ):
            current_config_str = f"Emb: {embedding_model.value}, VS: {vector_store.value}, Rerank: {reranker.value}, LLM: {llm_model.value}, ChunkS: {chunking_strategy_enum.value}"
            logging.info(
                f"Running permutation {i+1}/{num_permutations}: {current_config_str}"
            )
            progress_text = f"Permutation {i+1}/{num_permutations}: {embedding_model.value[:5]}..-{llm_model.value[:5]}.."
            try:
                progress_bar.progress((i + 1) / num_permutations, text=progress_text)
            except Exception as pb_e:
                logging.warning(f"Could not update progress bar: {pb_e}")

            missing_keys = check_api_keys(
                embedding_model, vector_store, reranker, llm_model
            )

            current_result = {}
            if missing_keys:
                st.warning(
                    f"Skipping permutation {current_config_str} due to missing keys: {', '.join(missing_keys)}"
                )
                current_result = {
                    "embedding_model": embedding_model.value,
                    "vector_store": vector_store.value,
                    "reranker": reranker.value,
                    "llm_model": llm_model.value,
                    "chunking_strategy": chunking_strategy_enum.value,
                    "response": "SKIPPED - Missing API Keys",
                    "avg_custom_score": 0,
                    "elapsed_time": 0,
                    "contexts": [],
                    **{
                        f"custom_{m.value.lower().replace(' ', '_')}": "N/A"
                        for m in EvaluationMetricType.get_metrics_for_backend(
                            EvaluationBackendType.CUSTOM
                        )
                    },
                    **{
                        f"ragas_{m.value.lower().replace(' ', '_')}": "N/A"
                        for m in EvaluationMetricType.get_metrics_for_backend(
                            EvaluationBackendType.RAGAS_V2
                        )
                    },
                }
            else:
                current_result = run_pipeline_with_config(
                    file_path=file_path,
                    user_query=user_query,
                    ground_truth=ground_truth,
                    embedding_model_enum=embedding_model,
                    vector_store_enum=vector_store,
                    reranker_enum=reranker,
                    llm_enum=llm_model,
                    chunking_strategy_enum=chunking_strategy_enum,
                    hybrid_alpha=hybrid_alpha,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    top_k=top_k,
                )

            current_result["embedding_model"] = current_result.get(
                "embedding_model", embedding_model.value
            )
            current_result["vector_store"] = current_result.get(
                "vector_store", vector_store.value
            )
            current_result["reranker"] = current_result.get("reranker", reranker.value)
            current_result["llm_model"] = current_result.get(
                "llm_model", llm_model.value
            )
            current_result["chunking_strategy"] = current_result.get(
                "chunking_strategy", chunking_strategy_enum.value
            )
            current_result["elapsed_time"] = current_result.get("elapsed_time", 0)
            current_result["response"] = current_result.get("response", "ERROR")

            flat_result_for_csv = {
                k: v
                for k, v in current_result.items()
                if not isinstance(v, (list, dict)) or k in ["response"]
            }

            custom_scores = current_result.get("custom_evaluation_scores", {})
            if isinstance(custom_scores, dict):
                for k, v in custom_scores.items():
                    flat_result_for_csv[f"custom_{k}"] = v

            ragas_scores = current_result.get("ragas_evaluation_scores", {})
            if isinstance(ragas_scores, dict):
                for k, v in ragas_scores.items():
                    flat_result_for_csv[f"ragas_{k}"] = v

            llm_metrics = current_result.get("metrics", {})
            if isinstance(llm_metrics, dict):
                for k, v in llm_metrics.items():
                    flat_result_for_csv[f"llm_{k}"] = v

            if not csv_header_written:
                header = list(flat_result_for_csv.keys())
                csv_writer = csv.DictWriter(csvfile, fieldnames=header)
                csv_writer.writeheader()
                csv_header_written = True

            row_to_write = {
                field: flat_result_for_csv.get(field, "N/A")
                for field in csv_writer.fieldnames
            }
            csv_writer.writerow(row_to_write)
            csvfile.flush()

            all_results_list.append(current_result)

    end_permutations_time = time.time()
    total_time = end_permutations_time - start_permutations_time
    logging.info(
        f"All {num_permutations} permutations completed in {total_time:.2f} seconds."
    )
    try:
        progress_bar.progress(1.0, text="Permutations complete! Results saved.")
        time.sleep(2)
        progress_bar.empty()
    except Exception as pb_e:
        logging.warning(f"Could not update/empty progress bar: {pb_e}")

    results_df = pd.DataFrame(all_results_list)

    base_df_columns = [
        "embedding_model",
        "vector_store",
        "reranker",
        "llm_model",
        "chunking_strategy",
        "avg_custom_score",
        "elapsed_time",
    ]

    custom_metric_cols_df = sorted(
        [
            col
            for col in results_df.columns
            if col.startswith("custom_") and col != "custom_evaluation_scores"
        ]
    )
    ragas_metric_cols_df = sorted(
        [
            col
            for col in results_df.columns
            if col.startswith("ragas_") and col != "ragas_evaluation_scores"
        ]
    )
    llm_metric_cols_df = sorted(
        [col for col in results_df.columns if col.startswith("llm_")]
    )

    display_columns = (
        base_df_columns
        + custom_metric_cols_df
        + ragas_metric_cols_df
        + llm_metric_cols_df
        + ["response"]
    )

    for col in display_columns:
        if col not in results_df.columns:
            results_df[col] = pd.NA

    cols_to_numeric = (
        ["avg_custom_score", "elapsed_time"]
        + custom_metric_cols_df
        + ragas_metric_cols_df
        + llm_metric_cols_df
    )
    for col in cols_to_numeric:
        if col in results_df.columns:
            results_df[col] = pd.to_numeric(results_df[col], errors="coerce")

    display_df = results_df[display_columns].copy()

    logging.info(f"Returning DataFrame with columns: {display_df.columns.tolist()}")
    return display_df, all_results_list
