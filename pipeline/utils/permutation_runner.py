import logging
import time
import itertools
import pandas as pd
import csv
import streamlit as st
from typing import Dict, Any, List, Tuple
from utils.enums import (
    EmbeddingModelType,
    RerankerModelType,
    LLMModelType,
    VectorStoreType,
    ChunkingStrategyType,
    EvaluationBackendType,
    EvaluationMetricType
)
from utils.api_management.api_utils import check_api_keys
from .pipeline_runner import PipelineRunner

class PermutationRunner:
    @staticmethod
    def run_all_permutations(
        file_path: str,
        user_query: str,
        ground_truth: str,
        chunk_size: int,
        chunk_overlap: int,
        top_k: int,
        hybrid_alpha: float,
        chunking_strategy_enum: ChunkingStrategyType,
        output_csv_file: str = "permutation_results.csv"
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Run all permutations of models and return results as a dataframe, writing to CSV incrementally"""
        logging.info(f"Starting 'Run All Permutations'. Results will be saved to {output_csv_file}")
        embedding_models = [
            EmbeddingModelType.VOYAGE, EmbeddingModelType.GEMINI, EmbeddingModelType.MISTRAL
        ]
        vector_stores = [
            VectorStoreType.FAISS, VectorStoreType.CHROMA
        ]
        rerankers = [r for r in RerankerModelType if r != RerankerModelType.NONE] + [RerankerModelType.NONE]
        llm_models = [
            LLMModelType.CLAUDE_37_SONNET, LLMModelType.GEMINI
        ]

        all_permutations = list(itertools.product(embedding_models, vector_stores, rerankers, llm_models))
        num_permutations = len(all_permutations)
        logging.info(f"Total permutations to run: {num_permutations}")

        progress_bar = st.progress(0, text="Starting permutations...")
        all_results_list = []
        start_permutations_time = time.time()
        
        csv_header_written = False

        with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = None

            for i, (embedding_model, vector_store, reranker, llm_model) in enumerate(all_permutations):
                current_config_str = f"Emb: {embedding_model.value}, VS: {vector_store.value}, Rerank: {reranker.value}, LLM: {llm_model.value}, ChunkS: {chunking_strategy_enum.value}"
                logging.info(f"Running permutation {i+1}/{num_permutations}: {current_config_str}")
                progress_text = f"Permutation {i+1}/{num_permutations}: {embedding_model.value[:5]}..-{llm_model.value[:5]}.."
                try:
                    progress_bar.progress((i + 1) / num_permutations, text=progress_text)
                except Exception as pb_e:
                    logging.warning(f"Could not update progress bar: {pb_e}")

                missing_keys = check_api_keys(embedding_model, vector_store, reranker, llm_model)
                
                current_result = {}
                if missing_keys:
                    st.warning(f"Skipping permutation {current_config_str} due to missing keys: {', '.join(missing_keys)}")
                    current_result = {
                        "embedding_model": embedding_model.value, "vector_store": vector_store.value,
                        "reranker": reranker.value, "llm_model": llm_model.value,
                        "chunking_strategy": chunking_strategy_enum.value, "response": "SKIPPED - Missing API Keys",
                        "avg_custom_score": 0, "elapsed_time": 0, "contexts": [],
                        **{f"custom_{m.value.lower().replace(' ', '_')}": "N/A" for m in EvaluationMetricType.get_metrics_for_backend(EvaluationBackendType.CUSTOM)},
                        **{f"ragas_{m.value.lower().replace(' ', '_')}": "N/A" for m in EvaluationMetricType.get_metrics_for_backend(EvaluationBackendType.RAGAS_V2)}
                    }
                else:
                    current_result = PipelineRunner.run_pipeline_with_config(
                        file_path=file_path, user_query=user_query, ground_truth=ground_truth,
                        embedding_model_enum=embedding_model, vector_store_enum=vector_store,
                        reranker_enum=reranker, llm_enum=llm_model,
                        chunking_strategy_enum=chunking_strategy_enum, hybrid_alpha=hybrid_alpha,
                        chunk_size=chunk_size, chunk_overlap=chunk_overlap, top_k=top_k
                    )
                
                current_result["embedding_model"] = current_result.get("embedding_model", embedding_model.value)
                current_result["vector_store"] = current_result.get("vector_store", vector_store.value)
                current_result["reranker"] = current_result.get("reranker", reranker.value)
                current_result["llm_model"] = current_result.get("llm_model", llm_model.value)
                current_result["chunking_strategy"] = current_result.get("chunking_strategy", chunking_strategy_enum.value)
                current_result["elapsed_time"] = current_result.get("elapsed_time", 0)
                current_result["response"] = current_result.get("response", "ERROR")

                flat_result_for_csv = {k: v for k, v in current_result.items() if not isinstance(v, (list, dict)) or k in ["response"]}
                
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
                    for k,v in llm_metrics.items():
                        flat_result_for_csv[f"llm_{k}"] = v

                if not csv_header_written:
                    header = list(flat_result_for_csv.keys())
                    csv_writer = csv.DictWriter(csvfile, fieldnames=header)
                    csv_writer.writeheader()
                    csv_header_written = True
                
                row_to_write = {field: flat_result_for_csv.get(field, "N/A") for field in csv_writer.fieldnames}
                csv_writer.writerow(row_to_write)
                csvfile.flush()

                all_results_list.append(current_result)

        end_permutations_time = time.time()
        total_time = end_permutations_time - start_permutations_time
        logging.info(f"All {num_permutations} permutations completed in {total_time:.2f} seconds.")
        try:
            progress_bar.progress(1.0, text="Permutations complete! Results saved.")
            time.sleep(2)
            progress_bar.empty()
        except Exception as pb_e:
            logging.warning(f"Could not update/empty progress bar: {pb_e}")

        results_df = pd.DataFrame(all_results_list)

        base_df_columns = ["embedding_model", "vector_store", "reranker", "llm_model", "chunking_strategy", "avg_custom_score", "elapsed_time"]
        
        custom_metric_cols_df = sorted([col for col in results_df.columns if col.startswith("custom_") and col != "custom_evaluation_scores"])
        ragas_metric_cols_df = sorted([col for col in results_df.columns if col.startswith("ragas_") and col != "ragas_evaluation_scores"])
        llm_metric_cols_df = sorted([col for col in results_df.columns if col.startswith("llm_")])

        display_columns = base_df_columns + custom_metric_cols_df + ragas_metric_cols_df + llm_metric_cols_df + ["response"]

        for col in display_columns:
            if col not in results_df.columns:
                results_df[col] = pd.NA

        cols_to_numeric = ["avg_custom_score", "elapsed_time"] + custom_metric_cols_df + ragas_metric_cols_df + llm_metric_cols_df
        for col in cols_to_numeric:
            if col in results_df.columns:
                results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
        
        display_df = results_df[display_columns].copy()
        
        logging.info(f"Returning DataFrame with columns: {display_df.columns.tolist()}")
        return display_df, all_results_list 