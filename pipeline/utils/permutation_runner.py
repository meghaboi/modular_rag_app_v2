import logging
import time
import itertools
import pandas as pd
import csv
import streamlit as st
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field

from utils.enums import (
    EmbeddingModelType, RerankerModelType, LLMModelType, VectorStoreType,
    ChunkingStrategyType, EvaluationBackendType, EvaluationMetricType
)
from utils.api_management.api_utils import check_api_keys
from .pipeline_runner import PipelineRunner

@dataclass
class ModelCombination:
    """Represents a single combination of models for a pipeline run."""
    embedding_model: EmbeddingModelType
    vector_store: VectorStoreType
    reranker: RerankerModelType
    llm: LLMModelType

    def to_string(self) -> str:
        """Return a string representation of the model combination."""
        return f"Emb: {self.embedding_model.value}, VS: {self.vector_store.value}, Rerank: {self.reranker.value}, LLM: {self.llm.value}"

@dataclass
class PermutationConfig:
    """Configuration for which models to use in permutation runs."""
    embedding_models: List[EmbeddingModelType] = field(default_factory=lambda: [
        EmbeddingModelType.VOYAGE, EmbeddingModelType.GEMINI, EmbeddingModelType.MISTRAL
    ])
    vector_stores: List[VectorStoreType] = field(default_factory=lambda: [
        VectorStoreType.FAISS, VectorStoreType.CHROMA
    ])
    rerankers: List[RerankerModelType] = field(default_factory=lambda: 
        [r for r in RerankerModelType if r != RerankerModelType.NONE] + [RerankerModelType.NONE]
    )
    llm_models: List[LLMModelType] = field(default_factory=lambda: [
        LLMModelType.CLAUDE_3_SONNET, LLMModelType.GEMINI
    ])

    def get_permutations(self) -> List[ModelCombination]:
        """Generate all model combinations from the configuration."""
        product = itertools.product(
            self.embedding_models, self.vector_stores, self.rerankers, self.llm_models
        )
        return [ModelCombination(*p) for p in product]

class PermutationRunner:
    """Runs a pipeline with all permutations of configured models."""
    def __init__(self, file_path: str, user_query: str, ground_truth: str, 
                 chunk_size: int, chunk_overlap: int, top_k: int, hybrid_alpha: float,
                 chunking_strategy: ChunkingStrategyType, 
                 output_csv_file: str = "permutation_results.csv",
                 perm_config: PermutationConfig = None):
        
        self.pipeline_params = {
            "file_path": file_path, "user_query": user_query, "ground_truth": ground_truth,
            "chunk_size": chunk_size, "chunk_overlap": chunk_overlap, "top_k": top_k,
            "hybrid_alpha": hybrid_alpha, "chunking_strategy_enum": chunking_strategy
        }
        self.output_csv_file = output_csv_file
        self.perm_config = perm_config or PermutationConfig()
        self.all_results = []

    def run_all_permutations(self) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Run all configured permutations and return the results."""
        permutations = self.perm_config.get_permutations()
        num_perms = len(permutations)
        logging.info(f"Starting 'Run All Permutations' for {num_perms} combinations. Results will be saved to {self.output_csv_file}")

        progress_bar = st.progress(0, text="Starting permutations...")
        start_time = time.time()

        with open(self.output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = None
            for i, combo in enumerate(permutations):
                self._update_progress(progress_bar, i, num_perms, combo)
                result = self._run_single_permutation(combo)
                self.all_results.append(result)
                
                if csv_writer is None:
                    csv_writer = self._create_csv_writer(result, csvfile)
                
                self._write_csv_row(csv_writer, result)

        self._finalize_run(start_time, num_perms, progress_bar)
        return self._format_results_dataframe(), self.all_results

    def _run_single_permutation(self, combo: ModelCombination) -> Dict[str, Any]:
        """Run a single pipeline permutation and return its result."""
        logging.info(f"Running permutation: {combo.to_string()}")
        missing_keys = check_api_keys(combo.embedding_model, combo.vector_store, combo.reranker, combo.llm)

        if missing_keys:
            st.warning(f"Skipping {combo.to_string()} due to missing keys: {', '.join(missing_keys)}")
            return self._create_skipped_result(combo)

        return PipelineRunner.run_pipeline_with_config(
            **self.pipeline_params,
            embedding_model_enum=combo.embedding_model,
            vector_store_enum=combo.vector_store,
            reranker_enum=combo.reranker,
            llm_enum=combo.llm
        )

    def _create_skipped_result(self, combo: ModelCombination) -> Dict[str, Any]:
        """Create a result dictionary for a skipped permutation."""
        return {
            "embedding_model": combo.embedding_model.value, "vector_store": combo.vector_store.value,
            "reranker": combo.reranker.value, "llm_model": combo.llm.value,
            "chunking_strategy": self.pipeline_params["chunking_strategy_enum"].value,
            "response": "SKIPPED - Missing API Keys", "avg_custom_score": 0, "elapsed_time": 0, "contexts": [],
            **{f"custom_{m.value.lower().replace(' ', '_')}": "N/A" for m in EvaluationMetricType.get_metrics_for_backend(EvaluationBackendType.CUSTOM)},
            **{f"ragas_{m.value.lower().replace(' ', '_')}": "N/A" for m in EvaluationMetricType.get_metrics_for_backend(EvaluationBackendType.RAGAS_V2)}
        }

    def _create_csv_writer(self, result: Dict[str, Any], csvfile) -> csv.DictWriter:
        """Create a CSV writer and write the header."""
        flat_result = self._flatten_result(result)
        writer = csv.DictWriter(csvfile, fieldnames=list(flat_result.keys()))
        writer.writeheader()
        return writer

    def _write_csv_row(self, writer: csv.DictWriter, result: Dict[str, Any]):
        """Write a single result row to the CSV file."""
        flat_result = self._flatten_result(result)
        row_to_write = {field: flat_result.get(field, "N/A") for field in writer.fieldnames}
        writer.writerow(row_to_write)
        writer.f.flush()

    def _flatten_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten a nested result dictionary for CSV writing."""
        flat = {k: v for k, v in result.items() if not isinstance(v, (list, dict)) or k == "response"}
        for prefix, key in [("custom", "custom_evaluation_scores"), ("ragas", "ragas_evaluation_scores"), ("llm", "metrics")]:
            scores = result.get(key, {})
            if isinstance(scores, dict):
                flat.update({f"{prefix}_{k}": v for k, v in scores.items()})
        return flat

    def _update_progress(self, bar, index: int, total: int, combo: ModelCombination):
        """Update the Streamlit progress bar."""
        try:
            progress_text = f"Permutation {index + 1}/{total}: {combo.embedding_model.value[:5]}..-{combo.llm.value[:5]}.."
            bar.progress((index + 1) / total, text=progress_text)
        except Exception as e:
            logging.warning(f"Could not update progress bar: {e}")

    def _finalize_run(self, start_time: float, num_perms: int, bar):
        """Log completion and update the progress bar."""
        total_time = time.time() - start_time
        logging.info(f"All {num_perms} permutations completed in {total_time:.2f} seconds.")
        try:
            bar.progress(1.0, text="Permutations complete! Results saved.")
            time.sleep(2)
            bar.empty()
        except Exception as e:
            logging.warning(f"Could not update/empty progress bar: {e}")

    def _format_results_dataframe(self) -> pd.DataFrame:
        """Create and format the final results DataFrame."""
        df = pd.DataFrame(self.all_results)
        base_cols = ["embedding_model", "vector_store", "reranker", "llm_model", "chunking_strategy", "avg_custom_score", "elapsed_time"]
        custom_cols = sorted([c for c in df.columns if c.startswith("custom_") and c != "custom_evaluation_scores"])
        ragas_cols = sorted([c for c in df.columns if c.startswith("ragas_") and c != "ragas_evaluation_scores"])
        llm_cols = sorted([c for c in df.columns if c.startswith("llm_")])
        
        display_columns = base_cols + custom_cols + ragas_cols + llm_cols + ["response"]
        df = df.reindex(columns=display_columns, fill_value=pd.NA)

        cols_to_numeric = ["avg_custom_score", "elapsed_time"] + custom_cols + ragas_cols + llm_cols
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logging.info(f"Returning DataFrame with columns: {df.columns.tolist()}")
        return df