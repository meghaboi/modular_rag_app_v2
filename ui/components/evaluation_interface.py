import streamlit as st
import logging
import time
import itertools
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from pipeline.utils.permutation_runner import PermutationRunner
from utils.api_management.api_utils import check_api_keys
from utils.text_processing.text_utils import get_csv_download_link
from utils.enums import (
    EmbeddingModelType,
    RerankerModelType,
    LLMModelType,
    VectorStoreType,
    ChunkingStrategyType
)
from pipeline.utils.pipeline_initializer import PipelineInitializer
from pipeline.components.config import PipelineConfig

class EvaluationInterface:
    """Handles the RAG evaluation interface for testing different configurations."""

    HEADER_TEXT = "🧪 RAG Evaluation Mode"
    DESCRIPTION = ("Let's test out different setups. Give me a question and the perfect answer "
                   "(ground truth) to see how well various RAG configurations perform.")
    PIPELINE_WARNING = "💡 Pipeline isn't active. Hit 'Initialize JEFF' in the sidebar."
    FILE_WARNING = "💡 Upload a document first using the sidebar!"
    METRICS_PER_ROW = 2
    MAX_DISPLAY_RESULTS = 10

    def __init__(self):
        """Initialize the EvaluationInterface."""
        st.session_state.is_evaluation_mode = True

    def display(self):
        """Display the RAG evaluation interface."""
        st.header(self.HEADER_TEXT)
        st.markdown(self.DESCRIPTION)

        self._show_warnings()
        
        col1, col2 = st.columns([1, 2])
        with col1:
            self._render_evaluation_inputs(col2)
        with col2:
            self._render_evaluation_results()

    def _show_warnings(self):
        """Display warnings for missing pipeline or file."""
        if self._is_pipeline_missing():
            st.warning(self.PIPELINE_WARNING)
        elif self._is_file_missing():
            st.warning(self.FILE_WARNING)

    def _is_pipeline_missing(self) -> bool:
        """Check if pipeline is missing but file exists."""
        return st.session_state.pipeline is None and st.session_state.file_path

    def _is_file_missing(self) -> bool:
        """Check if file is missing."""
        return not st.session_state.file_path

    def _render_evaluation_inputs(self, results_column):
        """Render the evaluation input section."""
        st.subheader("Evaluation Inputs")

        query = st.text_area("Enter your question:", height=100, key="eval_query")
        ground_truth = st.text_area("Enter the ideal 'ground truth' answer:", height=100, key="eval_ground_truth")
        st.info("Providing ground truth enables detailed RAGAS evaluation scores.")

        # Toggle: contextual RAG precompute once and reuse across permutations
        st.checkbox(
            "Contextual RAG (compute once, reuse across permutations)",
            key="use_contextual_once",
            help=(
                "When enabled with the 'Contextual' chunking strategy, the contextual chunks "
                "are computed a single time and reused for all model permutations."
            ),
        )

        is_disabled = self._are_evaluation_buttons_disabled()
        current_config_clicked = st.button("Evaluate Current Config", disabled=is_disabled)
        all_permutations_clicked = st.button("Run All Permutations Test", disabled=is_disabled)

        self._handle_evaluation_requests(current_config_clicked, all_permutations_clicked, query, ground_truth, results_column)

    def _are_evaluation_buttons_disabled(self) -> bool:
        """Determine if evaluation buttons should be disabled."""
        return st.session_state.pipeline is None or not st.session_state.file_path

    def _handle_evaluation_requests(self, current_config_clicked, all_permutations_clicked, query, ground_truth, results_column):
        """Handle evaluation button clicks."""
        if not query and (current_config_clicked or all_permutations_clicked):
            st.warning("⚠️ Please enter a question to evaluate.")
            return

        if current_config_clicked and query:
            self._evaluate_current_configuration(query, ground_truth, results_column)
        elif all_permutations_clicked and query:
            self._evaluate_all_permutations(query, ground_truth)

    def _evaluate_current_configuration(self, query: str, ground_truth: str, results_column):
        """Evaluate the current configuration."""
        if not self._validate_prerequisites():
            return

        config_enums = self._get_current_configuration_enums()
        if not config_enums:
            return

        if not self._validate_api_keys_for_configuration(config_enums):
            return

        with st.spinner("Evaluating current configuration..."):
            pipeline = self._initialize_pipeline_with_configuration(config_enums)
            if pipeline:
                st.session_state.pipeline = pipeline
                self._execute_single_evaluation(query, ground_truth, results_column)
            else:
                st.error("Failed to initialize pipeline with current configuration")

    def _validate_prerequisites(self) -> bool:
        """Validate that prerequisites are met for evaluation."""
        if not st.session_state.file_path:
            st.warning("Please upload a document first.")
            return False
        if st.session_state.pipeline is None:
            st.warning("Pipeline not initialized. Please initialize.")
            return False
        return True

    def _get_current_configuration_enums(self) -> Optional[Tuple[Any, ...]]:
        """Get the current configuration enums."""
        try:
            return (
                EmbeddingModelType.from_string(st.session_state.embedding_model),
                VectorStoreType.from_string(st.session_state.vector_store),
                RerankerModelType.from_string(st.session_state.reranker),
                LLMModelType.from_string(st.session_state.llm_model),
                ChunkingStrategyType.from_string(st.session_state.chunking_strategy)
            )
        except ValueError as e:
            st.error(f"Error reading current configuration: {e}")
            return None

    def _validate_api_keys_for_configuration(self, config_enums: Tuple[Any, ...]) -> bool:
        """Validate API keys for the current configuration."""
        missing_keys = check_api_keys(
            embedding_model_enum=config_enums[0],
            vector_store_enum=config_enums[1],
            reranker_enum=config_enums[2],
            llm_enum=config_enums[3]
        )
        if missing_keys:
            st.error(f"Missing keys for current config: {', '.join(missing_keys)}")
            return False
        return True

    def _initialize_pipeline_with_configuration(self, config_enums: Tuple[Any, ...]):
        """Initialize pipeline with the given configuration."""
        config = PipelineConfig(
            file_path=st.session_state.file_path,
            embedding_model_type=config_enums[0],
            vector_store_type=config_enums[1],
            reranker_type=config_enums[2],
            llm_type=config_enums[3],
            chunking_strategy_type=config_enums[4],
            hybrid_alpha=st.session_state.hybrid_alpha,
            chunk_size=st.session_state.chunk_size,
            chunk_overlap=st.session_state.chunk_overlap,
            top_k=st.session_state.top_k,
            evaluation_mode=True  # or use st.session_state.evaluation_mode if available
        )
        initializer = PipelineInitializer(config)
        return initializer.initialize_pipeline()

    def _execute_single_evaluation(self, query: str, ground_truth: str, results_column):
        """Execute evaluation for a single configuration."""
        logging.info("Evaluating single query using the existing pipeline.")
        start_time = time.time()

        try:
            response, contexts, metrics = st.session_state.pipeline.run(query)
            elapsed_time = time.time() - start_time

            evaluation_results = self._calculate_evaluation_scores(query, response, contexts, ground_truth)
            self._display_single_evaluation_output(results_column, response, contexts, evaluation_results, elapsed_time, metrics)

        except Exception as e:
            logging.error(f"Error running single evaluation: {e}", exc_info=True)
            st.error(f"Error processing evaluation: {str(e)}")
            with results_column:
                st.error(f"Failed to evaluate: {e}")

    def _calculate_evaluation_scores(self, query: str, response: str, contexts: List[str], ground_truth: str) -> Dict[str, Any]:
        """Calculate evaluation scores if ground truth is provided."""
        if not ground_truth:
            logging.warning("No ground truth provided, skipping evaluation metrics.")
            return {}

        try:
            evaluation_results = st.session_state.pipeline.evaluate_response(
                query=query,
                response=response,
                contexts=contexts,
                ground_truth=ground_truth
            )
            logging.info(f"Single config evaluation scores: {evaluation_results}")
            return evaluation_results
        except Exception as e:
            logging.error(f"Evaluation failed for single config run: {e}", exc_info=True)
            st.warning(f"Evaluation metrics failed: {e}")
            return {"error": str(e)}

    def _display_single_evaluation_output(self, results_column, response: str, contexts: List[str],
                                         evaluation_results: Dict[str, Any], elapsed_time: float, metrics: Dict[str, Any]):
        """Display the evaluation output for single configuration."""
        self._clear_permutation_results()

        with results_column:
            st.subheader("Evaluation Result (Current Config)")
            self._show_configuration_details()
            self._show_evaluation_metrics(elapsed_time, metrics, evaluation_results)
            self._show_response_and_contexts(response, contexts)
            self._show_evaluation_scores(evaluation_results)

    def _clear_permutation_results(self):
        """Clear previous permutation results."""
        st.session_state.permutation_df = None
        st.session_state.permutation_results = None

    def _show_configuration_details(self):
        """Display current configuration details."""
        config_display = (f"`{st.session_state.embedding_model} | {st.session_state.vector_store} | "
                          f"{st.session_state.reranker} | {st.session_state.llm_model} | "
                          f"{st.session_state.chunking_strategy}`")
        st.markdown(f"**Configuration:** {config_display}")

    def _show_evaluation_metrics(self, elapsed_time: float, metrics: Dict[str, Any], evaluation_results: Dict[str, Any]):
        """Display evaluation metrics in columns."""
        metric_columns = st.columns(4)

        with metric_columns[0]:
            st.metric(label="Processing Time", value=f"{elapsed_time:.2f}s")

        if not metrics:
            return

        input_tokens = metrics.get('input_tokens', 0)
        output_tokens = metrics.get('output_tokens', 0)
        cost = evaluation_results.get('llm_cost') if evaluation_results else metrics.get('llm_cost')

        with metric_columns[1]:
            st.metric(label="Input Tokens", value=input_tokens)
        with metric_columns[2]:
            st.metric(label="Output Tokens", value=output_tokens)
        with metric_columns[3]:
            cost_display = f"${cost:.4f}" if cost is not None else "$0.0000"
            st.metric(label="LLM Cost", value=cost_display)

    def _show_response_and_contexts(self, response: str, contexts: List[str]):
        """Display response and retrieved contexts."""
        with st.expander("Response", expanded=True):
            st.write(response)

        with st.expander("Retrieved Contexts", expanded=False):
            if contexts:
                for i, context in enumerate(contexts):
                    st.markdown(f"**Context {i + 1}:**")
                    st.text(context)
            else:
                st.write("No contexts were retrieved.")

    def _show_evaluation_scores(self, evaluation_results: Dict[str, Any]):
        """Display evaluation scores in a grid layout."""
        if not self._has_valid_evaluation_results(evaluation_results):
            self._show_evaluation_status(evaluation_results)
            return

        st.subheader("Evaluation Scores")
        score_metrics = self._extract_score_metrics(evaluation_results)
        
        if score_metrics:
            self._display_score_metrics_grid(score_metrics)
        else:
            st.info("No scores generated.")

    def _has_valid_evaluation_results(self, evaluation_results: Dict[str, Any]) -> bool:
        """Check if evaluation results are valid."""
        return evaluation_results and isinstance(evaluation_results, dict) and "error" not in evaluation_results

    def _show_evaluation_status(self, evaluation_results: Dict[str, Any]):
        """Show evaluation status when results are invalid."""
        if "error" in evaluation_results:
            st.warning(f"Scores not calculated: {evaluation_results['error']}")
        else:
            st.info("Provide ground truth to see scores.")

    def _extract_score_metrics(self, evaluation_results: Dict[str, Any]) -> List[Tuple[str, Any]]:
        """Extract score metrics excluding cost."""
        return [(name, value) for name, value in evaluation_results.items() if name != 'llm_cost']

    def _display_score_metrics_grid(self, score_metrics: List[Tuple[str, Any]]):
        """Display score metrics in a grid layout."""
        for i in range(0, len(score_metrics), self.METRICS_PER_ROW):
            current_row_metrics = score_metrics[i:i + self.METRICS_PER_ROW]
            score_columns = st.columns(len(current_row_metrics))

            for j, (metric_name, metric_value) in enumerate(current_row_metrics):
                with score_columns[j]:
                    formatted_value = self._format_metric_value(metric_value)
                    display_name = metric_name.replace('_', ' ').title()
                    st.metric(label=display_name, value=formatted_value)

    def _format_metric_value(self, value: Any) -> str:
        """Format metric value for display."""
        return f"{value:.2f}" if isinstance(value, (int, float)) else "N/A"

    def _evaluate_all_permutations(self, query: str, ground_truth: str):
        """Evaluate all possible permutations."""
        if not self._validate_prerequisites():
            return

        chunking_strategy = self._get_chunking_strategy_enum()
        if not chunking_strategy:
            return

        self._check_permutation_api_keys()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"permutation_results_{timestamp}.csv"
        st.info(f"Results will be saved to {output_filename}")

        with st.spinner("Running all permutations... This might take a while! ☕️"):
            self._execute_permutation_evaluation(query, ground_truth, chunking_strategy, output_filename)

    def _get_chunking_strategy_enum(self) -> Optional[ChunkingStrategyType]:
        """Get chunking strategy enum."""
        try:
            return ChunkingStrategyType.from_string(st.session_state.chunking_strategy)
        except ValueError as e:
            st.error(f"Invalid chunking strategy: {e}")
            return None

    def _check_permutation_api_keys(self):
        """Check API keys needed for all permutations."""
        st.info("Checking API keys potentially needed for permutations...")

        embedding_models = [EmbeddingModelType.OPENAI, EmbeddingModelType.GEMINI, EmbeddingModelType.MISTRAL]
        reranker_models = list(RerankerModelType)
        llm_models = [LLMModelType.CLAUDE_4_SONNET, LLMModelType.GEMINI]
        
        missing_keys = set()
        for embedding, reranker, llm in itertools.product(embedding_models, reranker_models, llm_models):
            missing_keys.update(check_api_keys(embedding, VectorStoreType.FAISS, reranker, llm))

        if missing_keys:
            st.warning(f"Heads up! Missing potential keys: {', '.join(missing_keys)}. Some permutations might fail.")
        else:
            st.success("Looks like all potentially required API keys are present!")

    def _execute_permutation_evaluation(self, query: str, ground_truth: str, chunking_strategy: ChunkingStrategyType, output_filename: str):
        """Execute evaluation for all permutations."""
        results_df, all_results = PermutationRunner.run_all_permutations(
            file_path=st.session_state.file_path,
            user_query=query,
            ground_truth=ground_truth,
            chunk_size=st.session_state.chunk_size,
            chunk_overlap=st.session_state.chunk_overlap,
            top_k=st.session_state.top_k,
            hybrid_alpha=st.session_state.hybrid_alpha,
            chunking_strategy_enum=chunking_strategy,
            output_csv_file=output_filename,
            use_contextual_once=bool(st.session_state.get("use_contextual_once", False))
        )

        st.session_state.permutation_df = results_df
        st.session_state.permutation_results = all_results
        logging.info("Permutations completed.")
        st.success("All permutations complete! Results below.")
        st.rerun()

    def _render_evaluation_results(self):
        """Render the evaluation results section."""
        if self._has_permutation_results():
            self._show_permutation_results()
        elif st.session_state.permutation_results is not None:
            st.info("Permutation run finished, but no valid results generated.")

    def _has_permutation_results(self) -> bool:
        """Check if permutation results exist and are not empty."""
        return (st.session_state.permutation_df is not None and 
                not st.session_state.permutation_df.empty)

    def _show_permutation_results(self):
        """Display the permutation results summary."""
        st.subheader("Permutation Results Summary")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.markdown(f"Results have been saved to `permutation_results_{timestamp}.csv` in the application's root directory.")

        download_link = get_csv_download_link(st.session_state.permutation_df, f"displayed_summary_{timestamp}.csv")
        st.markdown(download_link, unsafe_allow_html=True)

        self._display_results_table()
        st.markdown("---")
        st.subheader("Explore Individual Results")

    def _display_results_table(self):
        """Display the formatted results table."""
        results_df = self._prepare_results_for_display()
        top_results = self._get_top_results(results_df)
        display_columns = self._get_display_columns(results_df)
        format_dict = self._get_format_dictionary(results_df)

        st.dataframe(
            top_results[display_columns].style.format(format_dict, na_rep="N/A"),
            use_container_width=True
        )

    def _prepare_results_for_display(self):
        """Prepare results dataframe for display."""
        results_df = st.session_state.permutation_df.copy()
        results_df['avg_score_numeric'] = results_df['avg_custom_score'].fillna(-1)
        return results_df

    def _get_top_results(self, results_df):
        """Get top results sorted by average score."""
        return results_df.sort_values('avg_score_numeric', ascending=False).head(self.MAX_DISPLAY_RESULTS)

    def _get_display_columns(self, results_df) -> List[str]:
        """Get columns to display in results table."""
        base_columns = ["embedding_model", "vector_store", "reranker", "llm_model",
                       "chunking_strategy", "avg_custom_score", "elapsed_time"]
        metric_columns = sorted([col for col in results_df.columns if col.startswith("metric_")])
        return base_columns + metric_columns

    def _get_format_dictionary(self, results_df) -> Dict[str, str]:
        """Get format dictionary for results table."""
        format_dict = {'avg_custom_score': "{:.2f}", 'elapsed_time': "{:.2f}"}
        metric_columns = [col for col in results_df.columns if col.startswith("metric_")]
        
        for col in metric_columns:
            format_dict[col] = "{:.2f}"
        
        return format_dict

def display_evaluation_interface():
    """Factory function to create and display evaluation interface."""
    evaluation_interface = EvaluationInterface()
    evaluation_interface.display()

__all__ = ['EvaluationInterface', 'display_evaluation_interface']