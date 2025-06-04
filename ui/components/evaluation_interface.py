import streamlit as st
import logging
import time
import itertools
from datetime import datetime
from typing import Dict, Any, List, Tuple
from .shared import (
    check_api_keys, get_csv_download_link, run_all_permutations,
    EmbeddingModelType, RerankerModelType, LLMModelType,
    VectorStoreType, ChunkingStrategyType
)

def display_evaluation_interface():
    """Display the RAG evaluation interface."""
    st.header("🧪 RAG Evaluation Mode")
    st.markdown("Let's test out different setups. Give me a question and the perfect answer (ground truth) to see how well various RAG configurations perform.")

    if st.session_state.pipeline is None and st.session_state.file_path:
        st.warning("💡 Pipeline isn't active. Hit 'Initialize JEFF' in the sidebar.")
    elif not st.session_state.file_path:
        st.warning("💡 Upload a document first using the sidebar!")

    col1, col2 = st.columns([1, 2])

    with col1:
        display_evaluation_inputs(col2)

    with col2:
        display_evaluation_results()

def display_evaluation_inputs(col2):
    """Display the evaluation input section."""
    st.subheader("Evaluation Inputs")
    user_query = st.text_area("Enter your question:", height=100, key="eval_query")
    ground_truth = st.text_area("Enter the ideal 'ground truth' answer:", height=100, key="eval_ground_truth")
    st.info("Providing ground truth enables detailed RAGAS evaluation scores.")

    disable_eval_buttons = st.session_state.pipeline is None or not st.session_state.file_path
    process_button = st.button("Evaluate Current Config", disabled=disable_eval_buttons)
    permutation_button = st.button("Run All Permutations Test", disabled=disable_eval_buttons)

    if process_button and user_query:
        handle_single_evaluation(user_query, ground_truth, col2)
    elif permutation_button and user_query:
        handle_permutation_evaluation(user_query, ground_truth)
    elif (process_button or permutation_button) and not user_query:
        st.warning("⚠️ Please enter a question to evaluate.")

def handle_single_evaluation(user_query: str, ground_truth: str, col2):
    """Handle evaluation of a single configuration."""
    if not st.session_state.file_path:
        st.warning("Please upload a document first.")
        return
    if st.session_state.pipeline is None:
        st.warning("Pipeline not initialized. Please initialize.")
        return

    try:
        config_enums = get_configuration_enums()
    except ValueError as e:
        st.error(f"Error reading current configuration: {e}")
        return

    missing_keys = check_api_keys(*config_enums)
    if missing_keys:
        st.error(f"Missing keys for current config: {', '.join(missing_keys)}")
        return

    with st.spinner("Evaluating current configuration..."):
        run_single_evaluation(user_query, ground_truth, col2)

def get_configuration_enums() -> Tuple[Any, ...]:
    """Get the current configuration enums."""
    return (
        EmbeddingModelType.from_string(st.session_state.embedding_model),
        VectorStoreType.from_string(st.session_state.vector_store),
        RerankerModelType.from_string(st.session_state.reranker),
        LLMModelType.from_string(st.session_state.llm_model)
    )

def run_single_evaluation(user_query: str, ground_truth: str, col2):
    """Run evaluation for a single configuration."""
    logging.info("Evaluating single query using the existing pipeline.")
    start_eval_time = time.time()
    try:
        response, contexts, metrics = st.session_state.pipeline.run(user_query)
        eval_elapsed_time = time.time() - start_eval_time

        evaluation_results = process_evaluation_results(
            user_query, response, contexts, ground_truth, metrics
        )

        display_evaluation_output(
            col2, response, contexts, evaluation_results,
            eval_elapsed_time, metrics
        )

    except Exception as e:
        logging.error(f"Error running single evaluation: {e}", exc_info=True)
        st.error(f"Error processing evaluation: {str(e)}")
        with col2:
            st.error(f"Failed to evaluate: {e}")

def process_evaluation_results(user_query: str, response: str, contexts: List[str],
                            ground_truth: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Process and return evaluation results."""
    evaluation_results = {}
    if ground_truth:
        try:
            evaluation_results = st.session_state.pipeline.evaluate_response(
                query=user_query,
                response=response,
                contexts=contexts,
                ground_truth=ground_truth
            )
            logging.info(f"Single config evaluation scores: {evaluation_results}")
        except Exception as eval_e:
            logging.error(f"Evaluation failed for single config run: {eval_e}", exc_info=True)
            st.warning(f"Evaluation metrics failed: {eval_e}")
            evaluation_results = {"error": str(eval_e)}
    else:
        logging.warning("No ground truth provided, skipping evaluation metrics.")
    
    return evaluation_results

def display_evaluation_output(col2, response: str, contexts: List[str],
                            evaluation_results: Dict[str, Any], eval_elapsed_time: float,
                            metrics: Dict[str, Any]):
    """Display the evaluation output."""
    st.session_state.permutation_df = None
    st.session_state.permutation_results = None
    
    with col2:
        st.subheader("Evaluation Result (Current Config)")
        st.markdown(f"**Configuration:** `{st.session_state.embedding_model} | {st.session_state.vector_store} | {st.session_state.reranker} | {st.session_state.llm_model} | {st.session_state.chunking_strategy}`")
        
        display_metrics(eval_elapsed_time, metrics, evaluation_results)
        display_response_and_contexts(response, contexts)
        display_evaluation_scores(evaluation_results)

def display_metrics(eval_elapsed_time: float, metrics: Dict[str, Any],
                   evaluation_results: Dict[str, Any]):
    """Display evaluation metrics."""
    summary_metric_cols = st.columns(4)
    with summary_metric_cols[0]:
        st.metric(label="Processing Time", value=f"{eval_elapsed_time:.2f}s")
    
    if metrics:
        input_tokens = metrics.get('input_tokens', 0)
        output_tokens = metrics.get('output_tokens', 0)
        
        with summary_metric_cols[1]:
            st.metric(label="Input Tokens", value=input_tokens)
        with summary_metric_cols[2]:
            st.metric(label="Output Tokens", value=output_tokens)
        
        with summary_metric_cols[3]:
            cost = evaluation_results.get('llm_cost') if evaluation_results else metrics.get('llm_cost')
            cost_display_value = f"${cost:.4f}" if cost is not None else "$0.0000"
            st.metric(label="LLM Cost", value=cost_display_value)

def display_response_and_contexts(response: str, contexts: List[str]):
    """Display response and contexts."""
    with st.expander("Response", expanded=True):
        st.write(response)
    with st.expander("Retrieved Contexts", expanded=False):
        if contexts:
            for i, ctx in enumerate(contexts):
                st.markdown(f"**Context {i+1}:**")
                st.text(ctx)
        else:
            st.write("No contexts were retrieved.")

def display_evaluation_scores(evaluation_results: Dict[str, Any]):
    """Display evaluation scores."""
    if evaluation_results and isinstance(evaluation_results, dict) and "error" not in evaluation_results:
        st.subheader("Evaluation Scores")
        if evaluation_results:
            metrics_list = [(name, val) for name, val in evaluation_results.items() 
                          if name != 'llm_cost']
            num_metrics = len(metrics_list)
            metrics_per_row = 2

            for i in range(0, num_metrics, metrics_per_row):
                current_row_metrics = metrics_list[i:i + metrics_per_row]
                score_display_cols = st.columns(len(current_row_metrics))
                for k, (metric_name, metric_value) in enumerate(current_row_metrics):
                    with score_display_cols[k]:
                        score_val_display = f"{metric_value:.2f}" if isinstance(metric_value, (int, float)) else "N/A"
                        st.metric(label=metric_name.replace('_', ' ').title(), value=score_val_display)
        else:
            st.info("No scores generated.")
    elif "error" in evaluation_results:
        st.warning(f"Scores not calculated: {evaluation_results['error']}")
    else:
        st.info("Provide ground truth to see scores.")

def handle_permutation_evaluation(user_query: str, ground_truth: str):
    """Handle evaluation of all permutations."""
    if not st.session_state.file_path:
        st.warning("Please upload a document first.")
        return
    if st.session_state.pipeline is None:
        st.warning("Pipeline not initialized. Please initialize.")
        return

    check_permutation_api_keys()
    
    try:
        chunking_strategy_enum = ChunkingStrategyType.from_string(st.session_state.chunking_strategy)
    except ValueError as e:
        st.error(f"Invalid chunking strategy: {e}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv_filename = f"permutation_results_{timestamp}.csv"
    st.info(f"Results will be saved to {output_csv_filename}")

    with st.spinner("Running all permutations... This might take a while! ☕️"):
        run_permutation_evaluation(user_query, ground_truth, chunking_strategy_enum, output_csv_filename)

def check_permutation_api_keys():
    """Check API keys needed for permutations."""
    st.info("Checking API keys potentially needed for permutations...")
    perm_emb = [EmbeddingModelType.OPENAI, EmbeddingModelType.GEMINI, EmbeddingModelType.MISTRAL]
    perm_rerank = list(RerankerModelType)
    perm_llm = [LLMModelType.CLAUDE_37_SONNET, LLMModelType.GEMINI]
    potential_missing = set()
    
    for e_perm, r_perm, l_perm in itertools.product(perm_emb, perm_rerank, perm_llm):
        potential_missing.update(check_api_keys(e_perm, VectorStoreType.FAISS, r_perm, l_perm))
    
    if potential_missing:
        st.warning(f"Heads up! Missing potential keys: {', '.join(potential_missing)}. Some permutations might fail.")
    else:
        st.success("Looks like all potentially required API keys are present!")

def run_permutation_evaluation(user_query: str, ground_truth: str,
                             chunking_strategy_enum: ChunkingStrategyType,
                             output_csv_filename: str):
    """Run evaluation for all permutations."""
    results_df, all_results = run_all_permutations(
        file_path=st.session_state.file_path,
        user_query=user_query,
        ground_truth=ground_truth,
        chunk_size=st.session_state.chunk_size,
        chunk_overlap=st.session_state.chunk_overlap,
        top_k=st.session_state.top_k,
        hybrid_alpha=st.session_state.hybrid_alpha,
        chunking_strategy_enum=chunking_strategy_enum,
        output_csv_file=output_csv_filename
    )
    
    st.session_state.permutation_df = results_df
    st.session_state.permutation_results = all_results
    logging.info("Permutations completed.")
    st.success("All permutations complete! Results below.")
    st.rerun()

def display_evaluation_results():
    """Display the evaluation results section."""
    if st.session_state.permutation_df is not None and not st.session_state.permutation_df.empty:
        display_permutation_results()
    elif st.session_state.permutation_results is not None:
        st.info("Permutation run finished, but no valid results generated.")

def display_permutation_results():
    """Display the permutation results."""
    st.subheader("Permutation Results Summary")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.markdown(f"Results have been saved to `permutation_results_{timestamp}.csv` in the application's root directory.")
    st.markdown(get_csv_download_link(st.session_state.permutation_df, f"displayed_summary_{timestamp}.csv"), unsafe_allow_html=True)

    results_to_display = st.session_state.permutation_df.copy()
    results_to_display['avg_score_numeric'] = results_to_display['avg_score'].fillna(-1)
    top_results = results_to_display.sort_values('avg_score_numeric', ascending=False).head(10)

    display_cols = ["embedding_model", "vector_store", "reranker", "llm_model", 
                   "chunking_strategy", "avg_score", "elapsed_time"]
    metric_cols_exist = sorted([col for col in results_to_display.columns if col.startswith("metric_")])
    display_cols.extend(metric_cols_exist)

    format_dict = {'avg_score': "{:.2f}", 'elapsed_time': "{:.2f}"}
    for col in metric_cols_exist:
        format_dict[col] = "{:.2f}"
    
    st.dataframe(top_results[display_cols].style.format(format_dict, na_rep="N/A"), 
                use_container_width=True)

    st.markdown("---")
    st.subheader("Explore Individual Results")
    display_individual_results() 