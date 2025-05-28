import streamlit as st
import logging
import time
import itertools
from typing import List, Dict, Any
from utils import text_to_speech, is_greeting, check_api_keys, get_csv_download_link
from subject_handler import update_rag_configuration # Modified import
from pipeline_utils import initialize_pipeline
from enums import (
    EmbeddingModelType,
    RerankerModelType,
    LLMModelType,
    VectorStoreType,
    ChunkingStrategyType
)
from pipeline_utils import run_all_permutations
from token_utils import TokenCostManager
from datetime import datetime

def display_chat_interface():
    st.header("💬 Chat with JEFF")
    st.markdown("Hey! Got questions about your textbook? Lay 'em on me. I'll break it down for ya.")

    if not st.session_state.messages:
         welcome_msg = "Alright, let's get this study session started! What's on your mind?"
         welcome_audio_bytes = text_to_speech(welcome_msg)
         st.session_state.messages.append({
             "role": "assistant",
             "content": welcome_msg,
             "audio": welcome_audio_bytes,
             "contexts": [],
             "elapsed_time": None
         })

    # Display message history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                response_text = message.get("content")
                audio_data = message.get("audio") 
                contexts = message.get("contexts", [])
                elapsed_time = message.get("elapsed_time")

                if response_text:
                    tab_labels = ["📖 Read Response", "🔊 Hear Response"]
                    try:
                        tab_text, tab_audio = st.tabs(tab_labels)
                    except Exception as e:
                        logging.error(f"Error creating tabs: {e}")
                        st.write(response_text) 
                        if audio_data: st.audio(audio_data, format="audio/mp3")
                        tab_text = None 

                    if tab_text: 
                        with tab_text:
                            st.write(response_text)

                        with tab_audio:
                            if audio_data:
                                st.audio(audio_data, format="audio/mp3")
                            else:
                                st.info("Audio playback is not available for this message.")

                    if elapsed_time is not None:
                         st.write(f"_(JEFF cooked that up in {elapsed_time:.2f} seconds)_")

                    if st.session_state.show_contexts and contexts:
                         with st.expander("🧠 Check out the textbook bits I used:"):
                            for i, context in enumerate(contexts):
                                st.markdown(f"**Snippet {i+1}:**")
                                st.text(context)

                else: 
                    st.write("*Assistant message content missing.*")

            else: 
                st.write(message["content"])

    user_query = st.chat_input("Type your question here...")

    if user_query:
        logging.info(f"User query received: {user_query}")
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        if st.session_state.pipeline is None:
            logging.warning("Chat query received, but pipeline not initialized.")
            warning_msg = "Whoa there! Looks like we haven't loaded your textbook into my brain yet. Upload it and hit 'Initialize' in the sidebar first!"
            warning_audio = text_to_speech(warning_msg)
            with st.chat_message("assistant"):
                 tab_labels_warn = ["📖 Read Message", "🔊 Hear Message"]
                 tab_warn_text, tab_warn_audio = st.tabs(tab_labels_warn)
                 with tab_warn_text: st.warning(warning_msg, icon="✋")
                 with tab_warn_audio:
                     if warning_audio: st.audio(warning_audio, format="audio/mp3")
                     else: st.info("Audio playback not available.")

            st.session_state.messages.append({
                "role": "assistant", "content": warning_msg, "audio": warning_audio,
                "contexts": [], "elapsed_time": None
            })
            st.stop()
        
        # === New Dynamic RAG Configuration Update ===
        if st.session_state.pipeline: # Ensure pipeline exists
            logging.info(f"Attempting to update RAG configuration for query: {user_query[:100]}...")
            # Pass current subject if available, otherwise it defaults to None
            current_subject = st.session_state.get('current_subject', None)
            config_update_status = update_rag_configuration(
                query=user_query, 
                pipeline=st.session_state.pipeline,
                subject=current_subject 
            )
            if config_update_status is False: # Explicitly check for False (failure)
                st.error("⚠️ Error updating RAG configuration based on your query. Using previous settings. Check logs for details.")
                logging.error("update_rag_configuration returned False, indicating a failure during re-initialization.")
            elif config_update_status is True:
                st.toast("✨ Smartly adjusted RAG settings for your query!")
                logging.info("update_rag_configuration returned True. Pipeline re-initialized.")
            else: # config_update_status is None
                logging.info("update_rag_configuration returned None. No changes to RAG configuration were necessary.")
        # ============================================

        # Dynamic Configuration Logic (OLD - to be commented out/removed)
        # config_changed = False
        # if 'current_subject' in st.session_state and st.session_state.current_subject:
        #     subject = st.session_state.current_subject
        #     logging.info(f"Current subject '{subject}' and user query '{user_query}' found, attempting dynamic configuration.")
        #     try:
        #         # This line calls the old, deprecated function.
        #         # from subject_handler import get_subject_configuration # Make sure this is the old one
        #         # new_config = get_subject_configuration(subject, user_query) 
        #         logging.warning("Old dynamic configuration block is active but should be replaced by update_rag_configuration call.")
        #         # if new_config:
        #         #     logging.info(f"Received dynamic config: {new_config}")
                    
        #         #     # Process chunk_size
        #         #     if 'chunk_size' in new_config and new_config['chunk_size'] != st.session_state.get('chunk_size'):
        #         #         st.session_state.chunk_size = new_config['chunk_size']
        #         #         config_changed = True
        #         #         logging.info(f"Updated chunk_size to {st.session_state.chunk_size}")
                    
        #         #     # Process chunk_overlap
        #         #     if 'chunk_overlap' in new_config and new_config['chunk_overlap'] != st.session_state.get('chunk_overlap'):
        #         #         st.session_state.chunk_overlap = new_config['chunk_overlap']
        #         #         config_changed = True
        #         #         logging.info(f"Updated chunk_overlap to {st.session_state.chunk_overlap}")

        #         #     # Log other params (not applying them yet)
        #         #     other_params = {k: v for k, v in new_config.items() if k not in ['chunk_size', 'chunk_overlap']}
        #         #     if other_params:
        #         #         logging.info(f"Other dynamic config params received (not applied): {other_params}")

        #         #     if config_changed:
        #         #         logging.info("Configuration changed, re-initializing pipeline.")
        #         #         try:
        #         #             embedding_enum = EmbeddingModelType.from_string(st.session_state.embedding_model)
        #         #             vs_enum = VectorStoreType.from_string(st.session_state.vector_store)
        #         #             reranker_enum = RerankerModelType.from_string(st.session_state.reranker)
        #         #             llm_enum = LLMModelType.from_string(st.session_state.llm_model)
        #         #             cs_enum = ChunkingStrategyType.from_string(st.session_state.chunking_strategy)

        #         #             pipeline_instance = initialize_pipeline(
        #         #                 file_path=st.session_state.file_path,
        #         #                 embedding_model_enum=embedding_enum,
        #         #                 vector_store_enum=vs_enum,
        #         #                 reranker_enum=reranker_enum,
        #         #                 llm_enum=llm_enum,
        #         #                 chunking_strategy_enum=cs_enum,
        #         #                 hybrid_alpha=st.session_state.hybrid_alpha,
        #         #                 chunk_size=st.session_state.chunk_size, # Use updated value
        #         #                 chunk_overlap=st.session_state.chunk_overlap, # Use updated value
        #         #                 top_k=st.session_state.top_k
        #         #             )
        #         #             if pipeline_instance:
        #         #                 st.session_state.pipeline = pipeline_instance
        #         #                 logging.info("Pipeline re-initialized successfully with new dynamic configuration.")
        #         #                 st.toast("✨ Smartly adjusted settings for your query!")
        #         #             else:
        #         #                 logging.error("Failed to re-initialize pipeline with dynamic config, initialize_pipeline returned None.")
        #         #                 st.warning("⚠️ Couldn't apply smart adjustments. Using previous settings.")
        #         #         except Exception as e_reinit:
        #         #             logging.error(f"Error re-initializing pipeline with dynamic config: {e_reinit}", exc_info=True)
        #         #             st.warning(f"⚠️ Error applying smart adjustments: {e_reinit}. Using previous settings.")
        #         #     else:
        #         #         logging.info("Dynamic configuration received, but no changes to chunk_size or chunk_overlap. No re-initialization needed.")
        #         # else:
        #         #     logging.info("get_subject_configuration returned None. No dynamic changes applied.")
        #     # except Exception as e_dyn_config:
        #     #     logging.error(f"Error during dynamic configuration call: {e_dyn_config}", exc_info=True)
        #     #     st.warning(f"⚠️ Could not fetch dynamic settings for your query ({e_dyn_config}). Using current settings.")
        # else:
        #     logging.info("Dynamic configuration skipped: 'current_subject' not in session state or is None.")

        # Check if it's a greeting
        is_greet, greeting_response = is_greeting(user_query)
        if is_greet:
            greeting_audio = text_to_speech(greeting_response)
            
            with st.chat_message("assistant"):
                tab_labels_greet = ["📖 Read Response", "🔊 Hear Response"]
                tab_greet_text, tab_greet_audio = st.tabs(tab_labels_greet)
                
                with tab_greet_text:
                    st.write(greeting_response)
                
                with tab_greet_audio:
                    if greeting_audio:
                        st.audio(greeting_audio, format="audio/mp3")
                    else:
                        st.info("Audio playback not available.")
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": greeting_response,
                    "audio": greeting_audio,
                    "contexts": [],
                    "elapsed_time": None
                })
            return

        # Start streaming response process for non-greeting queries
        with st.chat_message("assistant"):
            start_time = time.time()
            
            try:
                logging.info("Fetching contexts from vector store...")
                contexts = st.session_state.pipeline.retrieve_context(user_query)
                
                tab_labels_stream = ["📖 Read Response", "🔊 Hear Response"]
                tab_stream_text, tab_stream_audio = st.tabs(tab_labels_stream)
                
                with tab_stream_text:
                    stream_placeholder = st.empty()
                
                with tab_stream_audio:
                    audio_placeholder = st.empty()
                    audio_placeholder.info("Audio will be available when response is complete.")
                
                logging.info("Starting streaming generation...")
                full_response = ""
                
                for chunk in st.session_state.pipeline.stream_run(user_query):
                    if chunk is not None:
                        full_response += chunk
                        with tab_stream_text:
                            stream_placeholder.markdown(full_response + "▌")
                    else:
                        logging.warning("Received None chunk from stream_run, skipping")
                
                with tab_stream_text:
                    stream_placeholder.markdown(full_response)
                
                elapsed_time = time.time() - start_time
                st.write(f"_(JEFF cooked that up in {elapsed_time:.2f} seconds)_")
                
                logging.info("Generating TTS audio for the complete response...")
                tts_start_time = time.time()
                audio_bytes = text_to_speech(full_response)
                tts_elapsed_time = time.time() - tts_start_time
                
                log_msg = f"TTS generation {'succeeded' if audio_bytes else 'failed/skipped'} in {tts_elapsed_time:.2f}s."
                if audio_bytes: 
                    logging.info(log_msg)
                    with tab_stream_audio:
                        audio_placeholder.audio(audio_bytes, format="audio/mp3")
                else: 
                    logging.warning(log_msg)
                    with tab_stream_audio:
                        audio_placeholder.info("Audio playback is not available for this message.")
                
                if st.session_state.show_contexts and contexts:
                    with st.expander("🧠 Check out the textbook bits I used:"):
                        for i, context in enumerate(contexts):
                            st.markdown(f"**Snippet {i+1}:**")
                            st.text(context)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response, 
                    "contexts": contexts,
                    "elapsed_time": elapsed_time, 
                    "audio": audio_bytes
                })
                
            except Exception as e:
                logging.error(f"Error processing query or generating audio: {e}", exc_info=True)
                error_msg = f"Oof, hit a snag trying to answer that. Maybe try rephrasing? Error: {str(e)}"
                error_audio = text_to_speech(error_msg)
                
                tab_labels_err = ["📖 Read Error", "🔊 Hear Error"]
                tab_err_text, tab_err_audio = st.tabs(tab_labels_err)
                
                with tab_err_text: 
                    st.error(error_msg, icon="🔥")
                
                with tab_err_audio:
                    if error_audio: 
                        st.audio(error_audio, format="audio/mp3")
                    else: 
                        st.info("Audio playback not available.")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg, 
                    "audio": error_audio,
                    "contexts": [], 
                    "elapsed_time": None
                })

def display_evaluation_interface():
    st.header("🧪 RAG Evaluation Mode")
    st.markdown("Let's test out different setups. Give me a question and the perfect answer (ground truth) to see how well various RAG configurations perform.")

    if st.session_state.pipeline is None and st.session_state.file_path:
        st.warning("💡 Pipeline isn't active. Hit 'Initialize JEFF' in the sidebar.")
    elif not st.session_state.file_path:
         st.warning("💡 Upload a document first using the sidebar!")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Evaluation Inputs")
        user_query = st.text_area("Enter your question:", height=100, key="eval_query")
        ground_truth = st.text_area("Enter the ideal 'ground truth' answer:", height=100, key="eval_ground_truth")
        st.info("Providing ground truth enables detailed RAGAS evaluation scores.")

        disable_eval_buttons = st.session_state.pipeline is None or not st.session_state.file_path
        process_button = st.button("Evaluate Current Config", disabled=disable_eval_buttons)
        permutation_button = st.button("Run All Permutations Test", disabled=disable_eval_buttons)

    if process_button and user_query:
        if not st.session_state.file_path: st.warning("Please upload a document first.")
        elif st.session_state.pipeline is None: st.warning("Pipeline not initialized. Please initialize.")
        else:
            try: 
                 embedding_model_enum = EmbeddingModelType.from_string(st.session_state.embedding_model)
                 vector_store_enum = VectorStoreType.from_string(st.session_state.vector_store)
                 reranker_enum = RerankerModelType.from_string(st.session_state.reranker)
                 llm_enum = LLMModelType.from_string(st.session_state.llm_model)
                 chunking_strategy_enum = ChunkingStrategyType.from_string(st.session_state.chunking_strategy)
            except ValueError as e:
                 st.error(f"Error reading current configuration: {e}"); st.stop()

            missing_keys = check_api_keys(embedding_model_enum, vector_store_enum, reranker_enum, llm_enum)
            if missing_keys: st.error(f"Missing keys for current config: {', '.join(missing_keys)}")
            else:
                with st.spinner("Evaluating current configuration..."):
                    logging.info("Evaluating single query using the existing pipeline.")
                    start_eval_time = time.time()
                    try: 
                        print(user_query)
                        response, contexts, metrics = st.session_state.pipeline.process_query(user_query)
                        eval_elapsed_time = time.time() - start_eval_time

                        evaluation_results = {}
                        avg_score = 0
                        valid_scores = []
                        if ground_truth:
                             try:
                                evaluation_results = st.session_state.pipeline.evaluate_response(
                                    query=user_query,
                                    response=response,
                                    contexts=contexts,
                                    ground_truth=ground_truth
                                )
                                if evaluation_results and isinstance(evaluation_results, dict):
                                     valid_scores = [v for v in evaluation_results.values() if isinstance(v, (int, float))]
                                     if valid_scores: avg_score = sum(valid_scores) / len(valid_scores)
                                logging.info(f"Single config evaluation scores: {evaluation_results}")
                             except Exception as eval_e:
                                logging.error(f"Evaluation failed for single config run: {eval_e}", exc_info=True)
                                st.warning(f"Evaluation metrics failed: {eval_e}")
                                evaluation_results = {"error": str(eval_e)}
                        else: logging.warning("No ground truth provided, skipping evaluation metrics.")

                        st.session_state.permutation_df = None 
                        st.session_state.permutation_results = None
                        with col2:
                            st.subheader("Evaluation Result (Current Config)")
                            st.markdown(f"**Configuration:** `{st.session_state.embedding_model} | {st.session_state.vector_store} | {st.session_state.reranker} | {st.session_state.llm_model} | {st.session_state.chunking_strategy}`")
                            
                            # Layout for Time and Tokens using st.metric
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
                                
                                # Display LLM Cost from evaluation_results or metrics
                                with summary_metric_cols[3]:
                                    cost = None
                                    # Try to get cost from evaluation_results first (which should include metrics)
                                    if evaluation_results and isinstance(evaluation_results, dict) and "llm_cost" in evaluation_results:
                                        cost = evaluation_results.get('llm_cost')
                                    # Fallback to metrics if not in evaluation_results (e.g., if evaluation step was skipped)
                                    elif metrics and "llm_cost" in metrics:
                                        cost = metrics.get('llm_cost')
                                    
                                    cost_display_value = "N/A"
                                    if cost is not None and isinstance(cost, (float, int)):
                                        cost_display_value = f"${cost:.4f}"
                                    elif cost is None and metrics: # If cost is explicitly None but metrics were processed
                                        cost_display_value = "$0.0000" # Or indicate it was processed but no cost applicable
                                    
                                    st.metric(label="LLM Cost", value=cost_display_value)

                            with st.expander("Response", expanded=True): st.write(response)
                            with st.expander("Retrieved Contexts", expanded=False):
                                 if contexts:
                                    for i, ctx in enumerate(contexts): st.markdown(f"**Context {i+1}:**"); st.text(ctx)
                                 else: st.write("No contexts were retrieved.")

                            if evaluation_results and isinstance(evaluation_results, dict) and "error" not in evaluation_results:
                                 st.subheader("Evaluation Scores")
                                 if evaluation_results:
                                     metrics_list = list(evaluation_results.items())
                                     # Filter out 'llm_cost' if it accidentally appears in RAGAS results
                                     metrics_list = [(name, val) for name, val in metrics_list if name != 'llm_cost']
                                     num_metrics = len(metrics_list)
                                     metrics_per_row = 2 # Max 2 metrics per row

                                     for i_loop_var in range(0, num_metrics, metrics_per_row):
                                         current_row_metrics_data = metrics_list[i_loop_var : i_loop_var + metrics_per_row]
                                         # Create columns for the current row of metrics
                                         score_display_cols = st.columns(len(current_row_metrics_data)) # Will be 1 or 2 columns
                                         for k_loop_var, (metric_name, metric_value) in enumerate(current_row_metrics_data):
                                             with score_display_cols[k_loop_var]:
                                                  score_val_display = f"{metric_value:.2f}" if isinstance(metric_value, (int, float)) else "N/A"
                                                  st.metric(label=metric_name.replace('_', ' ').title(), value=score_val_display)
                                 else: st.info("No scores generated.")
                            elif "error" in evaluation_results: st.warning(f"Scores not calculated: {evaluation_results['error']}")
                            elif ground_truth: st.warning("Scores could not be calculated.")
                            else: st.info("Provide ground truth to see scores.")

                    except Exception as e:
                        logging.error(f"Error running single evaluation: {e}", exc_info=True)
                        st.error(f"Error processing evaluation: {str(e)}")
                        with col2: st.error(f"Failed to evaluate: {e}")

    elif permutation_button and user_query:
        if not st.session_state.file_path: st.warning("Please upload a document first.")
        elif st.session_state.pipeline is None: st.warning("Pipeline not initialized. Please initialize.")
        else:
            st.info("Checking API keys potentially needed for permutations...")
            perm_emb = [EmbeddingModelType.OPENAI, EmbeddingModelType.GEMINI, EmbeddingModelType.MISTRAL]
            perm_rerank = list(RerankerModelType)
            perm_llm = [LLMModelType.CLAUDE_37_SONNET, LLMModelType.GEMINI]
            potential_missing = set()
            for e_perm, r_perm, l_perm in itertools.product(perm_emb, perm_rerank, perm_llm):
                potential_missing.update(check_api_keys(e_perm, VectorStoreType.FAISS, r_perm, l_perm))
            if potential_missing: st.warning(f"Heads up! Missing potential keys: {', '.join(potential_missing)}. Some permutations might fail.")
            else: st.success("Looks like all potentially required API keys are present!")

            try: chunking_strategy_enum = ChunkingStrategyType.from_string(st.session_state.chunking_strategy)
            except ValueError as e: st.error(f"Invalid chunking strategy: {e}"); st.stop()

            # Create a unique filename for the CSV output
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_csv_filename = f"permutation_results_{timestamp}.csv"
            st.info(f"Results will be saved to {output_csv_filename}")

            with st.spinner("Running all permutations... This might take a while! ☕️"):
                 results_df, all_results = run_all_permutations(
                    file_path=st.session_state.file_path, user_query=user_query, ground_truth=ground_truth,
                    chunk_size=st.session_state.chunk_size, chunk_overlap=st.session_state.chunk_overlap,
                    top_k=st.session_state.top_k, hybrid_alpha=st.session_state.hybrid_alpha,
                    chunking_strategy_enum=chunking_strategy_enum,
                    output_csv_file=output_csv_filename
                )
            st.session_state.permutation_df = results_df
            st.session_state.permutation_results = all_results
            logging.info("Permutations completed.")
            st.success("All permutations complete! Results below.")
            st.rerun() 

    elif (process_button or permutation_button) and not user_query:
        st.warning("⚠️ Please enter a question to evaluate.")

    with col2:
        if st.session_state.permutation_df is not None and not st.session_state.permutation_df.empty:
            st.subheader("Permutation Results Summary")
            st.markdown(f"Results have been saved to `{output_csv_filename}` in the application's root directory.")
            st.markdown(get_csv_download_link(st.session_state.permutation_df, f"displayed_summary_{timestamp}.csv"), unsafe_allow_html=True)

            results_to_display = st.session_state.permutation_df.copy()
            results_to_display['avg_score_numeric'] = results_to_display['avg_score'].fillna(-1)
            top_results = results_to_display.sort_values('avg_score_numeric', ascending=False).head(10)

            display_cols = ["embedding_model", "vector_store", "reranker", "llm_model", "chunking_strategy", "avg_score", "elapsed_time"]
            metric_cols_exist = sorted([col for col in results_to_display.columns if col.startswith("metric_")])
            display_cols.extend(metric_cols_exist)

            format_dict = {'avg_score': "{:.2f}", 'elapsed_time': "{:.2f}"}
            for col in metric_cols_exist: format_dict[col] = "{:.2f}"
            st.dataframe(top_results[display_cols].style.format(format_dict, na_rep="N/A"), use_container_width=True)

            st.markdown("---")
            st.subheader("Explore Individual Results")
            config_labels = []
            if st.session_state.permutation_results:
                for index, row_data in enumerate(st.session_state.permutation_results):
                    label = (f"{index}: {row_data.get('embedding_model','?')} / {row_data.get('vector_store','?')} / "
                             f"{row_data.get('reranker','?')} / {row_data.get('llm_model','?')} "
                             f"(Score: {row_data.get('avg_score', 0):.2f}, Time: {row_data.get('elapsed_time', 0):.1f}s)")
                    config_labels.append(label)

                if config_labels:
                     selected_index = st.selectbox(
                         "Select Configuration to View Details:", options=range(len(config_labels)),
                         format_func=lambda index: config_labels[index], key="permutation_select"
                     )
                     if selected_index is not None and selected_index < len(st.session_state.permutation_results):
                        selected_result = st.session_state.permutation_results[selected_index]
                        st.markdown(f"**Details for Configuration {selected_index}:**")
                        st.markdown(f"**Models:** `{selected_result.get('embedding_model','N/A')} | {selected_result.get('vector_store','N/A')} | {selected_result.get('reranker','N/A')} | {selected_result.get('llm_model','N/A')}`")
                        st.markdown(f"**Chunking:** `{selected_result.get('chunking_strategy','N/A')}`")
                        st.write(f"**Processing Time:** {selected_result.get('elapsed_time', 0):.2f} seconds")
                        with st.expander("Response", expanded=True): st.write(selected_result.get('response', 'N/A'))
                        with st.expander("Retrieved Contexts", expanded=False):
                             contexts = selected_result.get('contexts', [])
                             if contexts:
                                 for i, ctx in enumerate(contexts): st.markdown(f"**Context {i+1}:**"); st.text(ctx)
                             else: st.write("No contexts available.")

                        eval_scores = selected_result.get('evaluation_scores', {})
                        if isinstance(eval_scores, dict) and "error" not in eval_scores and eval_scores:
                            st.subheader("Evaluation Scores")
                            scores_items = list(eval_scores.items())
                            metrics_per_row_detail = 2 # Max 2 metrics per row for detailed view

                            for i_detail_loop in range(0, len(scores_items), metrics_per_row_detail):
                                row_items = scores_items[i_detail_loop : i_detail_loop + metrics_per_row_detail]
                                cols_detail = st.columns(len(row_items)) # Will be 1 or 2 columns
                                for idx_detail, (metric, score) in enumerate(row_items):
                                    with cols_detail[idx_detail]:
                                        if metric == "llm_cost": # Check if the metric is "llm_cost"
                                            score_display = f"${score:.4f}" if isinstance(score, (int, float)) else "N/A"
                                        else:
                                            score_display = f"{score:.2f}" if isinstance(score, (int, float)) else "N/A"
                                        st.metric(label=metric.replace('_', ' ').title(), value=score_display)
                        elif isinstance(eval_scores, dict) and "error" in eval_scores: st.warning(f"Eval failed: {eval_scores['error']}")
                        elif ground_truth: st.warning("Scores not available (failed/skipped?).")
                        else: st.info("Provide ground truth during permutation run for scores.")
                else: st.info("No permutation results available.")
        elif st.session_state.permutation_results is not None:
             st.info("Permutation run finished, but no valid results generated.") 