import streamlit as st
import logging
import time
import itertools
from typing import List, Dict, Any
from utils import text_to_speech, is_greeting, check_api_keys, get_csv_download_link
from ai_configurator import get_ai_suggested_config, DEFAULT_SMARTER_JEFF_CONFIG
from pipeline_utils import initialize_pipeline # To trigger re-initialization
from enums import (
    EmbeddingModelType,
    RerankerModelType,
    LLMModelType,
    VectorStoreType,
    ChunkingStrategyType
)
from pipeline_utils import run_all_permutations
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

    # --- BEGIN SMARTER-JEFF INTEGRATION ---
    if st.session_state.get('smarter_jeff_enabled', False):
        logging.info("Smarter-Jeff mode is ON. Attempting to get AI suggested config.")
        
        # Gather current RAG config from session_state to pass to AI
        current_rag_config = {
            "embedding_model": st.session_state.get('embedding_model', DEFAULT_SMARTER_JEFF_CONFIG['embedding_model']),
            "vector_store": st.session_state.get('vector_store', DEFAULT_SMARTER_JEFF_CONFIG['vector_store']),
            "reranker": st.session_state.get('reranker', DEFAULT_SMARTER_JEFF_CONFIG['reranker']),
            "llm_model": st.session_state.get('llm_model', DEFAULT_SMARTER_JEFF_CONFIG['llm_model']),
            "chunking_strategy": st.session_state.get('chunking_strategy', DEFAULT_SMARTER_JEFF_CONFIG['chunking_strategy']),
            "hybrid_alpha": float(st.session_state.get('hybrid_alpha', DEFAULT_SMARTER_JEFF_CONFIG['hybrid_alpha'])),
            "chunk_size": int(st.session_state.get('chunk_size', DEFAULT_SMARTER_JEFF_CONFIG['chunk_size'])),
            "chunk_overlap": int(st.session_state.get('chunk_overlap', DEFAULT_SMARTER_JEFF_CONFIG['chunk_overlap'])),
            "top_k": int(st.session_state.get('top_k', DEFAULT_SMARTER_JEFF_CONFIG['top_k']))
        }

        ai_suggested_config = get_ai_suggested_config(user_query, current_rag_config)
        st.session_state.ai_suggested_config = ai_suggested_config # Store for potential use in sidebar initialization

        # Define parameters that trigger re-indexing if changed
        reindex_params = ["embedding_model", "vector_store", "chunking_strategy", "chunk_size", "chunk_overlap"]
        
        requires_reinitialization = False
        if st.session_state.pipeline is None: # If no pipeline, AI config means we need to init
            requires_reinitialization = True
        else:
            for param in reindex_params:
                if str(current_rag_config.get(param)) != str(ai_suggested_config.get(param)): # Compare as strings for enums
                    requires_reinitialization = True
                    logging.info(f"Smarter-Jeff: Change in '{param}' (from '{current_rag_config.get(param)}' to '{ai_suggested_config.get(param)}') requires re-initialization.")
                    break
            if not requires_reinitialization: # Check non-reindexing params
                non_reindex_params = ["reranker", "llm_model", "top_k", "hybrid_alpha"]
                for param in non_reindex_params:
                    if str(current_rag_config.get(param)) != str(ai_suggested_config.get(param)):
                         requires_reinitialization = True
                         logging.info(f"Smarter-Jeff: Change in '{param}' (from '{current_rag_config.get(param)}' to '{ai_suggested_config.get(param)}') will also trigger re-initialization for now.")
                         break

        if requires_reinitialization:
            logging.info("Smarter-Jeff: AI suggested changes require pipeline re-initialization.")
            
            st.session_state.embedding_model = ai_suggested_config["embedding_model"]
            st.session_state.vector_store = ai_suggested_config["vector_store"]
            st.session_state.reranker = ai_suggested_config["reranker"]
            st.session_state.llm_model = ai_suggested_config["llm_model"]
            st.session_state.chunking_strategy = ai_suggested_config["chunking_strategy"]
            st.session_state.hybrid_alpha = float(ai_suggested_config["hybrid_alpha"])
            st.session_state.chunk_size = int(ai_suggested_config["chunk_size"])
            st.session_state.chunk_overlap = int(ai_suggested_config["chunk_overlap"])
            st.session_state.top_k = int(ai_suggested_config["top_k"])

            if not st.session_state.get('file_path'):
                warning_msg = "Smarter-Jeff wants to reconfigure, but no textbook is loaded. Please upload a textbook first."
                # Display warning similar to existing pipeline warning (simplified here)
                st.warning(warning_msg) 
                # This will be caught by the subsequent "if st.session_state.pipeline is None:" check if it remains None
            else:
                with st.spinner("Smarter-Jeff is optimizing settings for your query. JEFF is re-initializing..."):
                    try:
                        embedding_enum = EmbeddingModelType.from_string(st.session_state.embedding_model)
                        vs_enum = VectorStoreType.from_string(st.session_state.vector_store)
                        reranker_enum = RerankerModelType.from_string(st.session_state.reranker)
                        llm_enum = LLMModelType.from_string(st.session_state.llm_model)
                        cs_enum = ChunkingStrategyType.from_string(st.session_state.chunking_strategy)

                        pipeline_instance = initialize_pipeline(
                            file_path=st.session_state.file_path,
                            embedding_model_enum=embedding_enum,
                            vector_store_enum=vs_enum,
                            reranker_enum=reranker_enum,
                            llm_enum=llm_enum,
                            chunking_strategy_enum=cs_enum,
                            hybrid_alpha=st.session_state.hybrid_alpha,
                            chunk_size=st.session_state.chunk_size,
                            chunk_overlap=st.session_state.chunk_overlap,
                            top_k=st.session_state.top_k
                        )
                        if pipeline_instance:
                            st.session_state.pipeline = pipeline_instance
                            logging.info("Smarter-Jeff: Pipeline re-initialized successfully with AI config.")
                            # Show success message in the chat area, not sidebar
                            st.chat_message("assistant").success("JEFF has been reconfigured by Smarter-Jeff for your query!")
                        else:
                            st.session_state.pipeline = None
                            logging.error("Smarter-Jeff: Pipeline re-initialization failed.")
                            st.chat_message("assistant").error("Smarter-Jeff failed to reconfigure the pipeline. Using previous settings if available.")
                    except Exception as e:
                        logging.error(f"Smarter-Jeff: Error during re-initialization: {e}", exc_info=True)
                        st.chat_message("assistant").error(f"Smarter-Jeff: Error reconfiguring: {e}")
        else:
            logging.info("Smarter-Jeff: AI suggested config does not require re-initialization or no changes suggested.")

    # --- END SMARTER-JEFF INTEGRATION ---

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
                            summary_metric_cols = st.columns(3)
                            with summary_metric_cols[0]:
                                st.metric(label="Processing Time", value=f"{eval_elapsed_time:.2f}s")
                            if metrics:
                                with summary_metric_cols[1]:
                                    st.metric(label="Input Tokens", value=metrics.get('input_tokens', "N/A"))
                                with summary_metric_cols[2]:
                                    st.metric(label="Output Tokens", value=metrics.get('output_tokens', "N/A"))

                            with st.expander("Response", expanded=True): st.write(response)
                            with st.expander("Retrieved Contexts", expanded=False):
                                 if contexts:
                                    for i, ctx in enumerate(contexts): st.markdown(f"**Context {i+1}:**"); st.text(ctx)
                                 else: st.write("No contexts were retrieved.")

                            if evaluation_results and isinstance(evaluation_results, dict) and "error" not in evaluation_results:
                                 st.subheader("Evaluation Scores")
                                 if evaluation_results:
                                     metrics_list = list(evaluation_results.items())
                                     num_metrics = len(metrics_list)
                                     metrics_per_row = 3 # Max 3 metrics per row

                                     for i_loop_var in range(0, num_metrics, metrics_per_row):
                                         current_row_metrics_data = metrics_list[i_loop_var : i_loop_var + metrics_per_row]
                                         # Create columns for the current row of metrics
                                         score_display_cols = st.columns(len(current_row_metrics_data))
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
                            metric_cols_detail = st.columns(len(eval_scores))
                            i = 0; valid_scores_detail = []
                            for metric, score in eval_scores.items():
                                 with metric_cols_detail[i]:
                                     score_display = f"{score:.2f}" if isinstance(score, (int, float)) else "N/A"
                                     st.metric(label=metric.replace('_', ' ').title(), value=score_display)
                                     if isinstance(score, (int, float)): valid_scores_detail.append(score)
                                 i+=1
                        elif isinstance(eval_scores, dict) and "error" in eval_scores: st.warning(f"Eval failed: {eval_scores['error']}")
                        elif ground_truth: st.warning("Scores not available (failed/skipped?).")
                        else: st.info("Provide ground truth during permutation run for scores.")
                else: st.info("No permutation results available.")
        elif st.session_state.permutation_results is not None:
             st.info("Permutation run finished, but no valid results generated.") 