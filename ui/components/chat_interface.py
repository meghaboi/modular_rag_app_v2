import streamlit as st
import logging
import time
from .shared import (display_message_with_audio)
from utils.utils import text_to_speech, is_greeting
from pipeline.subject_handler import update_rag_configuration

def display_chat_interface():
    """Display the main chat interface for interacting with JEFF."""
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

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            display_message_with_audio(message, st.session_state.show_contexts)

    user_query = st.chat_input("Type your question here...")

    if user_query:
        handle_user_query(user_query)

def handle_user_query(user_query: str):
    """Handle the user's query and generate a response."""
    logging.info(f"User query received: {user_query}")
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    if st.session_state.pipeline is None:
        handle_missing_pipeline()
        return

    if st.session_state.pipeline:
        handle_pipeline_query(user_query)

def handle_missing_pipeline():
    """Handle the case when pipeline is not initialized."""
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

def handle_pipeline_query(user_query: str):
    """Handle query processing with an initialized pipeline."""
    logging.info(f"Attempting to update RAG configuration for query: {user_query[:100]}...")
    current_subject = st.session_state.get('current_subject', None)
    config_update_status = update_rag_configuration(
        query=user_query, 
        pipeline=st.session_state.pipeline,
        subject=current_subject 
    )
    
    if config_update_status is False:
        st.error("⚠️ Error updating RAG configuration based on your query. Using previous settings. Check logs for details.")
        logging.error("update_rag_configuration returned False, indicating a failure during re-initialization.")
    elif config_update_status is True:
        st.toast("✨ Smartly adjusted RAG settings for your query!")
        logging.info("update_rag_configuration returned True. Pipeline re-initialized.")
    else:
        logging.info("update_rag_configuration returned None. No changes to RAG configuration were necessary.")

    is_greet, greeting_response = is_greeting(user_query)
    if is_greet:
        handle_greeting(greeting_response)
        return

    process_query_with_pipeline(user_query)

def handle_greeting(greeting_response: str):
    """Handle greeting responses."""
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

def process_query_with_pipeline(user_query: str):
    """Process the query using the pipeline and generate a response."""
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
            print(e)