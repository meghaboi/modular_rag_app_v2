import streamlit as st
import logging
import time
from .shared import display_message_with_audio
from utils.text_processing.text_utils import text_to_speech
from utils.analysis.analysis_utils import is_greeting
from pipeline.nature_handling import update_rag_configuration
from prompts import get_provider


class ChatInterface:
    """A class to handle the chat interface for interacting with JEFF."""

    def __init__(self):
        """Initialize the ChatInterface."""
        ui_provider = get_provider('ui')
        self.welcome_message = ui_provider.get_prompt('welcome')
        self.warning_message = ui_provider.get_prompt('warning')

    def display(self):
        """Display the main chat interface for interacting with JEFF."""
        st.header("💬 Chat with JEFF")
        st.markdown("Hey! Got questions about your textbook? Lay 'em on me. I'll break it down for ya.")

        self._initialize_welcome_message()
        self._display_message_history()

        user_query = st.chat_input("Type your question here...")
        if user_query:
            self.handle_user_query(user_query)

    def _initialize_welcome_message(self):
        """Initialize welcome message if no messages exist."""
        if not st.session_state.messages:
            welcome_audio_bytes = text_to_speech(self.welcome_message)
            st.session_state.messages.append({
                "role": "assistant",
                "content": self.welcome_message,
                "audio": welcome_audio_bytes,
                "contexts": [],
                "elapsed_time": None
            })

    def _display_message_history(self):
        """Display all messages in the chat history."""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                display_message_with_audio(message, st.session_state.show_contexts)

    def handle_user_query(self, user_query: str):
        """Handle the user's query and generate a response."""
        logging.info(f"User query received: {user_query}")

        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Display user message
        with st.chat_message("user"):
            st.write(user_query)

        # Check if pipeline is initialized
        if st.session_state.pipeline is None:
            self._handle_missing_pipeline()
            return

        if st.session_state.pipeline:
            self._handle_pipeline_query(user_query)

    def _handle_missing_pipeline(self):
        """Handle the case when pipeline is not initialized."""
        warning_audio = text_to_speech(self.warning_message)

        with st.chat_message("assistant"):
            tab_labels_warn = ["📖 Read Message", "🔊 Hear Message"]
            tab_warn_text, tab_warn_audio = st.tabs(tab_labels_warn)

            with tab_warn_text:
                st.warning(self.warning_message, icon="✋")

            with tab_warn_audio:
                if warning_audio:
                    st.audio(warning_audio, format="audio/mp3")
                else:
                    st.info("Audio playback not available.")

        st.session_state.messages.append({
            "role": "assistant",
            "content": self.warning_message,
            "audio": warning_audio,
            "contexts": [],
            "elapsed_time": None
        })
        st.stop()

    def _handle_pipeline_query(self, user_query: str):
        """Handle query processing with an initialized pipeline."""
        logging.info(f"Attempting to update RAG configuration for query: {user_query[:100]}...")

        # Update RAG configuration
        current_subject = st.session_state.get('current_subject', None)
        config_update_status = update_rag_configuration(
            query=user_query,
            pipeline=st.session_state.pipeline,
            subject=current_subject
        )

        self._handle_config_update_status(config_update_status)

        # Check if it's a greeting
        is_greet, greeting_response = is_greeting(user_query)
        if is_greet:
            self._handle_greeting(greeting_response)
            return

        self._process_query_with_pipeline(user_query)

    def _handle_config_update_status(self, config_update_status):
        """Handle the status of RAG configuration update."""
        if config_update_status is False:
            st.error(
                "⚠️ Error updating RAG configuration based on your query. Using previous settings. Check logs for details.")
            logging.error("update_rag_configuration returned False, indicating a failure during re-initialization.")
        elif config_update_status is True:
            st.toast("✨ Smartly adjusted RAG settings for your query!")
            logging.info("update_rag_configuration returned True. Pipeline re-initialized.")
        else:
            logging.info("update_rag_configuration returned None. No changes to RAG configuration were necessary.")

    def _handle_greeting(self, greeting_response: str):
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

    def _process_query_with_pipeline(self, user_query: str):
        """Process the query using the pipeline and generate a response."""
        with st.chat_message("assistant"):
            start_time = time.time()

            try:
                logging.info("Fetching contexts from vector store...")
                contexts = st.session_state.pipeline.retrieve_context(user_query)

                # Set up tabs for streaming response
                tab_labels_stream = ["📖 Read Response", "🔊 Hear Response"]
                tab_stream_text, tab_stream_audio = st.tabs(tab_labels_stream)

                with tab_stream_text:
                    stream_placeholder = st.empty()

                with tab_stream_audio:
                    audio_placeholder = st.empty()
                    audio_placeholder.info("Audio will be available when response is complete.")

                # Stream the response
                full_response = self._stream_response(user_query, stream_placeholder)

                # Finalize response display
                elapsed_time = self._finalize_response_display(
                    full_response, stream_placeholder, audio_placeholder,
                    tab_stream_text, tab_stream_audio, start_time
                )

                # Display contexts if enabled
                self._display_contexts(contexts)

                # Save message to session state
                self._save_message_to_session(full_response, contexts, elapsed_time)

            except Exception as e:
                logging.error(f"Error processing query: {e}")
                st.error("Sorry, I encountered an error processing your query. Please try again.")

    def _stream_response(self, user_query: str, stream_placeholder):
        """Stream the response from the pipeline."""
        logging.info("Starting streaming generation...")
        full_response = ""

        for chunk in st.session_state.pipeline.stream_run(user_query):
            if chunk is not None:
                full_response += chunk
                stream_placeholder.markdown(full_response + "▌")
            else:
                logging.warning("Received None chunk from stream_run, skipping")

        return full_response

    def _finalize_response_display(self, full_response, stream_placeholder, audio_placeholder,
                                   tab_stream_text, tab_stream_audio, start_time):
        """Finalize the response display with audio and timing."""
        with tab_stream_text:
            stream_placeholder.markdown(full_response)

        elapsed_time = time.time() - start_time
        st.write(f"_(JEFF cooked that up in {elapsed_time:.2f} seconds)_")

        # Generate TTS audio
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

        return elapsed_time

    def _display_contexts(self, contexts):
        """Display contexts if show_contexts is enabled."""
        if st.session_state.show_contexts and contexts:
            with st.expander("🧠 Check out the textbook bits I used:"):
                for i, context in enumerate(contexts):
                    st.markdown(f"**Snippet {i + 1}:**")
                    st.text(context)

    def _save_message_to_session(self, full_response, contexts, elapsed_time):
        """Save the assistant's message to session state."""
        audio_bytes = text_to_speech(full_response)
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "contexts": contexts,
            "elapsed_time": elapsed_time,
            "audio": audio_bytes
        })


# Factory function to maintain compatibility with existing code
def display_chat_interface():
    """Factory function to create and display chat interface."""
    chat_interface = ChatInterface()
    chat_interface.display()


# For the __init__.py file
__all__ = ['ChatInterface', 'display_chat_interface']