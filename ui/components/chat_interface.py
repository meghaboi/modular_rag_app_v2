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

    # Constants
    MAX_QUERY_LOG_LENGTH = 100
    AUDIO_FORMAT = "audio/mp3"
    STREAMING_CURSOR = "▌"
    READ_TAB_ICON = "📖"
    AUDIO_TAB_ICON = "🔊"
    CONTEXTS_ICON = "🧠"
    
    def __init__(self):
        """Initialize the ChatInterface."""
        ui_provider = get_provider('ui')
        self.welcome_message = ui_provider.get_prompt('welcome')
        self.warning_message = ui_provider.get_prompt('warning')
        self.tab_creator = TabCreator()
        self.message_handler = MessageHandler()
        self.response_processor = ResponseProcessor()

    def display(self):
        """Display the main chat interface for interacting with JEFF."""
        self._render_header()
        self._initialize_welcome_message()
        self._display_message_history()
        self._handle_user_input()

    def handle_user_query(self, user_query: str):
        """Handle the user's query and generate a response."""
        logging.info(f"User query received: {user_query}")
        
        self._add_user_message(user_query)
        self._display_user_message(user_query)
        
        if self._is_pipeline_missing():
            self._handle_missing_pipeline()
            return
            
        self._process_user_query(user_query)

    def _render_header(self):
        """Render the chat interface header."""
        st.header("💬 Chat with JEFF")
        st.markdown("Hey! Got questions about your textbook? Lay 'em on me. I'll break it down for ya.")

    def _initialize_welcome_message(self):
        """Initialize welcome message if no messages exist."""
        if self._should_show_welcome():
            welcome_audio = text_to_speech(self.welcome_message)
            welcome_message = self._create_message(
                role="assistant",
                content=self.welcome_message,
                audio=welcome_audio
            )
            st.session_state.messages.append(welcome_message)

    def _display_message_history(self):
        """Display all messages in the chat history."""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                display_message_with_audio(message, st.session_state.show_contexts)

    def _handle_user_input(self):
        """Handle user input from chat interface."""
        user_query = st.chat_input("Type your question here...")
        if user_query:
            self.handle_user_query(user_query)

    def _add_user_message(self, user_query: str):
        """Add user message to session state."""
        st.session_state.messages.append({"role": "user", "content": user_query})

    def _display_user_message(self, user_query: str):
        """Display user message in chat."""
        with st.chat_message("user"):
            st.write(user_query)

    def _is_pipeline_missing(self):
        """Check if pipeline is not initialized."""
        return st.session_state.pipeline is None

    def _should_show_welcome(self):
        """Check if welcome message should be displayed."""
        return not st.session_state.messages

    def _process_user_query(self, user_query: str):
        """Process user query with initialized pipeline."""
        if self._is_greeting(user_query):
            return
            
        self._update_rag_configuration(user_query)
        self._generate_response(user_query)

    def _update_rag_configuration(self, user_query: str):
        """Update RAG configuration based on user query."""
        logging.info(f"Updating RAG configuration for query: {user_query[:self.MAX_QUERY_LOG_LENGTH]}...")
        
        config_status = update_rag_configuration(query=user_query)
        self._handle_config_status(config_status)

    def _is_greeting(self, user_query: str):
        """Check if query is a greeting and handle if so."""
        is_greet, greeting_response = is_greeting(user_query)
        if is_greet:
            self._handle_greeting(greeting_response)
            return True
        return False

    def _generate_response(self, user_query: str):
        """Generate and display response for user query."""
        try:
            with st.chat_message("assistant"):
                response_data = self.response_processor.process_query(user_query)
                self._save_assistant_message(response_data)
        except Exception as e:
            self._handle_query_error(e)

    def _handle_missing_pipeline(self):
        """Handle the case when pipeline is not initialized."""
        warning_audio = text_to_speech(self.warning_message)
        
        with st.chat_message("assistant"):
            self.tab_creator.create_warning_tabs(self.warning_message, warning_audio)

        warning_message = self._create_message(
            role="assistant",
            content=self.warning_message,
            audio=warning_audio
        )
        st.session_state.messages.append(warning_message)
        st.stop()

    def _handle_config_status(self, config_status):
        """Handle RAG configuration update status."""
        if config_status is False:
            st.error("⚠️ Error updating RAG configuration. Using previous settings.")
            logging.error("RAG configuration update failed.")
        elif config_status is True:
            st.toast("✨ Smartly adjusted RAG settings for your query!")
            logging.info("RAG configuration updated successfully.")
        else:
            logging.info("No RAG configuration changes needed.")

    def _handle_greeting(self, greeting_response: str):
        """Handle greeting responses."""
        greeting_audio = text_to_speech(greeting_response)
        
        with st.chat_message("assistant"):
            self.tab_creator.create_response_tabs(greeting_response, greeting_audio)

        greeting_message = self._create_message(
            role="assistant",
            content=greeting_response,
            audio=greeting_audio
        )
        st.session_state.messages.append(greeting_message)

    def _handle_query_error(self, error: Exception):
        """Handle errors that occur during query processing."""
        logging.error(f"Error processing query: {error}")
        st.error("Sorry, I encountered an error processing your query. Please try again.")

    def _create_message(self, role: str, content: str, audio=None, contexts=None, elapsed_time=None):
        """Create a message dictionary with consistent structure."""
        return {
            "role": role,
            "content": content,
            "audio": audio,
            "contexts": contexts or [],
            "elapsed_time": elapsed_time
        }

    def _save_assistant_message(self, response_data: dict):
        """Save assistant message to session state."""
        message = self._create_message(
            role="assistant",
            content=response_data["content"],
            audio=response_data["audio"],
            contexts=response_data["contexts"],
            elapsed_time=response_data["elapsed_time"]
        )
        st.session_state.messages.append(message)

class TabCreator:
    """Handles creation of UI tabs for different message types."""
    
    def create_response_tabs(self, content: str, audio_bytes):
        """Create tabs for response display."""
        tab_labels = [f"{ChatInterface.READ_TAB_ICON} Read Response", 
                     f"{ChatInterface.AUDIO_TAB_ICON} Hear Response"]
        tab_text, tab_audio = st.tabs(tab_labels)
        
        with tab_text:
            st.write(content)
            
        with tab_audio:
            self._display_audio_or_fallback(audio_bytes)

    def create_warning_tabs(self, warning_message: str, warning_audio):
        """Create tabs for warning display."""
        tab_labels = [f"{ChatInterface.READ_TAB_ICON} Read Message", 
                     f"{ChatInterface.AUDIO_TAB_ICON} Hear Message"]
        tab_text, tab_audio = st.tabs(tab_labels)
        
        with tab_text:
            st.warning(warning_message, icon="✋")
            
        with tab_audio:
            self._display_audio_or_fallback(warning_audio)

    def create_streaming_tabs(self):
        """Create tabs for streaming response."""
        tab_labels = [f"{ChatInterface.READ_TAB_ICON} Read Response", 
                     f"{ChatInterface.AUDIO_TAB_ICON} Hear Response"]
        return st.tabs(tab_labels)

    def _display_audio_or_fallback(self, audio_bytes):
        """Display audio if available, otherwise show fallback message."""
        if audio_bytes:
            st.audio(audio_bytes, format=ChatInterface.AUDIO_FORMAT)
        else:
            st.info("Audio playback not available.")

class MessageHandler:
    """Handles message processing and storage."""
    
    def create_message(self, role: str, content: str, **kwargs):
        """Create a standardized message structure."""
        return {
            "role": role,
            "content": content,
            "audio": kwargs.get("audio"),
            "contexts": kwargs.get("contexts", []),
            "elapsed_time": kwargs.get("elapsed_time")
        }

class ResponseProcessor:
    """Handles response generation and processing."""
    
    def __init__(self):
        self.tab_creator = TabCreator()
    
    def process_query(self, user_query: str):
        """Process query and return response data."""
        start_time = time.time()
        
        contexts = self._retrieve_contexts(user_query)
        tab_text, tab_audio = self.tab_creator.create_streaming_tabs()
        
        response_content = self._generate_streaming_response(user_query, tab_text)
        elapsed_time = self._finalize_response(response_content, tab_text, tab_audio, start_time)
        
        self._display_contexts_if_enabled(contexts)
        
        return {
            "content": response_content,
            "audio": text_to_speech(response_content),
            "contexts": contexts,
            "elapsed_time": elapsed_time
        }

    def _retrieve_contexts(self, user_query: str):
        """Retrieve contexts from vector store."""
        logging.info("Fetching contexts from vector store...")
        return st.session_state.pipeline.retrieve_context(user_query)

    def _generate_streaming_response(self, user_query: str, tab_text):
        """Generate streaming response from pipeline."""
        with tab_text:
            stream_placeholder = st.empty()
            
        logging.info("Starting streaming generation...")
        full_response = ""
        
        for chunk in st.session_state.pipeline.stream_run(user_query):
            if chunk is not None:
                full_response += chunk
                stream_placeholder.markdown(full_response + ChatInterface.STREAMING_CURSOR)
            else:
                logging.warning("Received None chunk from stream_run, skipping")
        
        stream_placeholder.markdown(full_response)
                
        return full_response

    def _finalize_response(self, response_content: str, tab_text, tab_audio, start_time: float):
        """Finalize response display with audio and timing."""
        elapsed_time = time.time() - start_time
        
        with tab_text:
            st.write(f"_(JEFF cooked that up in {elapsed_time:.2f} seconds)_")
        
        self._generate_and_display_audio(response_content, tab_audio)
        
        return elapsed_time

    def _generate_and_display_audio(self, content: str, tab_audio):
        """Generate TTS audio and display in tab."""
        logging.info("Generating TTS audio for the complete response...")
        tts_start_time = time.time()
        audio_bytes = text_to_speech(content)
        tts_elapsed_time = time.time() - tts_start_time
        
        success_msg = f"TTS generation {'succeeded' if audio_bytes else 'failed/skipped'} in {tts_elapsed_time:.2f}s."
        
        with tab_audio:
            if audio_bytes:
                logging.info(success_msg)
                st.audio(audio_bytes, format=ChatInterface.AUDIO_FORMAT)
            else:
                logging.warning(success_msg)
                st.info("Audio playback is not available for this message.")

    def _display_contexts_if_enabled(self, contexts):
        """Display contexts if show_contexts is enabled."""
        if st.session_state.show_contexts and contexts:
            with st.expander(f"{ChatInterface.CONTEXTS_ICON} Check out the textbook bits I used:"):
                for i, context in enumerate(contexts):
                    st.markdown(f"**Snippet {i + 1}:**")
                    st.text(context)

def display_chat_interface():
    """Factory function to create and display chat interface."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    chat_interface = ChatInterface()
    chat_interface.display()

__all__ = ['ChatInterface', 'display_chat_interface']