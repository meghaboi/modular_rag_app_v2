import streamlit as st
import logging
from typing import List, Dict, Any

# Constants
AUDIO_FORMAT = "audio/mp3"
TAB_LABELS = ["📖 Read Response", "🔊 Hear Response"]
MAX_FUNCTION_LINES = 20

def display_message_with_audio(message: Dict[str, Any], show_contexts: bool = False):
    """Helper function to display a message with audio and contexts."""
    if _is_assistant_message(message):
        _display_assistant_message(message, show_contexts)
    else:
        _display_user_message(message)

def _is_assistant_message(message_dict: Dict[str, Any]) -> bool:
    """Check if message is from assistant."""
    return message_dict["role"] == "assistant"

def _display_user_message(message_dict: Dict[str, Any]) -> None:
    """Display a user message."""
    st.write(message_dict["content"])

def _display_assistant_message(message_dict: Dict[str, Any], show_contexts: bool) -> None:
    """Display an assistant message with audio and context options."""
    message_content = _extract_message_content(message_dict)
    
    if not message_content.response_text:
        st.write("*Assistant message content missing.*")
        return
    
    _display_response_tabs(message_content)
    _display_elapsed_time(message_content.elapsed_time)
    _display_contexts_if_requested(message_content.contexts, show_contexts)

def _extract_message_content(message_dict: Dict[str, Any]) -> 'MessageContent':
    """Extract and organize message content into a structured object."""
    return MessageContent(
        response_text=message_dict.get("content"),
        audio_data=message_dict.get("audio"),
        contexts=message_dict.get("contexts", []),
        elapsed_time=message_dict.get("elapsed_time")
    )

def _display_response_tabs(content: 'MessageContent') -> None:
    """Display response in tabbed format with text and audio options."""
    try:
        tab_text, tab_audio = st.tabs(TAB_LABELS)
        _display_text_tab(tab_text, content.response_text)
        _display_audio_tab(tab_audio, content.audio_data)
    except Exception as error:
        _handle_tab_creation_error(error, content)

def _display_text_tab(tab_text, response_text: str) -> None:
    """Display response text in the text tab."""
    with tab_text:
        st.write(response_text)

def _display_audio_tab(tab_audio, audio_data) -> None:
    """Display audio player in the audio tab."""
    with tab_audio:
        if audio_data:
            st.audio(audio_data, format=AUDIO_FORMAT)
        else:
            st.info("Audio playback is not available for this message.")

def _handle_tab_creation_error(error: Exception, content: 'MessageContent') -> None:
    """Handle error when tab creation fails by falling back to simple display."""
    logging.error(f"Error creating tabs: {error}")
    st.write(content.response_text)
    if content.audio_data:
        st.audio(content.audio_data, format=AUDIO_FORMAT)

def _display_elapsed_time(elapsed_time_seconds) -> None:
    """Display processing time if available."""
    if elapsed_time_seconds is not None:
        st.write(f"_(JEFF cooked that up in {elapsed_time_seconds:.2f} seconds)_")

def _display_contexts_if_requested(contexts: List[str], should_show_contexts: bool) -> None:
    """Display context snippets in an expandable section if requested."""
    if should_show_contexts and contexts:
        with st.expander("🧠 Check out the textbook bits I used:"):
            _display_context_snippets(contexts)

def _display_context_snippets(contexts: List[str]) -> None:
    """Display numbered context snippets."""
    for snippet_index, context_text in enumerate(contexts):
        snippet_number = snippet_index + 1
        st.markdown(f"**Snippet {snippet_number}:**")
        st.text(context_text)

class MessageContent:
    """Data structure for organizing message content."""
    
    def __init__(self, response_text: str, audio_data, contexts: List[str], elapsed_time):
        self.response_text = response_text
        self.audio_data = audio_data
        self.contexts = contexts
        self.elapsed_time = elapsed_time