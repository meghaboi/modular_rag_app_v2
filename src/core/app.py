"""
Main application module for the ModularRAG application.
"""

import os
import logging
import streamlit as st
from dotenv import load_dotenv

# Core imports
from src.core.session_state import initialize_session_state
from src.core.auto_init import attempt_automatic_initialization

# UI imports
from src.ui import display_settings_panel, display_chat_interface, display_evaluation_interface

# Service imports
from src.services.tts_service import text_to_speech
from src.services.greeting_service import is_greeting, get_greeting_response
from src.services.file_service import save_uploaded_file, get_csv_download_link
from src.services.pipeline_service import run_pipeline_with_config, run_all_permutations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="ModularRAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application entry point."""
    # Initialize session state
    initialize_session_state()

    # Attempt automatic initialization if not already attempted
    if not st.session_state.get('auto_init_attempted'):
        attempt_automatic_initialization()
         st.session_state.auto_init_attempted = True

    # Display settings panel
    display_settings_panel()

    # Display main interface based on mode
    if st.session_state.mode == "chat":
        display_chat_interface() 
    else: 
        display_evaluation_interface()

if __name__ == "__main__":
    main()