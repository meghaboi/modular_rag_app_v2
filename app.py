import streamlit as st
import logging
from ui_components import display_chat_interface, display_evaluation_interface
from sidebar import display_settings_panel
# Removed: from utils import check_api_keys # No longer directly used here, handled by individual components
from pipeline_utils import initialize_pipeline
import subject_configs # Changed import to use the module directly
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'file_path' not in st.session_state:
    st.session_state.file_path = None
if 'last_uploaded_filename' not in st.session_state:
    st.session_state.last_uploaded_filename = None
if 'show_contexts' not in st.session_state:
    st.session_state.show_contexts = False
if 'mode' not in st.session_state:
    st.session_state.mode = 'chat'
if 'permutation_results' not in st.session_state:
    st.session_state.permutation_results = None
if 'permutation_df' not in st.session_state:
    st.session_state.permutation_df = None
if 'api_key_status' not in st.session_state:
    st.session_state.api_key_status = None

# Set default configuration values
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = subject_configs.DEFAULT_EMBEDDING_MODEL.value
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = subject_configs.DEFAULT_VECTOR_STORE.value
if 'reranker' not in st.session_state:
    st.session_state.reranker = subject_configs.DEFAULT_RERANKER_MODEL.value
if 'llm_model' not in st.session_state:
    st.session_state.llm_model = subject_configs.DEFAULT_LLM_MODEL.value
if 'chunking_strategy' not in st.session_state:
    st.session_state.chunking_strategy = subject_configs.DEFAULT_CHUNKING_STRATEGY.value
if 'hybrid_alpha' not in st.session_state:
    st.session_state.hybrid_alpha = subject_configs.DEFAULT_HYBRID_ALPHA
if 'chunk_size' not in st.session_state:
    st.session_state.chunk_size = subject_configs.DEFAULT_CHUNK_SIZE
if 'chunk_overlap' not in st.session_state:
    st.session_state.chunk_overlap = subject_configs.DEFAULT_CHUNK_OVERLAP
if 'top_k' not in st.session_state:
    st.session_state.top_k = subject_configs.DEFAULT_TOP_K

def main():
    # Display settings panel in sidebar
    display_settings_panel()

    # Main content area
    if st.session_state.mode == "chat":
        display_chat_interface()
    else:
        display_evaluation_interface()

if __name__ == "__main__":
    main() 