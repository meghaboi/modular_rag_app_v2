import streamlit as st
import logging
from ui.ui_components import display_chat_interface, display_evaluation_interface
from ui.sidebar import display_settings_panel
from utils.subject_configs import (
    DEFAULT_EMBEDDING_MODEL, DEFAULT_VECTOR_STORE, DEFAULT_RERANKER_MODEL,
    DEFAULT_LLM_MODEL, DEFAULT_CHUNKING_STRATEGY, DEFAULT_HYBRID_ALPHA,
    DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_TOP_K
)
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

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

if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = DEFAULT_EMBEDDING_MODEL.value
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = DEFAULT_VECTOR_STORE.value
if 'reranker' not in st.session_state:
    st.session_state.reranker = DEFAULT_RERANKER_MODEL.value
if 'llm_model' not in st.session_state:
    st.session_state.llm_model = DEFAULT_LLM_MODEL.value
if 'chunking_strategy' not in st.session_state:
    st.session_state.chunking_strategy = DEFAULT_CHUNKING_STRATEGY.value
if 'config_changed' not in st.session_state:
    st.session_state.config_changed = False
if 'hybrid_alpha' not in st.session_state:
    st.session_state.hybrid_alpha = DEFAULT_HYBRID_ALPHA
if 'chunk_size' not in st.session_state:
    st.session_state.chunk_size = DEFAULT_CHUNK_SIZE
if 'chunk_overlap' not in st.session_state:
    st.session_state.chunk_overlap = DEFAULT_CHUNK_OVERLAP
if 'top_k' not in st.session_state:
    st.session_state.top_k = DEFAULT_TOP_K

def main():
    display_settings_panel()

    if st.session_state.mode == "chat":
        display_chat_interface()
    else:
        display_evaluation_interface()

if __name__ == "__main__":
    main() 