import streamlit as st
from ..config.enums import (
    EmbeddingModelType, RerankerModelType, LLMModelType, VectorStoreType,
    ChunkingStrategyType
)

# Default values
DEFAULT_EMBEDDING_MODEL = EmbeddingModelType.MISTRAL
DEFAULT_RERANKER_MODEL = RerankerModelType.COHERE_V3
DEFAULT_LLM_MODEL = LLMModelType.CLAUDE_37_SONNET
DEFAULT_VECTOR_STORE = VectorStoreType.CHROMA
DEFAULT_CHUNKING_STRATEGY = ChunkingStrategyType.HIERARCHICAL
DEFAULT_CHUNK_SIZE = 2095
DEFAULT_CHUNK_OVERLAP = 195
DEFAULT_TOP_K = 4
DEFAULT_HYBRID_ALPHA = 0.5

def initialize_session_state():
    """Initialize all session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'file_path' not in st.session_state:
        st.session_state.file_path = None
    if 'last_uploaded_filename' not in st.session_state:
        st.session_state.last_uploaded_filename = None
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'mode' not in st.session_state:
        st.session_state.mode = "chat"
    if 'permutation_results' not in st.session_state:
        st.session_state.permutation_results = None
    if 'permutation_df' not in st.session_state:
        st.session_state.permutation_df = None
    if 'show_contexts' not in st.session_state:
        st.session_state.show_contexts = False
    if 'api_key_status' not in st.session_state:
        st.session_state.api_key_status = {}
    if 'auto_init_attempted' not in st.session_state:
        st.session_state.auto_init_attempted = False

    # Initialize model configurations
    if 'embedding_model' not in st.session_state:
        st.session_state.embedding_model = DEFAULT_EMBEDDING_MODEL.value
    if 'reranker' not in st.session_state:
        st.session_state.reranker = DEFAULT_RERANKER_MODEL.value
    if 'llm_model' not in st.session_state:
        st.session_state.llm_model = DEFAULT_LLM_MODEL.value
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = DEFAULT_VECTOR_STORE.value
    if 'chunking_strategy' not in st.session_state:
        st.session_state.chunking_strategy = DEFAULT_CHUNKING_STRATEGY.value
    if 'chunk_size' not in st.session_state:
        st.session_state.chunk_size = DEFAULT_CHUNK_SIZE
    if 'chunk_overlap' not in st.session_state:
        st.session_state.chunk_overlap = DEFAULT_CHUNK_OVERLAP
    if 'top_k' not in st.session_state:
        st.session_state.top_k = DEFAULT_TOP_K
    if 'hybrid_alpha' not in st.session_state:
        st.session_state.hybrid_alpha = DEFAULT_HYBRID_ALPHA 