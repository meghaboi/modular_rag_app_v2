import streamlit as st
from ..core.app import (
    DEFAULT_EMBEDDING_MODEL, DEFAULT_RERANKER_MODEL, DEFAULT_LLM_MODEL,
    DEFAULT_VECTOR_STORE, DEFAULT_CHUNKING_STRATEGY, DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP, DEFAULT_TOP_K, DEFAULT_HYBRID_ALPHA,
    check_api_keys, initialize_pipeline
)

def display_settings_panel():
    """Display the settings panel"""
    st.sidebar.title("Settings")
    
    # Model Selection
    st.sidebar.subheader("Model Selection")
    embedding_model = st.sidebar.selectbox(
        "Embedding Model",
        options=[model.value for model in EmbeddingModelType],
        index=get_safe_index([model.value for model in EmbeddingModelType], st.session_state.embedding_model)
    )
    
    vector_store = st.sidebar.selectbox(
        "Vector Store",
        options=[store.value for store in VectorStoreType],
        index=get_safe_index([store.value for store in VectorStoreType], st.session_state.vector_store)
    )
    
    reranker = st.sidebar.selectbox(
        "Reranker",
        options=[model.value for model in RerankerModelType],
        index=get_safe_index([model.value for model in RerankerModelType], st.session_state.reranker)
    )
    
    llm_model = st.sidebar.selectbox(
        "LLM Model",
        options=[model.value for model in LLMModelType],
        index=get_safe_index([model.value for model in LLMModelType], st.session_state.llm_model)
    )
    
    # Chunking Settings
    st.sidebar.subheader("Chunking Settings")
    chunking_strategy = st.sidebar.selectbox(
        "Chunking Strategy",
        options=[strategy.value for strategy in ChunkingStrategyType],
        index=get_safe_index([strategy.value for strategy in ChunkingStrategyType], st.session_state.chunking_strategy)
    )
    
    chunk_size = st.sidebar.number_input(
        "Chunk Size",
        min_value=100,
        max_value=4000,
        value=st.session_state.chunk_size
    )
    
    chunk_overlap = st.sidebar.number_input(
        "Chunk Overlap",
        min_value=0,
        max_value=1000,
        value=st.session_state.chunk_overlap
    )
    
    # Retrieval Settings
    st.sidebar.subheader("Retrieval Settings")
    top_k = st.sidebar.number_input(
        "Top K",
        min_value=1,
        max_value=10,
        value=st.session_state.top_k
    )
    
    hybrid_alpha = st.sidebar.slider(
        "Hybrid Alpha",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.hybrid_alpha,
        step=0.1
    )
    
    # Display API Key Status
    st.sidebar.subheader("API Key Status")
    missing_keys = check_api_keys(
        EmbeddingModelType(embedding_model),
        VectorStoreType(vector_store),
        RerankerModelType(reranker),
        LLMModelType(llm_model)
    )
    
    for key_name, status in st.session_state.api_key_status.items():
        st.sidebar.write(f"{key_name}: {status}")
    
    if missing_keys:
        st.sidebar.warning("Missing API keys. Please check your .env file.")
    
    # Initialize Pipeline Button
    if st.sidebar.button("Initialize Pipeline"):
        if not st.session_state.file_path:
            st.sidebar.error("Please upload a document first.")
            return
        
        with st.spinner("Initializing pipeline..."):
            pipeline = initialize_pipeline(
                st.session_state.file_path,
                EmbeddingModelType(embedding_model),
                VectorStoreType(vector_store),
                RerankerModelType(reranker),
                LLMModelType(llm_model),
                ChunkingStrategyType(chunking_strategy),
                hybrid_alpha,
                chunk_size,
                chunk_overlap,
                top_k
            )
            
            if pipeline:
                st.session_state.pipeline = pipeline
                st.sidebar.success("Pipeline initialized successfully!")
            else:
                st.sidebar.error("Failed to initialize pipeline. Please check the logs.")

def get_safe_index(options_list, current_value, default_index=0):
    """Safely get the index of a value in a list"""
    try:
        return options_list.index(current_value)
    except ValueError:
        return default_index 