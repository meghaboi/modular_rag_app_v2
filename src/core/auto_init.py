"""
Automatic initialization functionality for the ModularRAG application.
"""

import os
import logging
import streamlit as st
from src.config.enums import (
    EmbeddingModelType,
    RerankerModelType,
    LLMModelType,
    VectorStoreType,
    ChunkingStrategyType
)
from src.core.session_state import (
    DEFAULT_EMBEDDING_MODEL, DEFAULT_VECTOR_STORE, DEFAULT_RERANKER_MODEL,
    DEFAULT_LLM_MODEL, DEFAULT_CHUNKING_STRATEGY, DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP, DEFAULT_TOP_K, DEFAULT_HYBRID_ALPHA
)
from src.core.pipeline_init import initialize_pipeline

def attempt_automatic_initialization():
    """Tries to initialize RAG pipeline automatically on startup if possible."""
    if st.session_state.pipeline is None and st.session_state.file_path and os.path.exists(st.session_state.file_path):
        logging.info("Attempting automatic RAG pipeline initialization on startup.")
        init_placeholder = st.empty()
        init_placeholder.info("Trying auto-setup...")

        default_embedding_enum = EmbeddingModelType.MISTRAL
        default_vs_enum = VectorStoreType.CHROMA
        default_reranker_enum = RerankerModelType.COHERE_V3
        default_llm_enum = LLMModelType.CLAUDE_37_SONNET
        default_cs_enum = ChunkingStrategyType.HIERARCHICAL

        missing_keys = check_api_keys(default_embedding_enum, default_vs_enum, default_reranker_enum, default_llm_enum)
        if missing_keys:
            init_placeholder.warning(f"Auto-setup skipped: Missing keys ({', '.join(missing_keys)}). Initialize manually.", icon="🔑")
            logging.warning(f"Auto-init skipped due to missing keys: {missing_keys}")
        else:
            logging.info("Default keys found. Proceeding with auto-initialization.")
            init_placeholder.empty()
            with st.spinner("JEFF is warming up... (Auto-initializing)"):
                try:
                    pipeline_instance = initialize_pipeline(
                        file_path=st.session_state.file_path,
                        embedding_model_enum=default_embedding_enum,
                        vector_store_enum=default_vs_enum,
                        reranker_enum=default_reranker_enum,
                        llm_enum=default_llm_enum,
                        chunking_strategy_enum=default_cs_enum,
                        hybrid_alpha=st.session_state.get('hybrid_alpha', 0.5),
                        chunk_size=st.session_state.get('chunk_size', 1000),
                        chunk_overlap=st.session_state.get('chunk_overlap', 200),
                        top_k=st.session_state.get('top_k', 3)
                    )
                    if pipeline_instance:
                        st.success("JEFF automatically initialized!")
                        logging.info("Auto-init successful.")
                        st.rerun()
                    else:
                        st.error("Auto-init failed. Try manual.")
                        logging.error("Auto-init failed.")
                except Exception as e:
                    st.error(f"Auto-init error: {e}. Try manual.")
                    logging.error(f"Auto-init error: {e}", exc_info=True)

def check_api_keys(embedding_model_enum, vector_store_enum, reranker_enum, llm_enum):
    """Check if required API keys are available in environment"""
    api_keys_status = {}
    missing_keys_list = []

    # Determine required keys based on selections
    openai_needed = (embedding_model_enum == EmbeddingModelType.OPENAI or
                     llm_enum in [LLMModelType.OPENAI_GPT35, LLMModelType.OPENAI_GPT4] or
                     True)  # OpenAI TTS always needs it
    cohere_needed = (embedding_model_enum == EmbeddingModelType.COHERE or
                     reranker_enum in [RerankerModelType.COHERE_V2, RerankerModelType.COHERE_V3, RerankerModelType.COHERE_MULTILINGUAL])
    gemini_needed = (embedding_model_enum == EmbeddingModelType.GEMINI or
                     llm_enum == LLMModelType.GEMINI)
    anthropic_needed = (llm_enum in [LLMModelType.CLAUDE_3_OPUS, LLMModelType.CLAUDE_37_SONNET])
    mistral_needed = (embedding_model_enum == EmbeddingModelType.MISTRAL or
                      llm_enum in [LLMModelType.MISTRAL_LARGE, LLMModelType.MISTRAL_MEDIUM, LLMModelType.MISTRAL_SMALL])
    voyage_needed = (embedding_model_enum == EmbeddingModelType.VOYAGE or
                     reranker_enum in [RerankerModelType.VOYAGE, RerankerModelType.VOYAGE_2])

    # Check and record status
    if openai_needed:
        key_name = "OpenAI API Key"
        is_available = bool(os.getenv("OPENAI_API_KEY"))
        api_keys_status[key_name] = "Available" if is_available else "Missing"
        if not is_available: missing_keys_list.append(key_name)

    if cohere_needed:
        key_name = "Cohere API Key"
        is_available = bool(os.getenv("COHERE_API_KEY"))
        api_keys_status[key_name] = "Available" if is_available else "Missing"
        if not is_available: missing_keys_list.append(key_name)

    if gemini_needed:
        key_name = "Gemini API Key"
        is_available = bool(os.getenv("GEMINI_API_KEY"))
        api_keys_status[key_name] = "Available" if is_available else "Missing"
        if not is_available: missing_keys_list.append(key_name)

    if anthropic_needed:
        key_name = "Anthropic API Key"
        is_available = bool(os.getenv("ANTHROPIC_API_KEY"))
        api_keys_status[key_name] = "Available" if is_available else "Missing"
        if not is_available: missing_keys_list.append(key_name)

    if mistral_needed:
        key_name = "Mistral API Key"
        is_available = bool(os.getenv("MISTRAL_API_KEY"))
        api_keys_status[key_name] = "Available" if is_available else "Missing"
        if not is_available: missing_keys_list.append(key_name)

    if voyage_needed:
        key_name = "Voyage AI API Key"
        is_available = bool(os.getenv("VOYAGE_API_KEY"))
        api_keys_status[key_name] = "Available" if is_available else "Missing"
        if not is_available: missing_keys_list.append(key_name)

    st.session_state.api_key_status = api_keys_status
    return missing_keys_list 