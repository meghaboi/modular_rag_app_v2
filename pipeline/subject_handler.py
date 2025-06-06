from typing import Dict, Any, Optional
import streamlit as st
from utils.subject_configs import (
    SubjectConfig, 
    get_subject_config,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    DEFAULT_HYBRID_ALPHA
)
from pipeline.pipeline_utils import initialize_pipeline
import logging
from utils.enums import (
    EmbeddingModelType,
    RerankerModelType,
    LLMModelType,
    VectorStoreType,
    ChunkingStrategyType
)
from utils.utils import determine_prompt_nature # Added import

def get_subject_configuration(subject: str, query: str) -> Dict[str, Any]:
    """
    Get the optimal RAG configuration for a specific subject using OpenAI's function calling API.
    Falls back to predefined configurations if API call fails.
    THIS FUNCTION IS DEPRECATED.
    """
    raise NotImplementedError("This function is deprecated. Use get_config_by_prompt_nature instead.")

def get_config_by_prompt_nature(query: str) -> SubjectConfig:
    """
    Determines the prompt nature and retrieves the corresponding SubjectConfig.
    """
    prompt_nature = determine_prompt_nature(query)
    logging.info(f"Determined prompt nature: {prompt_nature} for query: '{query[:50]}...'")
    
    config = get_subject_config(prompt_nature)
    logging.info(f"Using configuration for '{prompt_nature}': ChunkSize={config.chunk_size}, Overlap={config.chunk_overlap}, TopK={config.top_k}, Alpha={config.hybrid_alpha}")
    
    return config

def update_rag_configuration(query: str, pipeline, subject: Optional[str] = None) -> Optional[bool]:
    """
    Update the RAG configuration based on the determined nature of the query.
    Optionally uses subject as a fallback or for additional context if needed in future.
    Returns True if successful, False if failed, None if no update needed.
    """
    try:
        nature_config = get_config_by_prompt_nature(query)

        if not nature_config:
            logging.warning(f"Could not determine configuration for query: '{query[:50]}...'. No update will be performed based on nature.")
            if subject:
                logging.info(f"Falling back to subject-based configuration for: {subject}")
                nature_config = get_subject_config(subject)
            else:
                logging.warning("No subject provided for fallback. No RAG configuration update.")
                return None

        current_chunk_size = st.session_state.get('chunk_size', DEFAULT_CHUNK_SIZE)
        current_chunk_overlap = st.session_state.get('chunk_overlap', DEFAULT_CHUNK_OVERLAP)
        current_top_k = st.session_state.get('top_k', DEFAULT_TOP_K)
        current_hybrid_alpha = st.session_state.get('hybrid_alpha', DEFAULT_HYBRID_ALPHA)
        
        if (current_chunk_size == nature_config.chunk_size and
            current_chunk_overlap == nature_config.chunk_overlap and
            current_top_k == nature_config.top_k and
            current_hybrid_alpha == nature_config.hybrid_alpha):
            logging.info(f"Current RAG parameters already match query nature-derived settings. No pipeline re-initialization needed for query: '{query[:50]}...'")
            return None

        logging.info(f"Updating RAG parameters based on query nature. New config: ChunkSize={nature_config.chunk_size}, Overlap={nature_config.chunk_overlap}, TopK={nature_config.top_k}, Alpha={nature_config.hybrid_alpha} for query: '{query[:50]}...'")

        st.session_state.chunk_size = nature_config.chunk_size
        st.session_state.chunk_overlap = nature_config.chunk_overlap
        st.session_state.top_k = nature_config.top_k
        st.session_state.hybrid_alpha = nature_config.hybrid_alpha

        try:
            embedding_enum = EmbeddingModelType.from_string(st.session_state.embedding_model)
            vs_enum = VectorStoreType.from_string(st.session_state.vector_store)
            reranker_enum = RerankerModelType.from_string(st.session_state.reranker)
            llm_enum = LLMModelType.from_string(st.session_state.llm_model)
            cs_enum = ChunkingStrategyType.from_string(st.session_state.chunking_strategy)
        except ValueError as e:
            logging.error(f"Failed to convert model string to enum: {e}")
            return False

        if not st.session_state.get('file_path'):
            logging.error("Cannot reinitialize pipeline: File path is missing in session state.")
            return False

        pipeline_instance = initialize_pipeline(
            file_path=st.session_state.file_path,
            embedding_model_enum=embedding_enum,
            vector_store_enum=vs_enum,
            reranker_enum=reranker_enum,
            llm_enum=llm_enum,
            chunking_strategy_enum=cs_enum,
            hybrid_alpha=nature_config.hybrid_alpha, 
            chunk_size=nature_config.chunk_size,
            chunk_overlap=nature_config.chunk_overlap,
            top_k=nature_config.top_k
        )

        if pipeline_instance:
            st.session_state.pipeline = pipeline_instance
            logging.info(f"Successfully re-initialized RAG pipeline based on query nature and updated session_state for query: '{query[:50]}...'")
            return True
        else:
            logging.error(f"Failed to reinitialize RAG pipeline based on query nature for query: '{query[:50]}...'. initialize_pipeline returned None.")
            return False

    except Exception as e:
        log_subject = subject if subject else "N/A"
        logging.error(f"Error updating RAG configuration for query '{query[:50]}...' (Subject: {log_subject}): {str(e)}", exc_info=True)
        return False