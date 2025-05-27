from typing import Dict, Any, Optional
import openai
import streamlit as st
from subject_configs import (
    SubjectConfig, 
    get_subject_config,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    DEFAULT_HYBRID_ALPHA
)
from pipeline_utils import initialize_pipeline
import logging
from enums import (
    EmbeddingModelType,
    RerankerModelType,
    LLMModelType,
    VectorStoreType,
    ChunkingStrategyType
)

def get_subject_configuration(subject: str, query: str) -> Dict[str, Any]:
    """
    Get the optimal RAG configuration for a specific subject using OpenAI's function calling API.
    Falls back to predefined configurations if API call fails.
    """
    try:
        # Define the function schema for OpenAI
        functions = [{
            "name": "get_subject_config",
            "description": "Get the optimal RAG configuration for a specific subject",
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_size": {
                        "type": "integer",
                        "description": "Size of text chunks for processing"
                    },
                    "chunk_overlap": {
                        "type": "integer",
                        "description": "Overlap between chunks"
                    },
                    "similarity_threshold": {
                        "type": "number",
                        "description": "Threshold for similarity matching"
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens for response generation"
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Temperature for response generation"
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "System prompt for the LLM"
                    }
                },
                "required": ["chunk_size", "chunk_overlap", "similarity_threshold", 
                           "max_tokens", "temperature", "system_prompt"]
            }
        }]

        # Call OpenAI API to get optimal configuration
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"Determine the optimal RAG configuration for {subject} textbooks, considering the user query: {query}."},
                {"role": "user", "content": f"What is the optimal RAG configuration for processing {subject} textbooks, specifically for a query like: '{query}'?"}
            ],
            functions=functions,
            function_call={"name": "get_subject_config"}
        )

        # Extract configuration from response
        config = response.choices[0].message.function_call.arguments
        return eval(config)  # Convert string to dict

    except Exception as e:
        print(f"Error getting configuration from OpenAI: {e}")
        # Fall back to predefined configuration
        config_obj = get_subject_config(subject) # Renamed to avoid confusion
        return {
            "chunk_size": config_obj.chunk_size,
            "chunk_overlap": config_obj.chunk_overlap,
            "similarity_threshold": 0.7, # Default
            "max_tokens": 1000,          # Default
            "temperature": 0.7,          # Default
            "system_prompt": "You are a helpful assistant." # Default
        }

def update_rag_configuration(subject: str, pipeline) -> Optional[bool]:
    """
    Update the RAG configuration based on the selected subject.
    Returns True if successful, False if failed, None if no update needed.
    """
    try:
        # Get subject-specific configuration
        subject_config = get_subject_config(subject)

        # Get current configuration from session state
        current_chunk_size = st.session_state.get('chunk_size', DEFAULT_CHUNK_SIZE)
        current_chunk_overlap = st.session_state.get('chunk_overlap', DEFAULT_CHUNK_OVERLAP)
        current_top_k = st.session_state.get('top_k', DEFAULT_TOP_K)
        current_hybrid_alpha = st.session_state.get('hybrid_alpha', DEFAULT_HYBRID_ALPHA)
        
        # Check if current configuration matches subject config
        if (current_chunk_size == subject_config.chunk_size and
            current_chunk_overlap == subject_config.chunk_overlap and
            current_top_k == subject_config.top_k and
            current_hybrid_alpha == subject_config.hybrid_alpha):
            logging.info(f"Current RAG parameters already match {subject} settings. No pipeline re-initialization needed.")
            return None

        logging.info(f"Updating RAG parameters for subject: {subject}. New config: ChunkSize={subject_config.chunk_size}, Overlap={subject_config.chunk_overlap}, TopK={subject_config.top_k}, Alpha={subject_config.hybrid_alpha}")

        # Update session state with new RAG parameters immediately
        st.session_state.chunk_size = subject_config.chunk_size
        st.session_state.chunk_overlap = subject_config.chunk_overlap
        st.session_state.top_k = subject_config.top_k
        st.session_state.hybrid_alpha = subject_config.hybrid_alpha

        # Reinitialize pipeline with new configuration using values from session_state
        # Ensure all model enums are correctly retrieved from session_state and converted
        try:
            embedding_enum = EmbeddingModelType.from_string(st.session_state.embedding_model)
            vs_enum = VectorStoreType.from_string(st.session_state.vector_store)
            reranker_enum = RerankerModelType.from_string(st.session_state.reranker)
            llm_enum = LLMModelType.from_string(st.session_state.llm_model)
            cs_enum = ChunkingStrategyType.from_string(st.session_state.chunking_strategy)
        except ValueError as e:
            logging.error(f"Failed to convert model string to enum: {e}")
            return False

        if not st.session_state.file_path:
            logging.error("Cannot reinitialize pipeline: File path is missing in session state.")
            return False

        pipeline_instance = initialize_pipeline(
            file_path=st.session_state.file_path, # Use file_path from session_state
            embedding_model_enum=embedding_enum,   # Pass enum from session_state
            vector_store_enum=vs_enum,           # Pass enum from session_state
            reranker_enum=reranker_enum,         # Pass enum from session_state
            llm_enum=llm_enum,                   # Pass enum from session_state
            chunking_strategy_enum=cs_enum,      # Pass enum from session_state
            hybrid_alpha=subject_config.hybrid_alpha, # Use new subject config value
            chunk_size=subject_config.chunk_size,         # Use new subject config value
            chunk_overlap=subject_config.chunk_overlap,   # Use new subject config value
            top_k=subject_config.top_k                    # Use new subject config value
        )

        if pipeline_instance:
            st.session_state.pipeline = pipeline_instance # IMPORTANT: Update the pipeline in session state
            logging.info(f"Successfully re-initialized RAG pipeline for {subject} and updated session_state.")
            return True
        else:
            logging.error(f"Failed to reinitialize RAG pipeline for {subject}. initialize_pipeline returned None.")
            return False

    except Exception as e:
        logging.error(f"Error updating RAG configuration for {subject}: {str(e)}", exc_info=True)
        return False 