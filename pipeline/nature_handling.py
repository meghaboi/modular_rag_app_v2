import logging
from functools import lru_cache

import streamlit as st

from pipeline.components.config import PipelineConfig
from pipeline.components.exceptions import RAGPipelineInitializationError
from pipeline.utils.pipeline_initializer import PipelineInitializer
from utils.analysis.analysis_utils import determine_prompt_nature
from utils.enums import (
    ChunkingStrategyType,
    EmbeddingModelType,
    LLMModelType,
    RerankerModelType,
    VectorStoreType,
)
from utils.subject_configs import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_HYBRID_ALPHA,
    DEFAULT_TOP_K,
    SubjectConfig,
    get_subject_config,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=128)
def get_config_by_prompt_nature(query: str) -> SubjectConfig:
    """
    Determines the prompt nature and retrieves the corresponding SubjectConfig.
    Uses LRU caching to avoid repeated calls for the same query.
    """
    prompt_nature = determine_prompt_nature(query)
    logger.info(f"Determined prompt nature: {prompt_nature} for query: '{query[:50]}...'")
    config = get_subject_config(prompt_nature)
    logger.info(
        f"Using configuration for '{prompt_nature}': "
        f"ChunkSize={config.chunk_size}, Overlap={config.chunk_overlap}, "
        f"TopK={config.top_k}, Alpha={config.hybrid_alpha}"
    )
    return config

class NatureBasedRAGUpdater:
    """Handles dynamic RAG configuration updates based on prompt nature."""

    def __init__(self, session_state):
        self.session_state = session_state
        self.logger = logging.getLogger(__name__)

    def update_pipeline_if_needed(self, query: str) -> bool:
        """
        Updates the RAG pipeline if the prompt nature suggests a different configuration.
        Returns True if the pipeline was updated, False otherwise.
        """
        if self.session_state.get("is_evaluation_mode", False):
            self.logger.info("Skipping nature-based configuration update in evaluation mode.")
            return False

        try:
            nature_config = get_config_by_prompt_nature(query)
            current_config = self._get_current_pipeline_config()

            if not self._is_update_required(current_config, nature_config):
                self.logger.info("Current RAG parameters already match query nature. No re-initialization needed.")
                return False

            self.logger.info("Updating RAG parameters based on query nature.")
            new_config = self._create_new_pipeline_config(current_config, nature_config)

            initializer = PipelineInitializer(new_config)
            pipeline_instance = initializer.initialize_pipeline()

            self._update_session_state(new_config, pipeline_instance)
            self.logger.info("Successfully re-initialized RAG pipeline based on query nature.")
            return True

        except (ValueError, RAGPipelineInitializationError) as e:
            self.logger.error(f"Failed to update RAG configuration for query '{query[:50]}...': {e}", exc_info=True)
            return False
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while updating RAG configuration: {e}", exc_info=True)
            return False

    def _get_current_pipeline_config(self) -> PipelineConfig:
        """Constructs a PipelineConfig object from the current session state."""
        if not self.session_state.get("file_path"):
            raise RAGPipelineInitializationError("Cannot create config: File path is missing in session state.")

        return PipelineConfig(
            file_path=self.session_state.file_path,
            embedding_model_type=EmbeddingModelType.from_string(self.session_state.embedding_model),
            vector_store_type=VectorStoreType.from_string(self.session_state.vector_store),
            reranker_type=RerankerModelType.from_string(self.session_state.reranker),
            llm_type=LLMModelType.from_string(self.session_state.llm_model),
            chunking_strategy_type=ChunkingStrategyType.from_string(self.session_state.chunking_strategy),
            chunk_size=self.session_state.get("chunk_size", DEFAULT_CHUNK_SIZE),
            chunk_overlap=self.session_state.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP),
            top_k=self.session_state.get("top_k", DEFAULT_TOP_K),
            hybrid_alpha=self.session_state.get("hybrid_alpha", DEFAULT_HYBRID_ALPHA),
        )

    def _is_update_required(self, current_config: PipelineConfig, nature_config: SubjectConfig) -> bool:
        """Checks if the current configuration differs from the nature-based one."""
        return not (
            current_config.chunk_size == nature_config.chunk_size
            and current_config.chunk_overlap == nature_config.chunk_overlap
        )

    def _create_new_pipeline_config(self, base_config: PipelineConfig, nature_config: SubjectConfig) -> PipelineConfig:
        """Creates a new config object with updated parameters from nature_config."""
        new_config_dict = base_config.to_dict()
        new_config_dict.update(
            {
                "chunk_size": nature_config.chunk_size,
                "chunk_overlap": nature_config.chunk_overlap,
                "top_k": nature_config.top_k,
                "hybrid_alpha": nature_config.hybrid_alpha,
            }
        )
        return PipelineConfig.from_dict(new_config_dict)

    def _update_session_state(self, new_config: PipelineConfig, pipeline_instance):
        """Updates the session state with the new configuration and pipeline instance."""
        self.session_state.chunk_size = new_config.chunk_size
        self.session_state.chunk_overlap = new_config.chunk_overlap
        self.session_state.top_k = new_config.top_k
        self.session_state.hybrid_alpha = new_config.hybrid_alpha
        self.session_state.pipeline = pipeline_instance

def update_rag_configuration(query: str) -> bool:
    """
    Update the RAG configuration based on the determined nature of the query.
    This is a wrapper around the NatureBasedRAGUpdater class.
    Returns True if successful, False if failed.
    """
    updater = NatureBasedRAGUpdater(st.session_state)
    return updater.update_pipeline_if_needed(query)