import logging
import time
import os
import streamlit as st
from typing import Dict, Any
from pipeline.components.config import PipelineConfig
from pipeline.components.exceptions import RAGPipelineInitializationError
from models.embedding_models import EmbeddingModelFactory
from models.vector_stores import VectorStoreFactory
from models.rerankers import RerankerFactory
from models.llm_models import LLMFactory
from models.chunking_strategies import ChunkingStrategyFactory
from pipeline.rag_pipeline import RAGPipeline
from utils.enums import RerankerModelType

class PipelineInitializer:
    """Handles the creation and initialization of a RAG pipeline."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def initialize_pipeline(self) -> RAGPipeline:
        """Initialize RAG pipeline with the given configuration."""
        self._log_initialization_start()
        self._validate_file_path()

        try:
            components = self._create_pipeline_components()
            pipeline = RAGPipeline(**components, evaluation_mode=self.config.evaluation_mode)
            self._index_documents(pipeline)
            self.logger.info("RAG pipeline initialized successfully.")
            return pipeline
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG pipeline: {e}", exc_info=True)
            raise RAGPipelineInitializationError(f"Pipeline initialization failed: {e}") from e

    def _log_initialization_start(self):
        """Logs the start of the pipeline initialization process."""
        self.logger.info("Attempting to initialize RAG pipeline with config:")
        self.logger.info(f"  Embedding: {self.config.embedding_model_type.value}")
        self.logger.info(f"  Vector Store: {self.config.vector_store_type.value}")
        self.logger.info(f"  Reranker: {self.config.reranker_type.value}")
        self.logger.info(f"  LLM: {self.config.llm_type.value}")
        self.logger.info(f"  Chunking: {self.config.chunking_strategy_type.value}")

    def _validate_file_path(self):
        """Validates the existence of the input file path."""
        if not self.config.file_path or not os.path.exists(self.config.file_path):
            raise RAGPipelineInitializationError("Invalid or non-existent file path provided.")

    def _create_pipeline_components(self) -> Dict[str, Any]:
        """Creates and returns a dictionary of pipeline components."""
        embedding_model = EmbeddingModelFactory.create_model(self.config.embedding_model_type)
        vector_store = VectorStoreFactory.create_store(
            self.config.vector_store_type, alpha=self.config.hybrid_alpha
        )
        llm = LLMFactory.create_llm(self.config.llm_type)

        reranker = None
        if self.config.reranker_type != RerankerModelType.NONE:
            reranker = RerankerFactory.create_reranker(
                self.config.reranker_type, llm_client=llm
            )
        chunking_strategy = ChunkingStrategyFactory.get_strategy(
            self.config.chunking_strategy_type.value
        )

        return {
            "embedding_model": embedding_model,
            "vector_store": vector_store,
            "reranker": reranker,
            "llm": llm,
            "top_k": self.config.top_k,
            "chunking_strategy": chunking_strategy,
            "precomputed_chunks": getattr(self.config, 'precomputed_chunks', None),
        }

    def _index_documents(self, pipeline: RAGPipeline):
        """Indexes documents into the pipeline."""
        self.logger.info(f"Indexing documents from: {self.config.file_path}")
        start_time = time.time()
        try:
            pipeline.index_documents(
                self.config.file_path,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
        except Exception as e:
            raise RAGPipelineInitializationError(f"Document indexing failed: {e}") from e
        
        duration = time.time() - start_time
        self.logger.info(f"Document indexing completed in {duration:.2f} seconds.")