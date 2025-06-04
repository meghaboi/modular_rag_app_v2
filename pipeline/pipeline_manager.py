import os
import logging
import pandas as pd
import streamlit as st
from typing import Optional, Dict, Any, List, Tuple
import itertools
from dataclasses import dataclass
from abc import ABC, abstractmethod

from utils.enums import (
    EmbeddingModelType, RerankerModelType, LLMModelType, VectorStoreType,
    ChunkingStrategyType
)
from models.embedding_models import EmbeddingModelFactory
from models.rerankers import RerankerFactory
from models.vector_stores import VectorStoreFactory
from models.llm_models import LLMFactory
from pipeline.rag_pipeline import RAGPipeline, ChunkingStrategyFactory
from utils.utils import check_api_keys
from utils.subject_configs import DEFAULT_EMBEDDING_MODEL

class PipelineException(Exception):
    """Base exception class for pipeline-related errors"""
    def __init__(self, message: str, config: Optional[Dict[str, Any]] = None):
        self.message = message
        self.config = config
        super().__init__(self.message)

class PipelineInitializationError(PipelineException):
    """Raised when pipeline initialization fails"""
    pass

class PipelineExecutionError(PipelineException):
    """Raised when pipeline execution fails"""
    pass

@dataclass
class PipelineConfig:
    """Configuration class for RAG pipeline settings"""
    file_path: str
    vector_store_type: VectorStoreType
    reranker_type: RerankerModelType
    llm_type: LLMModelType
    chunking_strategy_type: ChunkingStrategyType
    hybrid_alpha: float = 0.5
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 3
    evaluation_mode: bool = False

    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate()

    def _validate(self) -> None:
        """Validate configuration values"""
        if not os.path.exists(self.file_path):
            raise ValueError(f"File path does not exist: {self.file_path}")
        
        if not 0 <= self.hybrid_alpha <= 1:
            raise ValueError(f"Hybrid alpha must be between 0 and 1, got {self.hybrid_alpha}")
        
        if self.chunk_size <= 0:
            raise ValueError(f"Chunk size must be positive, got {self.chunk_size}")
        
        if self.chunk_overlap < 0:
            raise ValueError(f"Chunk overlap must be non-negative, got {self.chunk_overlap}")
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"Chunk overlap ({self.chunk_overlap}) must be less than chunk size ({self.chunk_size})")
        
        if self.top_k <= 0:
            raise ValueError(f"Top k must be positive, got {self.top_k}")

    @classmethod
    def create_default(cls, file_path: str) -> 'PipelineConfig':
        """Create a default configuration with the given file path"""
        return cls(
            file_path=file_path,
            vector_store_type=VectorStoreType.CHROMA,
            reranker_type=RerankerModelType.NONE,
            llm_type=LLMModelType.CLAUDE_37_SONNET,
            chunking_strategy_type=ChunkingStrategyType.SIMPLE
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging and storage"""
        return {
            "file_path": self.file_path,
            "vector_store": self.vector_store_type.value,
            "reranker": self.reranker_type.value,
            "llm": self.llm_type.value,
            "chunking": self.chunking_strategy_type.value,
            "hybrid_alpha": self.hybrid_alpha,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "evaluation_mode": self.evaluation_mode
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create a PipelineConfig from a dictionary"""
        return cls(
            file_path=config_dict["file_path"],
            vector_store_type=VectorStoreType(config_dict["vector_store"]),
            reranker_type=RerankerModelType(config_dict["reranker"]),
            llm_type=LLMModelType(config_dict["llm"]),
            chunking_strategy_type=ChunkingStrategyType(config_dict["chunking"]),
            hybrid_alpha=config_dict.get("hybrid_alpha", 0.5),
            chunk_size=config_dict.get("chunk_size", 1000),
            chunk_overlap=config_dict.get("chunk_overlap", 200),
            top_k=config_dict.get("top_k", 3),
            evaluation_mode=config_dict.get("evaluation_mode", False)
        )

    def save_to_file(self, file_path: str) -> None:
        """Save configuration to a JSON file"""
        import json
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, file_path: str) -> 'PipelineConfig':
        """Load configuration from a JSON file"""
        import json
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def copy(self) -> 'PipelineConfig':
        """Create a copy of the configuration"""
        return PipelineConfig(
            file_path=self.file_path,
            vector_store_type=self.vector_store_type,
            reranker_type=self.reranker_type,
            llm_type=self.llm_type,
            chunking_strategy_type=self.chunking_strategy_type,
            hybrid_alpha=self.hybrid_alpha,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            top_k=self.top_k,
            evaluation_mode=self.evaluation_mode
        )

@dataclass
class PipelineMetrics:
    """Metrics for pipeline execution"""
    total_time: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    llm_cost: float
    ragas_scores: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        metrics_dict = {
            "total_time": self.total_time,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "llm_cost": self.llm_cost
        }
        if self.ragas_scores:
            metrics_dict.update(self.ragas_scores)
        return metrics_dict

@dataclass
class PipelineResult:
    """Result of a pipeline execution"""
    status: str
    response: Optional[str] = None
    contexts: Optional[List[str]] = None
    metrics: Optional[PipelineMetrics] = None
    error: Optional[str] = None
    config: Optional[Dict[str, Any]] = None

    @classmethod
    def success(
        cls,
        response: str,
        contexts: List[str],
        metrics: PipelineMetrics,
        config: Dict[str, Any]
    ) -> 'PipelineResult':
        """Create a successful result"""
        return cls(
            status="success",
            response=response,
            contexts=contexts,
            metrics=metrics,
            config=config
        )

    @classmethod
    def error(
        cls,
        error: str,
        config: Optional[Dict[str, Any]] = None
    ) -> 'PipelineResult':
        """Create an error result"""
        return cls(
            status="error",
            error=error,
            config=config
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        result_dict = {
            "status": self.status
        }
        if self.response is not None:
            result_dict["response"] = self.response
        if self.contexts is not None:
            result_dict["contexts"] = self.contexts
        if self.metrics is not None:
            result_dict["metrics"] = self.metrics.to_dict()
        if self.error is not None:
            result_dict["error"] = self.error
        if self.config is not None:
            result_dict["config"] = self.config
        return result_dict

@dataclass
class ModelCombination:
    """Represents a combination of models for testing"""
    embedding_model: EmbeddingModelType
    vector_store: VectorStoreType
    reranker: RerankerModelType
    llm: LLMModelType

    def to_config(self, base_config: PipelineConfig) -> PipelineConfig:
        """Convert to PipelineConfig using base configuration"""
        return PipelineConfig(
            file_path=base_config.file_path,
            vector_store_type=self.vector_store,
            reranker_type=self.reranker,
            llm_type=self.llm,
            chunking_strategy_type=base_config.chunking_strategy_type,
            hybrid_alpha=base_config.hybrid_alpha,
            chunk_size=base_config.chunk_size,
            chunk_overlap=base_config.chunk_overlap,
            top_k=base_config.top_k,
            evaluation_mode=base_config.evaluation_mode
        )

    @classmethod
    def get_default_combinations(cls) -> List['ModelCombination']:
        """Get default model combinations for testing"""
        return [
            cls(
                embedding_model=emb,
                vector_store=vec,
                reranker=rer,
                llm=llm
            )
            for emb, vec, rer, llm in itertools.product(
                [EmbeddingModelType.MISTRAL, EmbeddingModelType.OPENAI],
                [VectorStoreType.CHROMA, VectorStoreType.HYBRID],
                [RerankerModelType.NONE, RerankerModelType.COHERE_V3],
                [LLMModelType.CLAUDE_37_SONNET, LLMModelType.MISTRAL_LARGE]
            )
        ]

    @classmethod
    def from_config(cls, config: PipelineConfig) -> 'ModelCombination':
        """Create a ModelCombination from a PipelineConfig"""
        return cls(
            embedding_model=DEFAULT_EMBEDDING_MODEL,  # Using default as it's fixed
            vector_store=config.vector_store_type,
            reranker=config.reranker_type,
            llm=config.llm_type
        )

class ProgressReporter(ABC):
    """Abstract base class for progress reporting"""
    
    @abstractmethod
    def initialize(self, total_steps: int) -> None:
        """Initialize progress reporting with total number of steps"""
        pass
    
    @abstractmethod
    def update(self, current_step: int, message: str) -> None:
        """Update progress with current step and message"""
        pass
    
    @abstractmethod
    def complete(self) -> None:
        """Mark progress as complete"""
        pass

class StreamlitProgressReporter(ProgressReporter):
    """Streamlit implementation of progress reporting"""
    
    def __init__(self):
        self._progress_bar = None
    
    def initialize(self, total_steps: int) -> None:
        """Initialize Streamlit progress bar"""
        self._progress_bar = st.progress(0)
    
    def update(self, current_step: int, message: str) -> None:
        """Update Streamlit progress bar and display message"""
        if self._progress_bar:
            self._progress_bar.progress(current_step)
            st.write(message)
    
    def complete(self) -> None:
        """Complete progress reporting"""
        if self._progress_bar:
            self._progress_bar.progress(1.0)

class PipelineManager:
    """Manages RAG pipeline operations and configurations"""
    
    def __init__(self, progress_reporter: Optional[ProgressReporter] = None):
        self._current_pipeline: Optional[RAGPipeline] = None
        self._current_config: Optional[PipelineConfig] = None
        self._model_combinations: List[ModelCombination] = ModelCombination.get_default_combinations()
        self._progress_reporter = progress_reporter or StreamlitProgressReporter()
    
    def set_model_combinations(self, combinations: List[ModelCombination]) -> None:
        """Set custom model combinations for testing"""
        self._model_combinations = combinations
    
    def set_progress_reporter(self, reporter: ProgressReporter) -> None:
        """Set custom progress reporter"""
        self._progress_reporter = reporter
    
    def initialize_pipeline(self, config: PipelineConfig) -> Optional[RAGPipeline]:
        """
        Initialize RAG pipeline with the given configuration.
        
        Args:
            config: Pipeline configuration
            
        Returns:
            Optional[RAGPipeline]: Initialized pipeline or None if initialization fails
            
        Raises:
            PipelineInitializationError: If pipeline initialization fails
        """
        logging.info(f"Attempting to initialize RAG pipeline with config:")
        logging.info(f"  Embedding: {DEFAULT_EMBEDDING_MODEL.value}, Vector Store: {config.vector_store_type.value}, "
                    f"Reranker: {config.reranker_type.value}, LLM: {config.llm_type.value}")
        
        try:
            # Initialize components with fixed embedding model
            embedding_model_instance = EmbeddingModelFactory.create_model(DEFAULT_EMBEDDING_MODEL)
            vector_store_instance = VectorStoreFactory.create_store(
                config.vector_store_type, 
                alpha=config.hybrid_alpha
            ) if config.vector_store_type == VectorStoreType.HYBRID else VectorStoreFactory.create_store(config.vector_store_type)
            
            reranker_instance = RerankerFactory.create_reranker(config.reranker_type) if config.reranker_type != RerankerModelType.NONE else None
            llm_instance = LLMFactory.create_llm(config.llm_type)
            chunking_strategy_instance = ChunkingStrategyFactory.get_strategy(config.chunking_strategy_type.value)

            pipeline = RAGPipeline(
                embedding_model=embedding_model_instance,
                vector_store=vector_store_instance,
                reranker=reranker_instance,
                llm=llm_instance,
                top_k=config.top_k,
                chunking_strategy=chunking_strategy_instance,
                evaluation_mode=config.evaluation_mode 
            )

            # Indexing
            pipeline.index_documents(config.file_path, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
            
            self._current_pipeline = pipeline
            self._current_config = config
            return pipeline
            
        except Exception as e:
            error_msg = f"Failed to initialize pipeline: {str(e)}"
            logging.error(error_msg)
            raise PipelineInitializationError(error_msg, config.to_dict())
    
    def run_query(self, query: str, ground_truth: Optional[str] = None) -> PipelineResult:
        """
        Run a query through the current pipeline.
        
        Args:
            query: User's question
            ground_truth: Optional expected answer for evaluation
            
        Returns:
            PipelineResult: Result of the pipeline execution
            
        Raises:
            PipelineExecutionError: If pipeline execution fails
        """
        if not self._current_pipeline:
            error_msg = "No pipeline initialized"
            logging.error(error_msg)
            return PipelineResult.error(
                error_msg,
                self._current_config.to_dict() if self._current_config else None
            )
        
        try:
            # Run query and get non-streaming response
            response_text, contexts, metrics_from_run = self._current_pipeline.run(query)
            
            # Get evaluation metrics if ground truth is provided
            evaluation_metrics = {}
            if ground_truth:
                evaluation_metrics = self._current_pipeline.evaluate_response(
                    query=query,
                    response=response_text,
                    contexts=contexts,
                    ground_truth=ground_truth
                )
            else:
                evaluation_metrics = metrics_from_run

            # Create metrics object
            metrics = PipelineMetrics(
                total_time=evaluation_metrics.get("total_time", 0.0),
                input_tokens=evaluation_metrics.get("input_tokens", 0),
                output_tokens=evaluation_metrics.get("output_tokens", 0),
                total_tokens=evaluation_metrics.get("total_tokens", 0),
                llm_cost=evaluation_metrics.get("llm_cost", 0.0),
                ragas_scores={k: v for k, v in evaluation_metrics.items() 
                            if k not in ["total_time", "input_tokens", "output_tokens", 
                                       "total_tokens", "llm_cost"]}
            )

            return PipelineResult.success(
                response=response_text,
                contexts=contexts,
                metrics=metrics,
                config=self._current_config.to_dict()
            )

        except Exception as e:
            error_msg = f"Error running pipeline: {str(e)}"
            logging.error(error_msg, exc_info=True)
            return PipelineResult.error(error_msg, self._current_config.to_dict())
    
    def run_permutations(
        self,
        config: PipelineConfig,
        query: str,
        ground_truth: Optional[str] = None
    ) -> Tuple[List[PipelineResult], pd.DataFrame]:
        """
        Run all permutations of models and return results as a dataframe.
        
        Args:
            config: Base configuration to use
            query: User's question
            ground_truth: Optional expected answer for evaluation
            
        Returns:
            Tuple[List[PipelineResult], pd.DataFrame]: List of results and summary dataframe
        """
        results = []
        total_combinations = len(self._model_combinations)
        self._progress_reporter.initialize(total_combinations)
        
        for i, combination in enumerate(self._model_combinations, 1):
            message = (f"Testing combination {i}/{total_combinations}:\n"
                      f"Embedding: {DEFAULT_EMBEDDING_MODEL.value}, "
                      f"Vector Store: {combination.vector_store.value}, "
                      f"Reranker: {combination.reranker.value}, "
                      f"LLM: {combination.llm.value}")
            self._progress_reporter.update(i, message)
            
            try:
                # Create config for this combination
                test_config = combination.to_config(config)
                
                # Initialize and run pipeline
                pipeline = self.initialize_pipeline(test_config)
                if pipeline:
                    result = self.run_query(query, ground_truth)
                    results.append(result)
            except PipelineException as e:
                logging.error(f"Error in permutation {i}: {str(e)}")
                results.append(PipelineResult.error(str(e), test_config.to_dict()))
        
        self._progress_reporter.complete()

        # Convert results to DataFrame
        df_data = []
        for result in results:
            if result.status == "success" and result.metrics:
                row = {
                    "Embedding": DEFAULT_EMBEDDING_MODEL.value,
                    "Vector Store": result.config["vector_store"],
                    "Reranker": result.config["reranker"],
                    "LLM": result.config["llm"],
                    "Response": result.response,
                    "Metrics": result.metrics.to_dict()
                }
                df_data.append(row)
        
        df = pd.DataFrame(df_data)
        return results, df

    def save_current_config(self, file_path: str) -> None:
        """Save current pipeline configuration to file"""
        if not self._current_config:
            raise PipelineException("No configuration to save")
        self._current_config.save_to_file(file_path)
    
    def load_config(self, file_path: str) -> PipelineConfig:
        """Load pipeline configuration from file"""
        return PipelineConfig.load_from_file(file_path)
    
    def get_current_config(self) -> Optional[PipelineConfig]:
        """Get current pipeline configuration"""
        return self._current_config.copy() if self._current_config else None