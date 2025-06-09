import logging
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple

from utils.enums import VectorStoreType
from utils.subject_configs import DEFAULT_EMBEDDING_MODEL
from models.embedding_models import EmbeddingModelFactory
from models.rerankers import RerankerFactory
from models.vector_stores import VectorStoreFactory
from models.llm_models import LLMFactory
from models.chunking_strategies import ChunkingStrategyFactory
from pipeline.rag_pipeline import RAGPipeline, PipelineMetrics

from pipeline.components.exceptions import PipelineException, PipelineInitializationError
from pipeline.components.config import PipelineConfig
from pipeline.components.result import PipelineResult
from pipeline.components.model_combination import ModelCombination
from pipeline.components.progress import ProgressReporter, StreamlitProgressReporter

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

            # Create a metrics object
            metrics = PipelineMetrics(
                total_time=evaluation_metrics.get("total_time", 0.0),
                input_tokens=evaluation_metrics.get("input_tokens", 0),
                output_tokens=evaluation_metrics.get("output_tokens", 0),
                total_tokens=evaluation_metrics.get("total_tokens", 0),
                llm_cost=evaluation_metrics.get("llm_cost", 0.0),
                evaluation_scores={k: v for k, v in evaluation_metrics.items()
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