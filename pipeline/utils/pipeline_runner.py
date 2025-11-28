import logging
import time
from typing import Dict, Any, Optional

from pipeline.components.config import PipelineConfig
from pipeline.components.result import PipelineResult
from models.evaluator import EvaluatorFactory, EvaluationBackendType, EvaluationMetricType
from pipeline.utils.pipeline_initializer import PipelineInitializer

class PipelineRunner:
    """Manages the execution and evaluation of a RAG pipeline."""

    @classmethod
    def run_pipeline_with_config(cls, user_query: str, ground_truth: str = None, embedding_model_enum=None, vector_store_enum=None, reranker_enum=None, llm_enum=None, chunking_strategy_enum=None, **kwargs):
        """
        Convenience method to build PipelineConfig from enums/params and run the pipeline.
        Accepts model enums and any additional pipeline params.
        """
        # Import here to avoid circular imports
        from pipeline.components.config import PipelineConfig
        config = PipelineConfig(
            file_path=kwargs.get('file_path'),
            embedding_model_type=embedding_model_enum,
            vector_store_type=vector_store_enum,
            reranker_type=reranker_enum,
            llm_type=llm_enum,
            chunking_strategy_type=chunking_strategy_enum,
            hybrid_alpha=kwargs.get('hybrid_alpha', PipelineConfig.DEFAULT_HYBRID_ALPHA),
            chunk_size=kwargs.get('chunk_size', PipelineConfig.DEFAULT_CHUNK_SIZE),
            chunk_overlap=kwargs.get('chunk_overlap', PipelineConfig.DEFAULT_CHUNK_OVERLAP),
            top_k=kwargs.get('top_k', PipelineConfig.DEFAULT_TOP_K),
            evaluation_mode=kwargs.get('evaluation_mode', PipelineConfig.DEFAULT_EVALUATION_MODE),
            precomputed_chunks=kwargs.get('precomputed_chunks')
        )
        runner = cls(config, user_query, ground_truth)
        return runner.run()


    def __init__(self, config: PipelineConfig, user_query: str, ground_truth: Optional[str] = None):
        self.config = config
        self.user_query = user_query
        self.ground_truth = ground_truth
        self.logger = logging.getLogger(__name__)

    def run(self) -> Dict[str, Any]:
        """Executes the pipeline, evaluates the result, and returns a dictionary."""
        start_time = time.time()
        try:
            initializer = PipelineInitializer(self.config)
            pipeline = initializer.initialize_pipeline()

            self.logger.info(f"Running pipeline with query: '{self.user_query[:50]}...' ")
            response, contexts, metrics = pipeline.run(self.user_query)
            self.logger.info(f"Query processed. Response length: {len(response)}")

            eval_scores = self._run_evaluation(response, contexts)
            
            result = PipelineResult.success(
                response=response, 
                contexts=contexts, 
                metrics=metrics, 
                config=self.config.to_dict()
            )
            
            # Merge evaluation scores into the final dictionary
            final_dict = result.to_dict()
            final_dict.update(eval_scores)
            final_dict["elapsed_time"] = time.time() - start_time
            return final_dict

        except Exception as e:
            self.logger.error(f"Error running pipeline: {e}", exc_info=True)
            error_result = PipelineResult.error(error=str(e), config=self.config.to_dict())
            final_dict = error_result.to_dict()
            final_dict["elapsed_time"] = time.time() - start_time
            return final_dict

    def _run_evaluation(self, response: str, contexts: list) -> Dict[str, Any]:
        """Runs evaluations if ground truth is available."""
        if not self.ground_truth:
            self.logger.warning("No ground truth provided, skipping evaluation.")
            return {}
        
        ragas_scores = self._evaluate_with_backend(EvaluationBackendType.RAGAS_V2, response, contexts)

        return {
            "ragas_evaluation_scores": ragas_scores,
        }

    def _evaluate_with_backend(self, backend: EvaluationBackendType, response: str, contexts: list) -> Dict[str, float]:
        """Helper to run evaluation with a specific backend."""
        try:
            evaluator = EvaluatorFactory.create_evaluator(
                backend, EvaluationMetricType.get_metrics_for_backend(backend)
            )
            scores = evaluator.evaluate(
                query=self.user_query, response=response, 
                contexts=contexts, ground_truth=self.ground_truth
            )
            self.logger.info(f"{backend.value} evaluation scores: {scores}")
            return scores
        except Exception as e:
            self.logger.error(f"Error during {backend.value} evaluation: {e}", exc_info=True)
            return {"error": str(e)}