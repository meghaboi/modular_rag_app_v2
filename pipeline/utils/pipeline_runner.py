import logging
import time
from typing import Dict, Any, Optional

from pipeline.components.config import PipelineConfig
from pipeline.components.result import PipelineResult
from pipeline.components.exceptions import RAGPipelineExecutionError
from models.evaluator import EvaluatorFactory, EvaluationBackendType, EvaluationMetricType
from .pipeline_initializer import PipelineInitializer

class PipelineRunner:
    """Manages the execution and evaluation of a RAG pipeline."""

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
        # custom_scores = self._evaluate_with_backend(EvaluationBackendType.CUSTOM, response, contexts)
        
        # Placeholder for aggregation until custom evaluation is fully implemented
        # avg_custom_score = sum(custom_scores.values()) / len(custom_scores) if custom_scores else 0

        return {
            "ragas_evaluation_scores": ragas_scores,
            # "custom_evaluation_scores": custom_scores,
            # "avg_custom_score": avg_custom_score
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