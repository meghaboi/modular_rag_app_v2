import logging
import time
import streamlit as st
from typing import Dict, Any
from utils.enums import (
    EmbeddingModelType,
    RerankerModelType,
    LLMModelType,
    VectorStoreType,
    ChunkingStrategyType,
    EvaluationBackendType,
    EvaluationMetricType
)
from utils.api_management.api_utils import check_api_keys
from models.embedding_models import EmbeddingModelFactory
from models.vector_stores import VectorStoreFactory
from models.rerankers import RerankerFactory
from models.llm_models import LLMFactory
from models.chunking_strategies import ChunkingStrategyFactory
from models.evaluator import EvaluatorFactory
from pipeline.rag_pipeline import RAGPipeline

class PipelineRunner:
    @staticmethod
    def run_pipeline_with_config(
        file_path: str,
        user_query: str,
        ground_truth: str,
        embedding_model_enum: EmbeddingModelType,
        vector_store_enum: VectorStoreType,
        reranker_enum: RerankerModelType,
        llm_enum: LLMModelType,
        chunking_strategy_enum: ChunkingStrategyType,
        hybrid_alpha: float = 0.5,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """Run a single pipeline configuration and return results"""
        config_str = f"{embedding_model_enum.value}, {vector_store_enum.value}, {reranker_enum.value}, {llm_enum.value}, {chunking_strategy_enum.value}"
        logging.info(f"Running pipeline with config: {config_str}")
        start_run_time = time.time()
        try:
            # Initialize components
            embedding_model_instance = EmbeddingModelFactory.create_model(embedding_model_enum)
            vector_store_instance = VectorStoreFactory.create_store(vector_store_enum, alpha=hybrid_alpha) if vector_store_enum == VectorStoreType.HYBRID else VectorStoreFactory.create_store(vector_store_enum)
            reranker_instance = RerankerFactory.create_reranker(reranker_enum) if reranker_enum != RerankerModelType.NONE else None
            llm_instance = LLMFactory.create_llm(llm_enum)
            chunking_strategy_instance = ChunkingStrategyFactory.get_strategy(chunking_strategy_enum.value)

            is_in_evaluation_mode = st.session_state.mode == "evaluation"

            pipeline = RAGPipeline(
                embedding_model=embedding_model_instance,
                vector_store=vector_store_instance,
                reranker=reranker_instance,
                llm=llm_instance,
                top_k=top_k,
                chunking_strategy=chunking_strategy_instance,
                evaluation_mode=is_in_evaluation_mode 
            )

            # Indexing (re-index per config for isolation in eval)
            pipeline.index_documents(file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            # Process query
            start_query_time = time.time()
            response, contexts, metrics = pipeline.run(user_query)
            query_elapsed_time = time.time() - start_query_time
            logging.info(f"Query processed in {query_elapsed_time:.2f}s. Response length: {len(response)}")

            # Initialize evaluation results
            custom_evaluation_results = {}
            ragas_evaluation_results = {}
            avg_custom_score = 0

            if ground_truth:
                # RAGAS Evaluation
                try:
                    ragas_evaluator = EvaluatorFactory.create_evaluator(
                        EvaluationBackendType.RAGAS_V2,
                        EvaluationMetricType.get_metrics_for_backend(EvaluationBackendType.RAGAS_V2)
                    )
                    ragas_evaluation_results = ragas_evaluator.evaluate(
                        query=user_query, response=response, contexts=contexts, ground_truth=ground_truth
                    )
                    logging.info(f"RAGAS evaluation scores: {ragas_evaluation_results}")
                except Exception as ragas_eval_e:
                    logging.error(f"Error during RAGAS evaluation for config {config_str}: {ragas_eval_e}", exc_info=True)
                    ragas_evaluation_results = {"error": str(ragas_eval_e)}
            else:
                logging.warning("No ground truth provided, skipping RAGAS evaluation.")

            total_elapsed_time = time.time() - start_run_time
            logging.info(f"Total run time for config {config_str}: {total_elapsed_time:.2f}s")

            # Combine all results
            flat_custom_scores = {f"custom_{k}": v for k, v in custom_evaluation_results.items()}
            flat_ragas_scores = {f"ragas_{k}": v for k, v in ragas_evaluation_results.items()}

            results = {
                "embedding_model": embedding_model_enum.value,
                "vector_store": vector_store_enum.value,
                "reranker": reranker_enum.value,
                "llm_model": llm_enum.value,
                "chunking_strategy": chunking_strategy_enum.value,
                "response": response,
                "contexts": contexts,
                "custom_evaluation_scores": custom_evaluation_results,
                "ragas_evaluation_scores": ragas_evaluation_results,
                "avg_custom_score": avg_custom_score,
                "metrics": metrics,
                "elapsed_time": total_elapsed_time
            }
            results.update(flat_custom_scores)
            results.update(flat_ragas_scores)

            return results

        except Exception as e:
            logging.error(f"Error running pipeline with config {config_str}: {e}", exc_info=True)
            return {
                "embedding_model": embedding_model_enum.value,
                "vector_store": vector_store_enum.value,
                "reranker": reranker_enum.value,
                "llm_model": llm_enum.value,
                "chunking_strategy": chunking_strategy_enum.value,
                "response": "ERROR",
                "contexts": [],
                "custom_evaluation_scores": {"error": str(e)},
                "ragas_evaluation_scores": {"error": "Pipeline Error"},
                "avg_custom_score": 0,
                "metrics": {},
                "elapsed_time": time.time() - start_run_time,
                "error": str(e)
            } 