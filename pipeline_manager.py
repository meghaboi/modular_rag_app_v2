import os
import logging
import pandas as pd
import streamlit as st
from typing import Optional, Dict, Any, List, Tuple
import itertools

from enums import (
    EmbeddingModelType, RerankerModelType, LLMModelType, VectorStoreType,
    ChunkingStrategyType, EvaluationBackendType, EvaluationMetricType # Added Evaluation Enums
)
from embedding_models import EmbeddingModelFactory
from rerankers import RerankerFactory
from vector_stores import VectorStoreFactory
from llm_models import LLMFactory
from rag_pipeline import RAGPipeline, Indexer, Retriever # ChunkingStrategyFactory removed from here
from chunking_strategies import ChunkingStrategyFactory # Added import from new location
from evaluator import EvaluatorFactory, BaseEvaluator
# from config import check_api_keys
import subject_configs

class PipelineBuilder:
    def __init__(self,
                 file_path: str,
                 vector_store_enum: VectorStoreType,
                 reranker_enum: RerankerModelType,
                 llm_enum: LLMModelType,
                 chunking_strategy_enum: ChunkingStrategyType,
                 hybrid_alpha: float,
                 chunk_size: int,
                 chunk_overlap: int,
                 top_k: int,
                 evaluation_backend_type: Optional[EvaluationBackendType] = None,
                 evaluation_metrics: Optional[List[str]] = None):
        self.file_path = file_path
        self.embedding_model_enum = subject_configs.DEFAULT_EMBEDDING_MODEL # Fixed
        self.vector_store_enum = vector_store_enum
        self.reranker_enum = reranker_enum
        self.llm_enum = llm_enum
        self.chunking_strategy_enum = chunking_strategy_enum
        self.hybrid_alpha = hybrid_alpha
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.evaluation_backend_type = evaluation_backend_type
        self.evaluation_metrics = evaluation_metrics

    def build_pipeline(self) -> Optional[RAGPipeline]:
        # Use defaults from subject_configs if specific values are not provided
        final_hybrid_alpha = self.hybrid_alpha if self.hybrid_alpha is not None else subject_configs.DEFAULT_HYBRID_ALPHA
        final_chunk_size = self.chunk_size if self.chunk_size is not None else subject_configs.DEFAULT_CHUNK_SIZE
        final_chunk_overlap = self.chunk_overlap if self.chunk_overlap is not None else subject_configs.DEFAULT_CHUNK_OVERLAP
        final_top_k = self.top_k if self.top_k is not None else subject_configs.DEFAULT_TOP_K

        final_eval_backend = self.evaluation_backend_type if self.evaluation_backend_type is not None else EvaluationBackendType.RAGAS_V2
        final_eval_metrics = self.evaluation_metrics if self.evaluation_metrics is not None else EvaluationMetricType.get_metrics_for_backend(final_eval_backend)

        logging.info(f"Attempting to build RAG pipeline with config:")
        logging.info(f"  File Path: {self.file_path}")
        logging.info(f"  Embedding: {self.embedding_model_enum.value}, Vector Store: {self.vector_store_enum.value}, "
                    f"Reranker: {self.reranker_enum.value}, LLM: {self.llm_enum.value}")
        logging.info(f"  Chunking: Strategy={self.chunking_strategy_enum.value}, Size={final_chunk_size}, Overlap={final_chunk_overlap}, TopK={final_top_k}, HybridAlpha={final_hybrid_alpha}")
        logging.info(f"  Evaluation: Backend={final_eval_backend.value}, Metrics={final_eval_metrics}")

        try:
            embedding_model_instance = EmbeddingModelFactory.create_model(self.embedding_model_enum)
            vector_store_instance = VectorStoreFactory.create_store(self.vector_store_enum, alpha=final_hybrid_alpha) if self.vector_store_enum == VectorStoreType.HYBRID else VectorStoreFactory.create_store(self.vector_store_enum)
            reranker_instance = RerankerFactory.create_reranker(self.reranker_enum) if self.reranker_enum != RerankerModelType.NONE else None
            llm_instance = LLMFactory.create_llm(self.llm_enum)
            chunking_strategy_instance = ChunkingStrategyFactory.get_strategy(self.chunking_strategy_enum.value)

            indexer_instance = Indexer(
                chunking_strategy=chunking_strategy_instance,
                embedding_model=embedding_model_instance,
                vector_store=vector_store_instance
            )

            retriever_instance = Retriever(
                embedding_model=embedding_model_instance,
                vector_store=vector_store_instance,
                reranker=reranker_instance,
                top_k=final_top_k
            )

            evaluator_instance: Optional[BaseEvaluator] = None
            if final_eval_backend and final_eval_metrics:
                try:
                    evaluator_instance = EvaluatorFactory.create_evaluator(final_eval_backend, final_eval_metrics)
                    logging.info(f"Successfully created evaluator: {final_eval_backend.value}")
                except ValueError as e:
                    logging.error(f"Failed to create evaluator: {e}. Evaluation will be skipped.")
                    evaluator_instance = None
            else:
                logging.info("Evaluation backend or metrics not specified, skipping evaluator creation.")

            is_in_evaluation_mode = False
            if hasattr(st, 'session_state') and st.session_state.get('mode') == "evaluation":
                 is_in_evaluation_mode = True

            pipeline = RAGPipeline(
                llm=llm_instance,
                indexer=indexer_instance,
                retriever=retriever_instance,
                evaluator=evaluator_instance,
                evaluation_mode=is_in_evaluation_mode
            )

            pipeline.initialize(self.file_path, final_chunk_size, final_chunk_overlap)

            return pipeline

        except Exception as e:
            logging.error(f"Failed to build pipeline: {str(e)}", exc_info=True)
            return None

def initialize_pipeline(
    file_path: str,
    vector_store_enum: VectorStoreType,
    reranker_enum: RerankerModelType,
    llm_enum: LLMModelType,
    chunking_strategy_enum: ChunkingStrategyType,
    hybrid_alpha: float,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    evaluation_backend_type: Optional[EvaluationBackendType] = None,
    evaluation_metrics: Optional[List[str]] = None
) -> Optional[RAGPipeline]:
    """
    Initialize RAG pipeline with selected configuration using PipelineBuilder.
    """
    builder = PipelineBuilder(
        file_path=file_path,
        vector_store_enum=vector_store_enum,
        reranker_enum=reranker_enum,
        llm_enum=llm_enum,
        chunking_strategy_enum=chunking_strategy_enum,
        hybrid_alpha=hybrid_alpha,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        evaluation_backend_type=evaluation_backend_type,
        evaluation_metrics=evaluation_metrics
    )
    return builder.build_pipeline()

class ExperimentRunner:
    def __init__(self,
                 file_path: str,
                 user_query: str,
                 ground_truth: str,
                 chunk_size: int, # Base chunk size for the experiment
                 chunk_overlap: int, # Base chunk overlap
                 top_k: int, # Base top_k
                 hybrid_alpha: float, # Base hybrid_alpha
                 chunking_strategy_enum: ChunkingStrategyType, # Base chunking strategy
                 evaluation_backend_type: EvaluationBackendType,
                 evaluation_metrics: List[str]):
        self.file_path = file_path
        self.user_query = user_query
        self.ground_truth = ground_truth
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.hybrid_alpha = hybrid_alpha
        self.chunking_strategy_enum = chunking_strategy_enum
        self.evaluation_backend_type = evaluation_backend_type
        self.evaluation_metrics = evaluation_metrics

    def run_permutations(self, configurations: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
        results = []
        total_configurations = len(configurations)
        progress_bar = st.progress(0)

        for i, config in enumerate(configurations, 1):
            st.write(f"Testing configuration {i}/{total_configurations}:")
            # Config provides: vector_store_enum, reranker_enum, llm_enum
            # Other params (file_path, query, ground_truth, chunk_*, top_k, etc.) come from self

            current_vector_store = config.get('vector_store_enum', VectorStoreType.CHROMA) # Default if not in config
            current_reranker = config.get('reranker_enum', RerankerModelType.NONE) # Default if not in config
            current_llm = config.get('llm_enum', subject_configs.DEFAULT_LLM_MODEL) # Default if not in config

            # Log the specific components being used for this permutation
            logging.info(f"Running permutation {i}/{total_configurations}: VS='{current_vector_store.value}', Reranker='{current_reranker.value}', LLM='{current_llm.value}'")
            st.write(f"  Vector Store: {current_vector_store.value}, Reranker: {current_reranker.value}, LLM: {current_llm.value}")


            result = run_pipeline_with_config(
                file_path=self.file_path,
                user_query=self.user_query,
                ground_truth=self.ground_truth,
                embedding_model_enum=subject_configs.DEFAULT_EMBEDDING_MODEL, # Fixed for now
                vector_store_enum=current_vector_store,
                reranker_enum=current_reranker,
                llm_enum=current_llm,
                chunking_strategy_enum=self.chunking_strategy_enum,
                hybrid_alpha=self.hybrid_alpha,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                top_k=self.top_k,
                evaluation_backend_type=self.evaluation_backend_type,
                evaluation_metrics=self.evaluation_metrics
            )
            results.append(result)
            progress_bar.progress(i / total_configurations)

        df_data = []
        for res_item in results:
            if res_item and res_item.get("status") == "success": # Check if res_item is not None
                row = {
                    "Embedding": res_item["config"].get("embedding", subject_configs.DEFAULT_EMBEDDING_MODEL.value),
                    "Vector Store": res_item["config"].get("vector_store"),
                    "Reranker": res_item["config"].get("reranker"),
                    "LLM": res_item["config"].get("llm"),
                    **res_item.get("metrics", {})
                }
                df_data.append(row)
        
        df = pd.DataFrame(df_data)
        return results, df

def run_pipeline_with_config(
    file_path: str,
    user_query: str,
    ground_truth: str,
    embedding_model_enum: EmbeddingModelType, # This is fixed to DEFAULT_EMBEDDING_MODEL in initialize_pipeline
    vector_store_enum: VectorStoreType,
    reranker_enum: RerankerModelType,
    llm_enum: LLMModelType,
    chunking_strategy_enum: ChunkingStrategyType,
    hybrid_alpha: Optional[float] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    top_k: Optional[int] = None,
    evaluation_backend_type: Optional[EvaluationBackendType] = None, # Added
    evaluation_metrics: Optional[List[str]] = None # Added
) -> Dict[str, Any]:
    """
    Run a single pipeline configuration and return results.
    
    Args:
        file_path (str): Path to the document file
        user_query (str): User's question
        ground_truth (str): Expected answer for evaluation
        embedding_model_enum (EmbeddingModelType): Embedding model to use (fixed in initialize_pipeline)
        vector_store_enum (VectorStoreType): Vector store to use
        reranker_enum (RerankerModelType): Reranker to use
        llm_enum (LLMModelType): LLM to use
        chunking_strategy_enum (ChunkingStrategyType): Chunking strategy to use
        hybrid_alpha (Optional[float]): Hybrid search alpha. Defaults to subject_configs.DEFAULT_HYBRID_ALPHA if None.
        chunk_size (Optional[int]): Chunk size. Defaults to subject_configs.DEFAULT_CHUNK_SIZE if None.
        chunk_overlap (Optional[int]): Chunk overlap. Defaults to subject_configs.DEFAULT_CHUNK_OVERLAP if None.
        top_k (Optional[int]): Top K results. Defaults to subject_configs.DEFAULT_TOP_K if None.
        evaluation_backend_type (Optional[EvaluationBackendType]): Backend for evaluation.
        evaluation_metrics (Optional[List[str]]): Specific metrics for evaluation.
        
    Returns:
        Dict[str, Any]: Results including response, metrics, and configuration
    """
    # Use defaults from subject_configs if specific values are not provided
    final_hybrid_alpha = hybrid_alpha if hybrid_alpha is not None else subject_configs.DEFAULT_HYBRID_ALPHA
    final_chunk_size = chunk_size if chunk_size is not None else subject_configs.DEFAULT_CHUNK_SIZE
    final_chunk_overlap = chunk_overlap if chunk_overlap is not None else subject_configs.DEFAULT_CHUNK_OVERLAP
    final_top_k = top_k if top_k is not None else subject_configs.DEFAULT_TOP_K
    # Evaluation defaults are handled within initialize_pipeline if None is passed

    try:
        pipeline = initialize_pipeline(
            file_path=file_path,
            vector_store_enum=vector_store_enum,
            reranker_enum=reranker_enum,
            llm_enum=llm_enum,
            chunking_strategy_enum=chunking_strategy_enum,
            hybrid_alpha=final_hybrid_alpha,
            chunk_size=final_chunk_size,
            chunk_overlap=final_chunk_overlap,
            top_k=final_top_k,
            evaluation_backend_type=evaluation_backend_type, # Pass through
            evaluation_metrics=evaluation_metrics # Pass through
        )
        
        if not pipeline:
            return {
                "status": "error",
                "error": "Failed to initialize pipeline",
                "config": {
                    "embedding": embedding_model_enum.value,
                    "vector_store": vector_store_enum.value,
                    "reranker": reranker_enum.value,
                    "llm": llm_enum.value,
                    "chunking": chunking_strategy_enum.value
                }
            }

        # Run query and get non-streaming response
        response_text, contexts, metrics_from_run = pipeline.run(user_query)
        
        # Get evaluation metrics if ground truth is provided
        evaluation_metrics = {}
        if ground_truth:
            # evaluate_response in RAGPipeline expects: query, response, contexts, ground_truth
            # It uses self.llm and self.last_metrics internally.
            evaluation_metrics = pipeline.evaluate_response(query=user_query, response=response_text, contexts=contexts, ground_truth=ground_truth)
        else:
            evaluation_metrics = metrics_from_run # If no ground truth, use metrics from the run itself

        return {
            "status": "success",
            "response": response_text,
            "metrics": evaluation_metrics, # This now contains RAGAS scores + performance metrics
            "contexts": contexts, # Adding contexts to the output
            "config": {
                "embedding": embedding_model_enum.value,
                "vector_store": vector_store_enum.value,
                "reranker": reranker_enum.value,
                "llm": llm_enum.value,
                "chunking": chunking_strategy_enum.value
            }
        }

    except Exception as e:
        logging.error(f"Error running pipeline: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "config": {
                "embedding": embedding_model_enum.value,
                "vector_store": vector_store_enum.value,
                "reranker": reranker_enum.value,
                "llm": llm_enum.value,
                "chunking": chunking_strategy_enum.value
            }
        }

def run_all_permutations(
    file_path: str,
    user_query: str,
    ground_truth: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    hybrid_alpha: float,
    chunking_strategy_enum: ChunkingStrategyType,
    evaluation_backend_type: EvaluationBackendType,
    evaluation_metrics: List[str]
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """
    Run all permutations of models and return results as a dataframe.
    Uses ExperimentRunner to execute permutations.
    """
    
    # Define model components to permute over
    # Embedding model is fixed by subject_configs.DEFAULT_EMBEDDING_MODEL via PipelineBuilder
    perm_vector_stores = [VectorStoreType.CHROMA, VectorStoreType.FAISS, VectorStoreType.HYBRID]
    perm_rerankers = list(set([RerankerModelType.NONE, RerankerModelType.COHERE_V3, subject_configs.DEFAULT_RERANKER_MODEL]))
    perm_llms = list(set([LLMModelType.CLAUDE_37_SONNET, LLMModelType.MISTRAL_LARGE, subject_configs.DEFAULT_LLM_MODEL]))

    # Generate all configuration dictionaries
    configurations = []
    for vs_enum, rer_enum, llm_enum in itertools.product(perm_vector_stores, perm_rerankers, perm_llms):
        configurations.append({
            "vector_store_enum": vs_enum,
            "reranker_enum": rer_enum,
            "llm_enum": llm_enum
            # Other parameters like chunk_size, top_k, etc., are taken from the ExperimentRunner's init or function call
        })

    # Initialize ExperimentRunner
    experiment_runner = ExperimentRunner(
        file_path=file_path,
        user_query=user_query,
        ground_truth=ground_truth,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        hybrid_alpha=hybrid_alpha,
        chunking_strategy_enum=chunking_strategy_enum,
        evaluation_backend_type=evaluation_backend_type,
        evaluation_metrics=evaluation_metrics
    )

    return experiment_runner.run_permutations(configurations)