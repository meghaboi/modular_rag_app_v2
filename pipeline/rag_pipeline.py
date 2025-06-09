from typing import List, Dict, Any, Optional, Tuple
import re
from rank_bm25 import BM25Okapi
import numpy as np  
from utils.token_utils import TokenCostManager
import logging
import time
from dataclasses import dataclass
from components.exceptions import (
    RAGPipelineInitializationError,
    RAGPipelineExecutionError,
    RAGPipelineEvaluationError
)
from pipeline.models.metrics import PipelineMetrics

class RAGPipeline:
    """RAG Pipeline that combines all components with streaming support"""
    
    def __init__(self, embedding_model, vector_store, 
                 llm, reranker=None, top_k=3,
                 chunking_strategy=None, chunk_size=1000, 
                 chunk_overlap=200, evaluation_mode=False):
        """
        Initialize the RAG pipeline with the selected components.
        
        Args:
            embedding_model: Model for generating embeddings
            vector_store: Vector store for document storage and retrieval
            llm: Language model for response generation
            reranker: Optional reranker for improving retrieval quality
            top_k: Number of documents to retrieve
            chunking_strategy: Strategy for document chunking
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            evaluation_mode: Whether to run in evaluation mode
            
        Raises:
            RAGPipelineInitializationError: If initialization fails
        """
        try:
            self.embedding_model = embedding_model
            self.vector_store = vector_store
            self.reranker = reranker
            self.llm = llm
            self.top_k = top_k
            self.documents = []
            self.chunking_strategy = chunking_strategy
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.evaluation_mode = evaluation_mode
            self._metrics = PipelineMetrics(0.0, 0, 0, 0, 0.0)
        except Exception as e:
            raise RAGPipelineInitializationError(f"Failed to initialize pipeline: {str(e)}")
    
    def index_documents(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        """
        Index documents from a file.
        
        Args:
            file_path: Path to the document file
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            
        Raises:
            RAGPipelineExecutionError: If indexing fails
        """
        try:
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Split text into chunks using the selected strategy
            chunks = self.chunking_strategy.chunk_text(text, chunk_size, chunk_overlap)
            self.documents = chunks
            
            # Get embeddings for chunks
            embeddings = self.embedding_model.embed_documents(chunks)
            
            # Add chunks to vector store
            self.vector_store.add_documents(chunks, embeddings)
        except Exception as e:
            raise RAGPipelineExecutionError(f"Failed to index documents: {str(e)}")
    
    def retrieve_context(self, query: str) -> List[str]:
        """
        Retrieve relevant contexts for a given query.
        
        Args:
            query: The query to retrieve contexts for
            
        Returns:
            List[str]: Retrieved context texts
            
        Raises:
            RAGPipelineExecutionError: If retrieval fails
        """
        try:
            # Get query embedding
            query_embedding = self.embedding_model.embed_query(query)
            
            # Retrieve documents - check if vector store supports hybrid search
            if hasattr(self.vector_store, 'search') and 'query' in self.vector_store.search.__code__.co_varnames:
                retrieved_docs = self.vector_store.search(query_embedding, self.top_k, query=query)
            else:
                retrieved_docs = self.vector_store.search(query_embedding, self.top_k)
                
            retrieved_texts = [doc[0] for doc in retrieved_docs]
            
            # Apply reranking if available
            if self.reranker and retrieved_texts:
                reranked_docs = self.reranker.rerank(query, retrieved_texts)
                # Select top 5 chunks after reranking
                reranked_docs = reranked_docs[:5]
                retrieved_texts = [doc[0] for doc in reranked_docs]
            
            return retrieved_texts
        except Exception as e:
            raise RAGPipelineExecutionError(f"Failed to retrieve context: {str(e)}")
    
    def _calculate_metrics(self, start_time: float, usage_info: Dict[str, Any]) -> PipelineMetrics:
        """Calculate pipeline metrics from execution data"""
        total_time = time.time() - start_time
        
        prompt_tokens = usage_info.get('prompt_tokens', 0)
        completion_tokens = usage_info.get('completion_tokens', 0)
        total_tokens = usage_info.get('total_tokens', 0)
        
        model_name = self.llm.get_model_name()
        calculated_cost = TokenCostManager.calculate_cost(model_name, prompt_tokens, completion_tokens)
        
        return PipelineMetrics(
            total_time=total_time,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            total_tokens=total_tokens,
            llm_cost=calculated_cost if calculated_cost is not None else 0.0
        )
    
    def run(self, query: str, system_prompt_override: Optional[str] = None) -> Tuple[str, List[str], Dict[str, Any]]:
        """
        Process a query and return the response, contexts, and metrics (non-streaming).
        
        Args:
            query: The query to process
            system_prompt_override: Optional system prompt to override the default
            
        Returns:
            Tuple[str, List[str], Dict[str, Any]]: Response text, contexts, and metrics
            
        Raises:
            RAGPipelineExecutionError: If execution fails
        """
        try:
            start_time = time.time()
            
            # Get context
            retrieved_texts = self.retrieve_context(query)
            
            # Combine retrieved documents
            context_str = "\n\n".join(retrieved_texts)
            
            # Generate response
            response_text, usage_info = self.llm.generate(
                prompt=query,
                context=context_str,
                evaluation_mode=self.evaluation_mode,
                system_prompt_override=system_prompt_override
            )
            
            # Calculate metrics
            self._metrics = self._calculate_metrics(start_time, usage_info)
            
            return response_text, retrieved_texts, self._metrics.to_dict()
        except Exception as e:
            raise RAGPipelineExecutionError(f"Failed to run pipeline: {str(e)}")
    
    def stream_run(self, query: str, system_prompt_override: Optional[str] = None):
        """
        Process a query and stream the response.
        
        Args:
            query: The query to process
            system_prompt_override: Optional system prompt to override the default
            
        Yields:
            str: Response chunks
            
        Raises:
            RAGPipelineExecutionError: If execution fails
        """
        try:
            # If we're in evaluation mode, use the non-streaming method instead
            if self.evaluation_mode:
                # Pass system_prompt_override to self.run if in evaluation mode
                response_text, _, _ = self.run(query, system_prompt_override=system_prompt_override)
                yield response_text
                return
                
            # Get context
            retrieved_texts = self.retrieve_context(query)
            
            # Combine retrieved documents
            context = "\n\n".join(retrieved_texts)
            
            # Stream response
            for chunk in self.llm.stream_generate(
                prompt=query,
                context=context,
                evaluation_mode=self.evaluation_mode,
                system_prompt_override=system_prompt_override
            ):
                if chunk is not None:
                    yield chunk
                else:
                    logging.warning("LLM returned None chunk, skipping")
        except Exception as e:
            raise RAGPipelineExecutionError(f"Failed to stream response: {str(e)}")
    
    def evaluate_response(self, query: str, response: str, contexts: List[str], ground_truth: str) -> Dict[str, float]:
        """
        Evaluate the response using RAGAS metrics.
        
        Args:
            query: The original query
            response: The generated response
            contexts: The retrieved contexts
            ground_truth: The expected answer
            
        Returns:
            Dict[str, float]: Evaluation scores
            
        Raises:
            RAGPipelineEvaluationError: If evaluation fails
        """
        try:
            from models.evaluator import EvaluatorFactory
            from utils.enums import EvaluationBackendType, EvaluationMetricType
            
            # Use RAGAS_V2 for consistency with permutation evaluations
            evaluator = EvaluatorFactory.create_evaluator(
                EvaluationBackendType.RAGAS_V2, 
                EvaluationMetricType.get_metrics_for_backend(EvaluationBackendType.RAGAS_V2)
            )
            
            scores = evaluator.evaluate(
                query=query,
                response=response,
                contexts=contexts,
                ground_truth=ground_truth,
                cost=self._metrics.llm_cost
            )
            
            # Update metrics with evaluation scores
            self._metrics.evaluation_scores = scores
            
            return self._metrics.to_dict()
        except Exception as e:
            raise RAGPipelineEvaluationError(f"Failed to evaluate response: {str(e)}")
    
    def get_metrics(self) -> PipelineMetrics:
        """Get the current pipeline metrics"""
        return self._metrics