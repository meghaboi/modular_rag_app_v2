from typing import List, Dict, Any, Optional, Tuple
import re
from rank_bm25 import BM25Okapi
import numpy as np  
from utils.token_utils import TokenCostManager
import logging
import time
from dataclasses import dataclass

class HybridSearch:
    """Combines dense vector search with sparse keyword search (BM25)"""
    
    def __init__(self, alpha: float = 0.5):
        """
        Initialize hybrid search
        
        Args:
            alpha: Weight for vector search scores (1-alpha = weight for BM25)
        """
        self.alpha = alpha
        self.documents = []
        self.bm25 = None
        self.doc_embeddings = None
        
    def index_documents(self, documents: List[str], embeddings: List[List[float]]) -> None:
        """Index documents for both vector search and BM25"""
        self.documents = documents
        self.doc_embeddings = np.array(embeddings)
        
        # Tokenize documents for BM25
        tokenized_docs = [self._tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25"""
        # Convert to lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\w+', text)
        return tokens
    
    def search(self, query: str, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Perform hybrid search using both vector similarity and BM25
        
        Args:
            query: Text query for keyword search
            query_embedding: Vector embedding of the query
            top_k: Number of results to return
            
        Returns:
            List of tuples with (document, score)
        """
        if not self.documents or len(self.documents) == 0:
            return []
        
        # Vector search scores
        vector_scores = self._vector_search(query_embedding)
        
        # BM25 search scores
        bm25_scores = self._bm25_search(query)
        
        # Normalize scores to [0, 1] range
        vector_scores_norm = self._normalize_scores(vector_scores)
        bm25_scores_norm = self._normalize_scores(bm25_scores)
        
        # Combine scores with alpha weighting
        combined_scores = self.alpha * vector_scores_norm + (1 - self.alpha) * bm25_scores_norm
        
        # Get top k results
        top_indices = np.argsort(-combined_scores)[:top_k]
        
        results = [(self.documents[i], combined_scores[i]) for i in top_indices]
        return results
    
    def _vector_search(self, query_embedding: List[float]) -> np.ndarray:
        """Calculate vector similarity scores for all documents"""
        query_embedding = np.array(query_embedding)
        
        # Calculate cosine similarity
        # Normalize vectors for cosine similarity
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm
            
        # Calculate dot product for normalized vectors (equal to cosine similarity)
        doc_norms = np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True)
        normalized_embeddings = np.divide(self.doc_embeddings, doc_norms, 
                                         where=doc_norms != 0)
        
        similarities = np.dot(normalized_embeddings, query_embedding)
        return similarities
    
    def _bm25_search(self, query: str) -> np.ndarray:
        """Calculate BM25 scores for all documents"""
        query_tokens = self._tokenize(query)
        scores = np.array(self.bm25.get_scores(query_tokens))
        return scores
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range"""
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score == min_score:
            return np.ones_like(scores)
            
        normalized = (scores - min_score) / (max_score - min_score)
        return normalized

class RAGPipelineError(Exception):
    """Base exception class for RAG pipeline errors"""
    pass

class RAGPipelineInitializationError(RAGPipelineError):
    """Raised when pipeline initialization fails"""
    pass

class RAGPipelineExecutionError(RAGPipelineError):
    """Raised when pipeline execution fails"""
    pass

class RAGPipelineEvaluationError(RAGPipelineError):
    """Raised when pipeline evaluation fails"""
    pass

@dataclass
class PipelineMetrics:
    """Metrics for pipeline execution"""
    total_time: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    llm_cost: float
    evaluation_scores: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        metrics_dict = {
            "total_time": self.total_time,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "llm_cost": self.llm_cost
        }
        if self.evaluation_scores:
            metrics_dict.update(self.evaluation_scores)
        return metrics_dict

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
            from utils.evaluator import EvaluatorFactory
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