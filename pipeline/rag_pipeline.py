from typing import List, Dict, Any, Optional, Tuple
from utils.token_utils import TokenCostManager
import logging
import time
from dataclasses import dataclass
from pipeline.components.exceptions import (
    RAGPipelineInitializationError,
    RAGPipelineExecutionError,
    RAGPipelineEvaluationError
)
from pipeline.models.metrics import PipelineMetrics

# Constants
DEFAULT_TOP_K = 3
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
RERANKING_TOP_K = 5
CONTEXT_SEPARATOR = "\n\n"

@dataclass
class PipelineConfiguration:
    """Configuration parameters for RAG Pipeline."""
    embedding_model: Any
    vector_store: Any
    llm: Any
    reranker: Optional[Any] = None
    top_k: int = DEFAULT_TOP_K
    chunking_strategy: Optional[Any] = None
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    evaluation_mode: bool = False

class TimingLogger:
    """Utility class for consistent timing and logging."""

    @staticmethod
    def log_operation_time(operation_name: str, start_time: float) -> float:
        """Log the time taken for an operation and return the duration."""
        duration = time.time() - start_time
        logging.info(f"{operation_name} took {duration:.2f} seconds")
        return duration

class DocumentIndexer:
    """Handles document indexing operations."""

    def __init__(self, config: PipelineConfiguration):
        self._config = config

    def index_file(self, file_path: str, chunk_size: int, chunk_overlap: int) -> None:
        """Index documents from a file."""
        total_start_time = time.time()

        text = self._read_file(file_path)
        chunks = self._chunk_document(text, chunk_size, chunk_overlap)
        embeddings = self._generate_embeddings(chunks)
        self._store_embeddings(chunks, embeddings)

        TimingLogger.log_operation_time("Total document indexing", total_start_time)

    def _read_file(self, file_path: str) -> str:
        """Read text from file with timing."""
        start_time = time.time()
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        TimingLogger.log_operation_time("File reading", start_time)
        return text

    def _chunk_document(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Chunk document with timing."""
        start_time = time.time()
        chunks = self._config.chunking_strategy.chunk_text(text, chunk_size, chunk_overlap)
        TimingLogger.log_operation_time("Document chunking", start_time)
        return chunks

    def _generate_embeddings(self, chunks: List[str]) -> Any:
        """Generate embeddings with timing."""
        start_time = time.time()
        embeddings = self._config.embedding_model.embed_documents(chunks)
        TimingLogger.log_operation_time("Embedding generation", start_time)
        return embeddings

    def _store_embeddings(self, chunks: List[str], embeddings: Any) -> None:
        """Store embeddings with timing."""
        start_time = time.time()
        self._config.vector_store.add_documents(chunks, embeddings)
        TimingLogger.log_operation_time("Vector storage", start_time)

class ContextRetriever:
    """Handles context retrieval operations."""

    def __init__(self, config: PipelineConfiguration):
        self._config = config

    def retrieve_contexts(self, query: str) -> List[str]:
        """Retrieve relevant contexts for a query."""
        query_embedding = self._config.embedding_model.embed_query(query)
        retrieved_docs = self._search_vector_store(query_embedding, query)
        retrieved_texts = [doc[0] for doc in retrieved_docs]

        return self._apply_reranking(query, retrieved_texts)

    def _search_vector_store(self, query_embedding: Any, query: str) -> List[Any]:
        """Search vector store with appropriate method."""
        if self._supports_hybrid_search():
            return self._config.vector_store.search(
                query_embedding, self._config.top_k, query=query
            )
        else:
            return self._config.vector_store.search(query_embedding, self._config.top_k)

    def _supports_hybrid_search(self) -> bool:
        """Check if vector store supports hybrid search."""
        return (hasattr(self._config.vector_store, 'search') and
                'query' in self._config.vector_store.search.__code__.co_varnames)

    def _apply_reranking(self, query: str, retrieved_texts: List[str]) -> List[str]:
        """Apply reranking if available."""
        if not self._config.reranker or not retrieved_texts:
            return retrieved_texts

        reranked_docs = self._config.reranker.rerank(query, retrieved_texts)
        reranked_docs = reranked_docs[:RERANKING_TOP_K]
        return [doc[0] for doc in reranked_docs]

class ResponseGenerator:
    """Handles response generation operations."""

    def __init__(self, config: PipelineConfiguration):
        self._config = config

    def generate_response(self, query: str, contexts: List[str],
                          system_prompt_override: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """Generate response from query and contexts."""
        start_time = time.time()
        context_str = self._combine_contexts(contexts)

        response_text, usage_info = self._config.llm.generate(
            prompt=query,
            context=context_str,
            evaluation_mode=self._config.evaluation_mode,
            system_prompt_override=system_prompt_override
        )

        TimingLogger.log_operation_time("Response generation", start_time)
        return response_text, usage_info

    def stream_response(self, query: str, contexts: List[str],
                        system_prompt_override: Optional[str] = None):
        """Stream response generation."""
        context_str = self._combine_contexts(contexts)

        for chunk in self._config.llm.stream_generate(
                prompt=query,
                context=context_str,
                evaluation_mode=self._config.evaluation_mode,
                system_prompt_override=system_prompt_override
        ):
            if chunk is not None:
                yield chunk
            else:
                logging.warning("LLM returned None chunk, skipping")

    def _combine_contexts(self, contexts: List[str]) -> str:
        """Combine contexts with timing."""
        start_time = time.time()
        context_str = CONTEXT_SEPARATOR.join(contexts)
        TimingLogger.log_operation_time("Context combination", start_time)
        return context_str

class PipelineEvaluator:
    """Handles pipeline evaluation operations."""

    def __init__(self, metrics: PipelineMetrics):
        self._metrics = metrics

    def evaluate_response(self, query: str, response: str, contexts: List[str],
                          ground_truth: str) -> Dict[str, float]:
        """Evaluate response using RAGAS metrics."""
        from models.evaluator import EvaluatorFactory
        from utils.enums import EvaluationBackendType, EvaluationMetricType

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

        self._metrics.evaluation_scores = scores
        return self._metrics.to_dict()

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
            self._config = PipelineConfiguration(
                embedding_model=embedding_model,
                vector_store=vector_store,
                llm=llm,
                reranker=reranker,
                top_k=top_k,
                chunking_strategy=chunking_strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                evaluation_mode=evaluation_mode
            )

            # Maintain backward compatibility with direct property access
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
            self._indexer = DocumentIndexer(self._config)
            self._retriever = ContextRetriever(self._config)
            self._generator = ResponseGenerator(self._config)
            self._evaluator = PipelineEvaluator(self._metrics)

        except Exception as e:
            raise RAGPipelineInitializationError(f"Failed to initialize pipeline: {str(e)}")

    def index_documents(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        """
        Index documents from a file using the configured chunking strategy.

        Args:
            file_path: Path to the document file
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks

        Raises:
            RAGPipelineExecutionError: If indexing fails
        """
        try:
            self._indexer.index_file(file_path, chunk_size, chunk_overlap)
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
            return self._retriever.retrieve_contexts(query)
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
            total_start_time = time.time()

            retrieved_texts = self._execute_retrieval_phase(query)
            response_text, usage_info = self._execute_generation_phase(
                query, retrieved_texts, system_prompt_override
            )
            self._execute_metrics_phase(total_start_time, usage_info)

            TimingLogger.log_operation_time("Total pipeline execution", total_start_time)
            return response_text, retrieved_texts, self._metrics.to_dict()

        except Exception as e:
            raise RAGPipelineExecutionError(f"Failed to run pipeline: {str(e)}")

    def _execute_retrieval_phase(self, query: str) -> List[str]:
        """Execute the context retrieval phase."""
        start_time = time.time()
        retrieved_texts = self.retrieve_context(query)
        TimingLogger.log_operation_time("Context retrieval", start_time)
        return retrieved_texts

    def _execute_generation_phase(self, query: str, contexts: List[str],
                                  system_prompt_override: Optional[str]) -> Tuple[str, Dict[str, Any]]:
        """Execute the response generation phase."""
        return self._generator.generate_response(query, contexts, system_prompt_override)

    def _execute_metrics_phase(self, start_time: float, usage_info: Dict[str, Any]) -> None:
        """Execute the metrics calculation phase."""
        metrics_start_time = time.time()
        self._metrics = self._calculate_metrics(start_time, usage_info)
        TimingLogger.log_operation_time("Metrics calculation", metrics_start_time)

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
            if self.evaluation_mode:
                response_text, _, _ = self.run(query, system_prompt_override=system_prompt_override)
                yield response_text
                return

            retrieved_texts = self.retrieve_context(query)

            for chunk in self._generator.stream_response(query, retrieved_texts, system_prompt_override):
                yield chunk

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
            return self._evaluator.evaluate_response(query, response, contexts, ground_truth)
        except Exception as e:
            raise RAGPipelineEvaluationError(f"Failed to evaluate response: {str(e)}")

    def get_metrics(self) -> PipelineMetrics:
        """Get the current pipeline metrics"""
        return self._metrics

    def get_config(self) -> Dict[str, Any]:
        """
        Get the current pipeline configuration.

        Returns:
            Dict[str, Any]: Dictionary containing the current configuration
        """
        return {
            'embedding_model': self._get_class_name(self.embedding_model),
            'vector_store': self._get_class_name(self.vector_store),
            'reranker': self._get_class_name(self.reranker) if self.reranker else None,
            'llm_model': self._get_class_name(self.llm),
            'chunking_strategy': self._get_class_name(self.chunking_strategy) if self.chunking_strategy else None,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'top_k': self.top_k
        }

    def _get_class_name(self, obj: Any) -> str:
        """Get class name safely."""
        return obj.__class__.__name__