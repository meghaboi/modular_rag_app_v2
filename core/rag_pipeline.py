from typing import List, Dict, Any, Optional, Tuple  # Removed Callable
from .embedding_models import EmbeddingModel
from .rerankers import Reranker
from .vector_stores import VectorStore
from .llm_models import StreamingLLM
from ..token_utils import (
    TokenCounter,
    TokenCostManager,
)  # Assuming token_utils is one level up from core
import logging
import time

# Removed re, abc, TfidfVectorizer, cosine_similarity, BM25Okapi, numpy
# as they are now in the new files (chunking.py, search_strategies.py)
# Or were only used by classes that were moved.

# Import for strategies and factory
from .chunking import ChunkingStrategy, ChunkingStrategyFactory

# It seems RAGPipeline uses the factory, not individual strategies directly for init.
# If specific strategies were instantiated directly, they'd need to be imported too.

# Import for HybridSearch (if RAGPipeline or other components here were to use it directly)
# from .search_strategies import HybridSearch # Not directly used by RAGPipeline class itself


class RAGPipeline:
    """RAG Pipeline that combines all components with streaming support"""

    def __init__(
        self,
        embedding_model,
        vector_store,
        llm,
        reranker=None,
        top_k=3,
        chunking_strategy=None,
        chunk_size=1000,
        chunk_overlap=200,
        evaluation_mode=False,
    ):
        """Initialize the RAG pipeline with the selected components"""
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.reranker = reranker
        self.llm = llm
        self.top_k = top_k
        self.documents = []
        # Ensure chunking_strategy is an instance of ChunkingStrategy from chunking.py
        # If it's a name, it should be resolved via the factory.
        # Assuming chunking_strategy passed is already an instantiated strategy object
        # or the name of a strategy to be fetched by the factory.
        if isinstance(chunking_strategy, str):
            self.chunking_strategy = ChunkingStrategyFactory.get_strategy(
                chunking_strategy
            )
        else:
            self.chunking_strategy = chunking_strategy

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.evaluation_mode = evaluation_mode
        # self.last_evaluation_scores = None  # Removed: Store the last evaluation scores
        self.last_metrics = {}  # Store the last performance metrics
        self.last_llm_usage = None  # Store usage from the last LLM call

    def initialize(self, file_path: str) -> None:
        """Initialize the pipeline with a document file"""
        self.index_documents(file_path, self.chunk_size, self.chunk_overlap)

    def index_documents(
        self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> None:
        """Index documents from a file"""
        # Read file
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Split text into chunks using the selected strategy
        chunks = self.chunking_strategy.chunk_text(text, chunk_size, chunk_overlap)
        self.documents = chunks

        # Get embeddings for chunks
        embeddings = self.embedding_model.embed_documents(chunks)

        # Add chunks to vector store
        self.vector_store.add_documents(chunks, embeddings)

    def retrieve_context(self, query: str) -> list:
        """Retrieve relevant contexts for a given query"""
        # Get query embedding
        query_embedding = self.embedding_model.embed_query(query)

        # Retrieve documents - check if vector store supports hybrid search
        # This part assumes vector_store might have plain search or hybrid search.
        # HybridSearch class itself is now in search_strategies.py.
        # If vector_store is an instance of HybridSearch or uses it, it's encapsulated there.

        # The logic for calling vector_store.search remains, assuming the VectorStore interface
        # is consistent whether it's a simple vector store or one using HybridSearch.
        if (
            hasattr(self.vector_store, "search")
            and "query" in self.vector_store.search.__code__.co_varnames
            and "query_embedding" in self.vector_store.search.__code__.co_varnames
        ):  # More robust check
            # Assumes a hybrid-capable search method signature
            retrieved_docs = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=self.top_k,
                query=query,  # Keyword query for hybrid
            )
        else:
            # Standard vector search (embedding only)
            retrieved_docs = self.vector_store.search(query_embedding, self.top_k)

        retrieved_texts = [
            doc[0] for doc in retrieved_docs
        ]  # Assuming doc is (text, score)

        # Apply reranking if available
        if self.reranker and retrieved_texts:
            # Reranker might take List[str] or List[Tuple[str, float]]
            # Assuming it takes List[str] for now.
            reranked_docs = self.reranker.rerank(query, retrieved_texts)
            # Assuming reranker returns List[Tuple[str, float, Optional[Any]]] or similar
            # and we need to extract text.
            # If reranker returns [(text, score)], then this is fine.
            # If it returns complex objects, adjust extraction.
            # For now, assuming reranked_docs are already in a format where doc[0] is text.
            retrieved_texts = [
                doc[0] for doc in reranked_docs[: self.top_k]
            ]  # Apply top_k after reranking

        return retrieved_texts

    def run(self, query: str) -> Tuple[str, List[str], Dict[str, Any]]:
        """Process a query and return the response, contexts, and metrics (non-streaming)"""
        start_time = time.time()

        # Get context
        retrieved_texts = self.retrieve_context(query)

        # Combine retrieved documents
        context_str = "\n\n".join(retrieved_texts)

        # Generate response
        response_text, usage_info = self.llm.generate(
            query, context_str, evaluation_mode=self.evaluation_mode
        )
        self.last_llm_usage = usage_info

        total_time = time.time() - start_time

        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        if usage_info:
            prompt_tokens = usage_info.get("prompt_tokens", 0)
            completion_tokens = usage_info.get("completion_tokens", 0)
            total_tokens = usage_info.get("total_tokens", 0)
        else:  # Fallback if usage_info is None, though unlikely with new changes
            logging.warning(
                "LLM usage_info was None in RAGPipeline.run. Token counts may be estimated."
            )
            # Fallback to TokenCounter if usage_info is not available
            try:
                token_counter = TokenCounter(model_name=self.llm.get_model_name())
                prompt_tokens = token_counter.count_tokens(query + context_str)
                completion_tokens = token_counter.count_tokens(response_text)
                total_tokens = prompt_tokens + completion_tokens
            except Exception as e:
                logging.error(f"TokenCounter fallback failed in RAGPipeline.run: {e}")
        model_name = self.llm.get_model_name()
        calculated_cost = TokenCostManager.calculate_cost(
            model_name, prompt_tokens, completion_tokens
        )

        self.last_metrics = {
            "total_time": total_time,
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "llm_cost": calculated_cost if calculated_cost is not None else 0.0,
        }

        return response_text, retrieved_texts, self.last_metrics

    def stream_run(self, query: str):
        """Process a query and stream the response

        In evaluation mode, this will use non-streaming to maintain consistency
        """
        # If we're in evaluation mode, use the non-streaming method instead
        if self.evaluation_mode:
            response_text, _, _ = self.run(
                query
            )  # Discard contexts and metrics for streaming yield
            yield response_text
            return

        # Get context
        retrieved_texts = self.retrieve_context(query)

        # Combine retrieved documents
        context = "\n\n".join(retrieved_texts)

        # Stream response
        for chunk in self.llm.stream_generate(
            query, context, evaluation_mode=self.evaluation_mode
        ):
            # Ensure we're not returning None values from our generator
            if chunk is not None:
                yield chunk
            else:
                logging.warning("LLM returned None chunk, skipping")

    def process_query(self, query: str) -> Tuple[str, List[str], Dict[str, Any]]:
        """Process a query and return the response, retrieved contexts, and metrics"""
        try:
            start_time = time.time()

            # Get contexts from vector store
            contexts = self.retrieve_context(query)

            # Generate response
            response_text, usage_info = self.llm.generate(
                query, context="\n".join(contexts), evaluation_mode=self.evaluation_mode
            )
            self.last_llm_usage = usage_info

            # Calculate total time
            total_time = time.time() - start_time

            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0

            if usage_info:
                prompt_tokens = usage_info.get("prompt_tokens", 0)
                completion_tokens = usage_info.get("completion_tokens", 0)
                total_tokens = usage_info.get("total_tokens", 0)
            else:  # Fallback if usage_info is None, though unlikely with new changes
                logging.warning(
                    "LLM usage_info was None in process_query. Token counts will be zero."
                )
                token_counter = TokenCounter(model_name=self.llm.get_model_name())
                prompt_tokens = token_counter.count_tokens(query + "\n".join(contexts))
                completion_tokens = token_counter.count_tokens(response_text)
                total_tokens = prompt_tokens + completion_tokens

            model_name = self.llm.get_model_name()
            calculated_cost = TokenCostManager.calculate_cost(
                model_name, prompt_tokens, completion_tokens
            )

            # Store metrics
            self.last_metrics = {
                "total_time": total_time,
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "llm_cost": calculated_cost if calculated_cost is not None else 0.0,
            }

            return response_text, contexts, self.last_metrics

        except Exception as e:
            logging.error(f"Error processing query: {e}")
            raise

    # evaluate_response method removed.
