import logging
from typing import Dict, Any, List
from src.config.enums import (
    EmbeddingModelType,
    RerankerModelType,
    LLMModelType,
    VectorStoreType,
    ChunkingStrategyType
)
from src.core.pipeline_init import initialize_pipeline

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
    """Run the RAG pipeline with a specific configuration"""
    try:
        # Initialize pipeline
        pipeline = initialize_pipeline(
            file_path,
            embedding_model_enum,
            vector_store_enum,
            reranker_enum,
            llm_enum,
            chunking_strategy_enum,
            hybrid_alpha,
            chunk_size,
            chunk_overlap,
            top_k
        )
        
        if not pipeline:
            return {
                "error": "Failed to initialize pipeline",
                "score": 0.0
            }
        
        # Run query
        response = pipeline.query(user_query)
        
        # Evaluate response
        evaluator = pipeline.evaluator
        score = evaluator.evaluate(response["answer"], ground_truth)
        
        return {
            "embedding_model": embedding_model_enum.value,
            "vector_store": vector_store_enum.value,
            "reranker": reranker_enum.value,
            "llm": llm_enum.value,
            "chunking_strategy": chunking_strategy_enum.value,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "top_k": top_k,
            "hybrid_alpha": hybrid_alpha,
            "answer": response["answer"],
            "score": score
        }
        
    except Exception as e:
        logging.error(f"Error running pipeline: {e}", exc_info=True)
        return {
            "error": str(e),
            "score": 0.0
        }

def run_all_permutations(
    file_path: str,
    user_query: str,
    ground_truth: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    hybrid_alpha: float,
    chunking_strategy_enum: ChunkingStrategyType
) -> List[Dict[str, Any]]:
    """Run the pipeline with all possible model combinations"""
    results = []
    
    # Define model combinations to test
    embedding_models = [EmbeddingModelType.MISTRAL, EmbeddingModelType.OPENAI]
    vector_stores = [VectorStoreType.CHROMA, VectorStoreType.HYBRID]
    rerankers = [RerankerModelType.NONE, RerankerModelType.COHERE_V3]
    llm_models = [LLMModelType.CLAUDE_37_SONNET, LLMModelType.OPENAI_GPT4]
    
    # Run all combinations
    for emb_model in embedding_models:
        for vec_store in vector_stores:
            for reranker in rerankers:
                for llm in llm_models:
                    result = run_pipeline_with_config(
                        file_path,
                        user_query,
                        ground_truth,
                        emb_model,
                        vec_store,
                        reranker,
                        llm,
                        chunking_strategy_enum,
                        hybrid_alpha,
                        chunk_size,
                        chunk_overlap,
                        top_k
                    )
                    results.append(result)
    
    return results 