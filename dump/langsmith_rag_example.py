import os
from dotenv import load_dotenv
from langsmith_evaluator import LangSmithRAGEvaluator
from langsmith_rag_adapter import LangSmithRAGAdapter
from langsmith_custom_evaluators import RAGEvaluators
from rag_pipeline import RAGPipeline
from embedding_models import EmbeddingModelFactory
from vector_stores import VectorStoreFactory
from llm_models import LLMFactory
from langsmith import Client

def run_example_evaluation():
    """Run an example RAG evaluation using LangSmith"""
    # Load environment variables
    load_dotenv()
    
    # Check if LangSmith API key is set
    if not os.environ.get("LANGCHAIN_API_KEY"):
        print("Error: LANGCHAIN_API_KEY not set. Please set it in your .env file.")
        return
    
    # Create a simple RAG pipeline
    embedding_model = EmbeddingModelFactory.create_model("OpenAI")
    vector_store = VectorStoreFactory.create_store("FAISS")
    llm = LLMFactory.create_llm("OpenAI GPT-4")
    
    pipeline = RAGPipeline(
        embedding_model=embedding_model,
        vector_store=vector_store,
        reranker=None,
        llm=llm,
        top_k=3,
        chunking_strategy=None  # Use default strategy
    )
    
    # Index a sample document
    sample_text = """
    LangSmith is a platform for debugging, testing, evaluating, and monitoring LLM applications.
    It helps developers iterate faster on prompt engineering and evaluation of their LLM applications.
    LangSmith provides tools for tracing, comparing runs, and evaluating LLM outputs systematically.
    It integrates with LangChain but can be used with any LLM application framework.
    """
    
    # Write sample text to a temporary file
    with open("sample.txt", "w") as f:
        f.write(sample_text)
    
    # Index the sample document
    pipeline.index_documents("sample.txt", chunk_size=500, chunk_overlap=100)
    
    # Create LangSmith adapter
    langsmith_adapter = LangSmithRAGAdapter(pipeline)
    
    # Create LangSmith evaluator
    evaluator = LangSmithRAGEvaluator()
    
    # Create dataset
    dataset_name = "sample_rag_evaluation"
    try:
        evaluator.create_dataset(dataset_name, "Sample RAG evaluation dataset")
        print(f"Created dataset: {dataset_name}")
    except Exception as e:
        if "already exists" in str(e):
            print(f"Using existing dataset: {dataset_name}")
        else:
            raise e
    
    # Create example
    query = "What is LangSmith used for?"
    response, contexts = pipeline.process_query(query)
    
    example = evaluator.generate_evaluation_example(
        query=query,
        response=response,
        contexts=contexts
    )
    
    # Add example to dataset
    evaluator.add_examples(dataset_name, [example])
    print("Added example to dataset")
    
    # Define custom evaluators
    custom_evaluators = {
        "answer_relevance": RAGEvaluators.answer_relevance_evaluator,
        "context_relevance": RAGEvaluators.context_relevance_evaluator,
        "groundedness": RAGEvaluators.groundedness_evaluator,
        "faithfulness": RAGEvaluators.faithfulness_evaluator
    }
    
    # Run evaluation
    print("Running LangSmith evaluation...")
    results = evaluator.evaluate_rag_pipeline(
        dataset_name=dataset_name,
        pipeline=langsmith_adapter,
        evaluators=["answer_relevance", "groundedness"],
        custom_metrics=custom_evaluators
    )
    
    # Print results
    print("Evaluation complete!")
    print(f"View detailed results in LangSmith: https://smith.langchain.com/projects/rag-evaluation-results")
    
    # Clean up
    os.remove("sample.txt")

if __name__ == "__main__":
    run_example_evaluation()