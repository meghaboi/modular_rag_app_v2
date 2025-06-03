from typing import Dict, Any  # Removed List, Optional

# Unused langchain imports removed below
from langsmith.evaluation import run_evaluator


class LangSmithRAGAdapter:
    """Adapter to make RAG pipeline compatible with LangSmith tracing and evaluation"""

    def __init__(self, pipeline):
        """Initialize with a RAG pipeline"""
        self.pipeline = pipeline

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single input through the RAG pipeline"""
        query = inputs["query"]

        # Process the query using the RAG pipeline
        response, contexts = self.pipeline.process_query(query)

        # Return the results in the format expected by LangSmith
        return {"response": response, "contexts": contexts, "query": query}

    def create_from_custom_evaluators(self, evaluator_class):
        """Create custom evaluators for LangSmith from RAGEvaluators class"""
        custom_evaluators = []

        # Map static methods to LangSmith evaluators
        evaluator_mappings = {
            "relevance": evaluator_class.answer_relevance_evaluator,
            "retrieval_relevance": evaluator_class.context_relevance_evaluator,
            "grounded": evaluator_class.groundedness_evaluator,
            "faithfulness": evaluator_class.faithfulness_evaluator,  # Corrected duplicate key "relevance" to "faithfulness"
        }

        for name, func in evaluator_mappings.items():
            # Create a run_evaluator for each evaluation function
            # Don't put this in a tuple with the name
            custom_evaluators.append(run_evaluator(func))

        return custom_evaluators
