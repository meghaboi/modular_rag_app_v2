
from typing import List, Dict, Any, Optional
import os
from langsmith import Client
from langsmith.evaluation import EvaluationResult, run_evaluator
from langsmith.schemas import Run, Example  # Changed RunTree to Run
import langchain
from langchain_openai import ChatOpenAI  # Import to create the model instance
from langchain.smith import RunEvalConfig
from langchain.schema import Document
from langsmith_rag_adapter import LangSmithRAGAdapter
import uuid

class LangSmithRAGEvaluator:
    """Class for evaluating RAG system performance using LangSmith"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the LangSmith RAG evaluator"""
        # Set up LangSmith API key if provided
        if api_key:
            os.environ["LANGCHAIN_API_KEY"] = api_key
        elif not os.environ.get("LANGCHAIN_API_KEY"):
            raise ValueError("LangSmith API key not found. Please provide it or set LANGCHAIN_API_KEY environment variable.")
        
        # Create LangSmith client
        self.client = Client()
        
        # Set up LangChain tracing
        langchain.debug = True
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "rag-evaluation"
    
    def create_dataset(self, name: str, description: str = "RAG evaluation dataset"):
        """Create a dataset for evaluation"""
        return self.client.create_dataset(
            dataset_name=name,
            description=description
        )
    
    def add_examples(self, dataset_name: str, examples: List[Dict[str, Any]]):
        """Add examples to the dataset"""
        dataset = self.client.read_dataset(dataset_name=dataset_name)
        
        for example in examples:
            self.client.create_example(
                inputs=example["inputs"],
                outputs=example.get("outputs", {}),
                dataset_id=dataset.id
            )
    
    def evaluate_rag_pipeline(self, 
                            dataset_name: str, 
                            pipeline, 
                            evaluators: List[str] = None,
                            custom_metrics: List[Any] = None):
        """Evaluate a RAG pipeline using LangSmith"""
        if evaluators is None:
            evaluators = ["answer_relevance", "context_relevance", "groundedness", "faithfulness"]
        
        # Create LangSmith adapter for the pipeline
        pipeline_adapter = LangSmithRAGAdapter(pipeline)
        
        # Convert string to actual LLM model instance
        eval_llm = ChatOpenAI(model="gpt-4")
        
        # Configure evaluation
        eval_config = RunEvalConfig(
            evaluators=evaluators,
            custom_evaluators=custom_metrics or [],
            eval_llm=eval_llm
        )
        
        # Generate a unique project name to avoid conflicts
        unique_project_name = f"rag-evaluation-results-{uuid.uuid4()}"
        
        # Run evaluation
        results = self.client.run_on_dataset(
            dataset_name=dataset_name,
            llm_or_chain_factory=pipeline_adapter,
            evaluation=eval_config,
            project_name=unique_project_name,
            # Add tags to easily identify this evaluation run
            tags=["rag-evaluation", f"dataset-{dataset_name}"]
        )
        
        return results
        
    def generate_evaluation_example(self, query: str, response: str, contexts: List[str], ground_truth: Optional[str] = None):
        """Generate an evaluation example for LangSmith"""
        example = {
            "inputs": {
                "query": query,
                "contexts": contexts
            },
            "outputs": {
                "response": response
            }
        }
        
        if ground_truth:
            example["outputs"]["ground_truth"] = ground_truth
            
        return example
    
    def custom_groundedness_evaluator(self, run: Run, example: Optional[Example] = None) -> EvaluationResult:
        """Custom evaluator for groundedness"""
        from langchain_openai import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate
        
        # Extract data
        query = run.inputs.get("query")
        contexts = run.inputs.get("contexts")
        response = run.outputs.get("response")
        
        # Format contexts
        context_text = "\n\n".join([f"Context {i+1}: {context}" for i, context in enumerate(contexts)])
        
        # Create prompt template
        template = """
        Evaluate the groundedness of the answer in the provided contexts on a scale of 1 to 5.
        
        Answer: {response}
        
        Contexts:
        {contexts}
        
        Scoring guidelines:
        1: The answer contains information not present in the contexts (hallucination).
        2: The answer has significant content not grounded in the contexts.
        3: The answer is partially grounded in the contexts but includes some ungrounded statements.
        4: The answer is mostly grounded in the contexts with minor extrapolations.
        5: The answer is completely grounded in the contexts with no hallucinations.
        
        Your response should include the score (a number between 1 and 5) and a brief explanation of why you assigned that score.
        """
        
        # Create evaluator model
        evaluator_model = ChatOpenAI(model="gpt-4")
        prompt_template = ChatPromptTemplate.from_template(template)
        chain = prompt_template | evaluator_model
        
        # Get evaluation
        eval_response = chain.invoke({
            "response": response,
            "contexts": context_text
        })
        
        # Extract score from response
        try:
            # This is a simplistic approach; you might need more robust parsing
            score_line = eval_response.content.split("\n")[0]
            score = float(score_line.split(":")[1].strip() if ":" in score_line else score_line.strip())
            score = min(max(score, 1), 5)
            
            # Create evaluation result
            return EvaluationResult(
                key="groundedness",
                score=score / 5.0,  # Normalize to 0-1
                comment=eval_response.content
            )
        except (ValueError, IndexError):
            return EvaluationResult(
                key="groundedness",
                score=0,
                comment="Failed to extract score from evaluation response."
            )