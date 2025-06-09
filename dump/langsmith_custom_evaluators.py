from typing import Optional, Dict, Any
from langsmith.evaluation import EvaluationResult

class RAGEvaluators:
    """Collection of custom evaluators for RAG systems"""
    
    @staticmethod
    def answer_relevance_evaluator(run, reference=None) -> EvaluationResult:
        """Evaluate answer relevance to the query"""
        from langchain_openai import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate
        
        # Extract data
        query = run.inputs.get("query")
        response = run.outputs.get("response")
        
        # Create prompt template
        template = """
        Evaluate the relevance of the answer to the question on a scale of 1 to 5.
        
        Question: {query}
        Answer: {response}
        """
        
        # Add ground truth if available
        ground_truth = None
        if reference and reference.get("outputs") and "ground_truth" in reference.get("outputs", {}):
            ground_truth = reference["outputs"]["ground_truth"]
            
        if ground_truth:
            template += """
            Ground Truth Answer: {ground_truth}
            
            Consider both how relevant the answer is to the question and how well it matches the ground truth.
            """
            
        template += """
        Scoring guidelines:
        1: The answer is completely irrelevant to the question.
        2: The answer is slightly relevant but misses the main point.
        3: The answer is moderately relevant but incomplete.
        4: The answer is relevant and mostly complete.
        5: The answer is highly relevant and complete.
        
        Your response should include the score (a number between 1 and 5) and a brief explanation of why you assigned that score.
        """
        
        # Create evaluator model
        evaluator_model = ChatOpenAI(model_name="gpt-4")
        prompt_template = ChatPromptTemplate.from_template(template)
        chain = prompt_template | evaluator_model
        
        # Get evaluation
        if ground_truth:
            eval_response = chain.invoke({
                "query": query,
                "response": response,
                "ground_truth": ground_truth
            })
        else:
            eval_response = chain.invoke({
                "query": query,
                "response": response
            })
        
        # Extract score from response
        try:
            # This is a simplistic approach; you might need more robust parsing
            score_line = eval_response.content.split("\n")[0]
            score = float(score_line.split(":")[1].strip() if ":" in score_line else score_line.strip())
            score = min(max(score, 1), 5)
            
            # Create evaluation result
            return EvaluationResult(
                key="answer_relevance",
                score=score / 5.0,  # Normalize to 0-1
                comment=eval_response.content
            )
        except (ValueError, IndexError):
            return EvaluationResult(
                key="answer_relevance",
                score=0,
                comment="Failed to extract score from evaluation response."
            )
    
    @staticmethod
    def context_relevance_evaluator(run, reference=None) -> EvaluationResult:
        """Evaluate context relevance to the query"""
        from langchain_openai import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate
        
        # Extract data
        query = run.inputs.get("query")
        contexts = run.outputs.get("contexts")
        
        # Format contexts
        context_text = "\n\n".join([f"Context {i+1}: {context}" for i, context in enumerate(contexts)])
        
        # Create prompt template
        template = """
        Evaluate the relevance of the provided contexts to the question on a scale of 1 to 5.
        
        Question: {query}
        
        Contexts:
        {contexts}
        
        Scoring guidelines:
        1: The contexts are completely irrelevant to the question.
        2: The contexts are slightly relevant but miss important information.
        3: The contexts are moderately relevant but incomplete.
        4: The contexts are relevant and contain most of the necessary information.
        5: The contexts are highly relevant and contain all necessary information.
        
        Your response should include the score (a number between 1 and 5) and a brief explanation of why you assigned that score.
        """
        
        # Create evaluator model
        evaluator_model = ChatOpenAI(model_name="gpt-4")
        prompt_template = ChatPromptTemplate.from_template(template)
        chain = prompt_template | evaluator_model
        
        # Get evaluation
        eval_response = chain.invoke({
            "query": query,
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
                key="context_relevance",
                score=score / 5.0,  # Normalize to 0-1
                comment=eval_response.content
            )
        except (ValueError, IndexError):
            return EvaluationResult(
                key="context_relevance",
                score=0,
                comment="Failed to extract score from evaluation response."
            )
    
    @staticmethod
    def groundedness_evaluator(run, reference=None) -> EvaluationResult:
        """Evaluate groundedness of the response in the contexts"""
        from langchain_openai import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate
        
        # Extract data
        response = run.outputs.get("response")
        contexts = run.outputs.get("contexts")
        
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
        evaluator_model = ChatOpenAI(model_name="gpt-4")
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
    
    @staticmethod
    def faithfulness_evaluator(run, reference=None) -> EvaluationResult:
        """Evaluate faithfulness of the response to the contexts"""
        from langchain_openai import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate
        
        # Extract data
        response = run.outputs.get("response")
        contexts = run.outputs.get("contexts")
        
        # Format contexts
        context_text = "\n\n".join([f"Context {i+1}: {context}" for i, context in enumerate(contexts)])
        
        # Create prompt template
        template = """
        Evaluate the faithfulness of the answer to the provided contexts on a scale of 1 to 5.
        
        Answer: {response}
        
        Contexts:
        {contexts}
        
        Scoring guidelines:
        1: The answer contradicts or misrepresents the information in the contexts.
        2: The answer includes significant misinterpretations of the contexts.
        3: The answer is partially faithful but includes some misinterpretations.
        4: The answer is mostly faithful with minor inaccuracies.
        5: The answer is completely faithful to the information in the contexts.
        
        Your response should include the score (a number between 1 and 5) and a brief explanation of why you assigned that score.
        """
        
        # Create evaluator model
        evaluator_model = ChatOpenAI(model_name="gpt-4")
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
                key="faithfulness",
                score=score / 5.0,  # Normalize to 0-1
                comment=eval_response.content
            )
        except (ValueError, IndexError):
            return EvaluationResult(
                key="faithfulness",
                score=0,
                comment="Failed to extract score from evaluation response."
            )