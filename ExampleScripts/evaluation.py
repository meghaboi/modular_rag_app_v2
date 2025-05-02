from typing import List, Dict, Any, Optional
import os

class RAGEvaluator:
    """Class for evaluating RAG system performance"""
    
    def __init__(self, metrics: List[str]):
        """Initialize the RAG evaluator with specific metrics"""
        from langchain_openai import ChatOpenAI
        
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key not found in environment variables")
        
        self._metrics = metrics
        self._evaluator_model = ChatOpenAI(model_name="gpt-4")
    
    def evaluate(self, query: str, response: str, contexts: List[str], ground_truth: Optional[str] = None) -> Dict[str, float]:
        """Evaluate the RAG system performance using selected metrics"""
        results = {}
        
        # Combine contexts for evaluation
        context_text = "\n\n".join([f"Context {i+1}: {context}" for i, context in enumerate(contexts)])
        
        # Evaluate each selected metric
        for metric in self._metrics:
            if metric == "answer_relevance":
                results[metric] = self._evaluate_answer_relevance(query, response, ground_truth)
            elif metric == "context_relevance":
                results[metric] = self._evaluate_context_relevance(query, contexts)
            elif metric == "groundedness":
                results[metric] = self._evaluate_groundedness(response, contexts)
            elif metric == "faithfulness":
                results[metric] = self._evaluate_faithfulness(response, contexts)
        
        return results
    
    def _evaluate_answer_relevance(self, query: str, response: str, ground_truth: Optional[str] = None) -> float:
        """Evaluate the relevance of the answer to the query"""
        from langchain.prompts import ChatPromptTemplate
        
        template = """
        Evaluate the relevance of the answer to the question on a scale of 1 to 5.
        
        Question: {query}
        Answer: {response}
        """
        
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
        
        Your response should be just the score (a number between 1 and 5).
        """
        
        prompt_template = ChatPromptTemplate.from_template(template)
        
        if ground_truth:
            chain = prompt_template | self._evaluator_model
            response_obj = chain.invoke({
                "query": query,
                "response": response,
                "ground_truth": ground_truth
            })
        else:
            chain = prompt_template | self._evaluator_model
            response_obj = chain.invoke({
                "query": query,
                "response": response
            })
        
        # Extract score from response
        try:
            score = float(response_obj.content.strip())
            return min(max(score, 1), 5)
        except ValueError:
            return 0  # Return 0 if score cannot be extracted
    
    def _evaluate_context_relevance(self, query: str, contexts: List[str]) -> float:
        """Evaluate the relevance of the contexts to the query"""
        from langchain.prompts import ChatPromptTemplate
        
        context_text = "\n\n".join([f"Context {i+1}: {context}" for i, context in enumerate(contexts)])
        
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
        
        Your response should be just the score (a number between 1 and 5).
        """
        
        prompt_template = ChatPromptTemplate.from_template(template)
        chain = prompt_template | self._evaluator_model
        response = chain.invoke({
            "query": query,
            "contexts": context_text
        })
        
        # Extract score from response
        try:
            score = float(response.content.strip())
            return min(max(score, 1), 5)
        except ValueError:
            return 0  # Return 0 if score cannot be extracted
    
    def _evaluate_groundedness(self, response: str, contexts: List[str]) -> float:
        """Evaluate if the response is grounded in the provided contexts"""
        from langchain.prompts import ChatPromptTemplate
        
        context_text = "\n\n".join([f"Context {i+1}: {context}" for i, context in enumerate(contexts)])
        
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
        
        Your response should be just the score (a number between 1 and 5).
        """
        
        prompt_template = ChatPromptTemplate.from_template(template)
        chain = prompt_template | self._evaluator_model
        response_obj = chain.invoke({
            "response": response,
            "contexts": context_text
        })
        
        # Extract score from response
        try:
            score = float(response_obj.content.strip())
            return min(max(score, 1), 5)
        except ValueError:
            return 0  # Return 0 if score cannot be extracted
    
    def _evaluate_faithfulness(self, response: str, contexts: List[str]) -> float:
        """Evaluate the faithfulness of the response to the provided contexts"""
        from langchain.prompts import ChatPromptTemplate
        
        context_text = "\n\n".join([f"Context {i+1}: {context}" for i, context in enumerate(contexts)])
        
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
        
        Your response should be just the score (a number between 1 and 5).
        """
        
        prompt_template = ChatPromptTemplate.from_template(template)
        chain = prompt_template | self._evaluator_model
        response_obj = chain.invoke({
            "response": response,
            "contexts": context_text
        })
        
        # Extract score from response
        try:
            score = float(response_obj.content.strip())
            return min(max(score, 1), 5)
        except ValueError:
            return 0  # Return 0 if score cannot be extracted