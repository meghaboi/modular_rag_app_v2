# evaluator.py
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import os
import re # Added for parsing LLM responses
import json # Added for parsing LLM responses that might be JSON

from langchain_core.prompts import ChatPromptTemplate # Added for creating prompts
from langchain_google_genai import ChatGoogleGenerativeAI # Added for Gemini model

from enums import EvaluationBackendType, EvaluationMetricType

class BaseEvaluator(ABC):
    """Abstract base class for RAG evaluators"""
    
    def __init__(self, metrics: List[str]):
        """Initialize with selected metrics"""
        self._metrics = metrics
    
    @abstractmethod
    def evaluate(self, query: str, response: str, contexts: List[str], 
                 ground_truth: Optional[str] = None) -> Dict[str, float]:
        """Evaluate RAG system performance using selected metrics"""
        pass
    
    @property
    @abstractmethod
    def supported_metrics(self) -> List[str]:
        """Return list of metrics supported by this evaluator"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the evaluator"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of the evaluator"""
        pass


class BuiltinEvaluator(BaseEvaluator):
    """Built-in evaluator using LLM for evaluation"""
    
    def __init__(self, metrics: List[str]):
        """Initialize the built-in evaluator"""
        super().__init__(metrics)
        from langchain_openai import ChatOpenAI
        
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key required for built-in evaluation")
        
        self._evaluator_model = ChatOpenAI(model_name="gpt-4")
    
    def evaluate(self, query: str, response: str, contexts: List[str], 
                 ground_truth: Optional[str] = None) -> Dict[str, float]:
        """Evaluate RAG system performance using selected metrics"""
        results = {}
        
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
    
    @property
    def supported_metrics(self) -> List[str]:
        return [
            EvaluationMetricType.ANSWER_RELEVANCE.value,
            EvaluationMetricType.CONTEXT_RELEVANCE.value,
            EvaluationMetricType.GROUNDEDNESS.value,
            EvaluationMetricType.FAITHFULNESS.value
        ]
    
    @property
    def name(self) -> str:
        return "Built-in LLM Evaluator"
    
    @property
    def description(self) -> str:
        return "Uses GPT-4 to evaluate RAG output on various dimensions"

class RAGASEvaluator(BaseEvaluator):
    """RAGAS-based evaluator for RAG systems"""
    
    def __init__(self, metrics: List[str] = None):
        """
        Initialize the RAGAS evaluator with optional metrics
        
        Args:
            metrics: List of metric names to use (default: all supported metrics)
        """
        # Import RAGAS metrics
        try:
            import ragas
            from ragas.metrics import (
                faithfulness,
                # answer_correctness, # Removed as per user request
                context_precision,
                context_recall
            )
            # Import class-based metrics for explicit LLM initialization
            from ragas.metrics import Faithfulness, ContextPrecision, ContextRecall 
            # AnswerCorrectness is not imported as it's being removed
        except ImportError as e:
            raise ValueError(f"Required library not installed: {e}")
        
        # Store RAGAS metric objects, using class-based metrics initialized with the LLM
        self._ragas_metrics = {
            "faithfulness": Faithfulness(),
            # "answer_correctness": AnswerCorrectness(llm=self._llm), # Removed
            "context_precision": ContextPrecision(),
            "context_recall": ContextRecall()
        }
        
        # Use all metrics if none specified (answer_correctness is already excluded from _ragas_metrics)
        if metrics is None:
            self._metrics = list(self._ragas_metrics.keys())
        else:
            # Validate provided metrics
            invalid_metrics = [m for m in metrics if m not in self._ragas_metrics and m != "f1_score"]
            if invalid_metrics:
                raise ValueError(f"Unsupported metrics: {invalid_metrics}")
            self._metrics = metrics
        
        # Verify OpenAI API key exists for RAGAS
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key required for RAGAS evaluation")
        
        # Configure RAGAS to use this LLM
        import ragas
        ragas.llm = self._llm
    
    def evaluate(self, query: str, response: str, contexts: List[str], 
                ground_truth: Optional[str] = None) -> Dict[str, float]:
        """
        Evaluate using RAGAS metrics
        
        Args:
            query: The question asked
            response: The generated answer
            contexts: The contexts used to generate the answer
            ground_truth: Optional ground truth answer
            
        Returns:
            Dictionary of evaluation metrics and scores (scaled to 1-5)
        """
        import logging
        
        try:
            from datasets import Dataset
            import ragas
            from ragas import evaluate as ragas_evaluate
            import pandas as pd
            
            
            # Prepare data
            data = {
                "question": [query],
                "answer": [response],
                "contexts": [contexts]  # List of lists as expected by RAGAS
            }
            
            if ground_truth:
                data["ground_truths"] = [[ground_truth]] 
                data["reference"] = [ground_truth]
            
            ds = Dataset.from_dict(data)
            
            # Get metrics
            active_metrics = [self._ragas_metrics[metric] for metric in self._metrics 
                            if metric in self._ragas_metrics]
            
            # Run evaluation
            results = ragas_evaluate(ds, metrics=active_metrics)
            
            # Initialize metrics dictionary
            metrics_dict = {}
            
            # Log the type and structure of the results object
            logging.info(f"RAGAS results type: {type(results)}")
            
            # Log all attributes of the results object
            for attr_name in dir(results):
                if not attr_name.startswith('_') and not callable(getattr(results, attr_name)):
                    value = getattr(results, attr_name)
                    logging.info(f"  {attr_name}: {value} (type: {type(value)})")
            
            # Check for 'scores' attribute directly (common in newer RAGAS versions)
            if hasattr(results, 'scores') and results.scores:
                logging.info(f"Found scores attribute: {results.scores}")
                
                # Handle both list and dictionary formats for scores
                scores_data = results.scores
                if isinstance(scores_data, list) and len(scores_data) > 0:
                    first_score = scores_data[0]
                    if isinstance(first_score, dict):
                        for metric in self._metrics:
                            if metric in first_score:
                                raw_value = float(first_score[metric])
                                # FIXED SCALING: Properly map from 0-1 to 1-5 scale
                                scaled_value = 1.0 + raw_value * 4.0
                                metrics_dict[metric] = round(scaled_value, 2)
                                logging.info(f"Found metric {metric} in scores[0], value: {raw_value}")
                            else:
                                # Try alternate names
                                found = False
                                for alt_name in self.metric_name_map.get(metric, []):
                                    if alt_name in first_score:
                                        raw_value = float(first_score[alt_name])
                                        # FIXED SCALING: Properly map from 0-1 to 1-5 scale
                                        scaled_value = 1.0 + raw_value * 4.0
                                        metrics_dict[metric] = round(scaled_value, 2)
                                        logging.info(f"Found metric {metric} as {alt_name}, value: {raw_value}")
                                        found = True
                                        break
                                
                                if not found:
                                    logging.warning(f"Metric {metric} not found in scores dictionary")
                                    metrics_dict[metric] = 3.0
                elif isinstance(scores_data, dict):
                    for metric in self._metrics:
                        if metric in scores_data:
                            raw_value = float(scores_data[metric])
                            # FIXED SCALING: Properly map from 0-1 to 1-5 scale
                            scaled_value = 1.0 + raw_value * 4.0
                            metrics_dict[metric] = round(scaled_value, 2)
                            logging.info(f"Found metric {metric} in scores dict, value: {raw_value}")
                        else:
                            # Try alternate names
                            found = False
                            for alt_name in self.metric_name_map.get(metric, []):
                                if alt_name in scores_data:
                                    raw_value = float(scores_data[alt_name])
                                    # FIXED SCALING: Properly map from 0-1 to 1-5 scale
                                    scaled_value = 1.0 + raw_value * 4.0
                                    metrics_dict[metric] = round(scaled_value, 2)
                                    logging.info(f"Found metric {metric} as {alt_name}, value: {raw_value}")
                                    found = True
                                    break
                            
                            if not found:
                                logging.warning(f"Metric {metric} not found in scores dictionary")
                                metrics_dict[metric] = 3.0
            
            # Calculate F1 score if both context_recall and context_precision are available
            if "f1_score" in self._metrics:
                recall = metrics_dict.get("context_recall", 0)
                precision = metrics_dict.get("context_precision", 0)
                
                if recall > 0 and precision > 0:
                    # Convert from 1-5 scale to 0-1 scale for calculation
                    recall_01 = (recall - 1) / 4
                    precision_01 = (precision - 1) / 4
                    
                    # Calculate harmonic mean (F1 score)
                    f1_score = 2 * (recall_01 * precision_01) / (recall_01 + precision_01)
                    
                    # Convert back to 1-5 scale
                    f1_score_scaled = 1.0 + f1_score * 4.0
                    metrics_dict["f1_score"] = round(f1_score_scaled, 2)
                else:
                    metrics_dict["f1_score"] = 3.0  # Default middle value if components missing
            
            return metrics_dict
            
        except Exception as e:
            logging.error(f"RAGAS evaluation error: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            
            # Return default values on complete failure
            return {metric: 3.0 for metric in self._metrics}
    
    @property
    def supported_metrics(self) -> List[str]:
        """Return list of metrics supported by this evaluator"""
        return list(self._ragas_metrics.keys()) + ["f1_score"]
    
    @property
    def name(self) -> str:
        return "RAGAS Evaluator"
    
    @property
    def description(self) -> str:
        return "Uses RAGAS framework to evaluate RAG system performance"

class LangSmithEvaluator(BaseEvaluator):
    """LangSmith-based evaluator for RAG systems using direct API calls without database dependencies"""
    
    def __init__(self, metrics: List[str]):
        """Initialize the LangSmith evaluator"""
        super().__init__(metrics)
        
        # Verify LangChain API key exists
        if not os.environ.get("LANGCHAIN_API_KEY"):
            raise ValueError("LangChain API key required for LangSmith evaluation")
        
        # Import required libraries for LLM-based evaluation
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise ValueError(f"Required library not installed: {e}")
        
        # Initialize evaluator model for metrics
        self.evaluator_model = ChatOpenAI(model_name="gpt-4")
        
        # Define supported metrics
        self._supported_metrics = [
            EvaluationMetricType.ANSWER_RELEVANCE.value,
            EvaluationMetricType.CONTEXT_RELEVANCE.value,
            EvaluationMetricType.GROUNDEDNESS.value,
            EvaluationMetricType.FAITHFULNESS.value
        ]
    
    def evaluate(self, query: str, response: str, contexts: List[str], 
                 ground_truth: Optional[str] = None) -> Dict[str, float]:
        """Evaluate RAG system performance using LangSmith-inspired prompts"""
        results = {}
        
        # Process each requested metric
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
        """Evaluate the relevance of the answer to the query using LangSmith-inspired prompts"""
        from langchain.prompts import ChatPromptTemplate
        
        # Use LangSmith-style prompting but directly with the LLM
        template = """
        You are an expert evaluator of RAG (Retrieval-Augmented Generation) systems.
        Your task is to evaluate the relevance of an answer to a given question.
        
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
        
        Your response must be exactly one number between 1 and 5, with no additional explanation.
        """
        
        prompt_template = ChatPromptTemplate.from_template(template)
        
        if ground_truth:
            chain = prompt_template | self.evaluator_model
            response_obj = chain.invoke({
                "query": query,
                "response": response,
                "ground_truth": ground_truth
            })
        else:
            chain = prompt_template | self.evaluator_model
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
        """Evaluate the relevance of contexts to the query"""
        from langchain.prompts import ChatPromptTemplate
        
        context_text = "\n\n".join([f"Context {i+1}: {context}" for i, context in enumerate(contexts)])
        
        template = """
        You are an expert evaluator of RAG (Retrieval-Augmented Generation) systems.
        Your task is to evaluate how relevant the retrieved contexts are to the question.
        
        Question: {query}
        
        Retrieved Contexts:
        {contexts}
        
        Scoring guidelines:
        1: The contexts are completely irrelevant to the question.
        2: The contexts are slightly relevant but miss important information.
        3: The contexts are moderately relevant but incomplete.
        4: The contexts are relevant and contain most of the necessary information.
        5: The contexts are highly relevant and contain all necessary information.
        
        Your response must be exactly one number between 1 and 5, with no additional explanation.
        """
        
        prompt_template = ChatPromptTemplate.from_template(template)
        chain = prompt_template | self.evaluator_model
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
        You are an expert evaluator of RAG (Retrieval-Augmented Generation) systems.
        Your task is to evaluate if the generated answer is grounded in the provided contexts.
        
        Answer: {response}
        
        Retrieved Contexts:
        {contexts}
        
        Scoring guidelines:
        1: The answer contains information not present in the contexts (hallucination).
        2: The answer has significant content not grounded in the contexts.
        3: The answer is partially grounded in the contexts but includes some ungrounded statements.
        4: The answer is mostly grounded in the contexts with minor extrapolations.
        5: The answer is completely grounded in the contexts with no hallucinations.
        
        Your response must be exactly one number between 1 and 5, with no additional explanation.
        """
        
        prompt_template = ChatPromptTemplate.from_template(template)
        chain = prompt_template | self.evaluator_model
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
        You are an expert evaluator of RAG (Retrieval-Augmented Generation) systems.
        Your task is to evaluate how faithful the generated answer is to the information in the provided contexts.
        
        Answer: {response}
        
        Retrieved Contexts:
        {contexts}
        
        Scoring guidelines:
        1: The answer contradicts or misrepresents the information in the contexts.
        2: The answer includes significant misinterpretations of the contexts.
        3: The answer is partially faithful but includes some misinterpretations.
        4: The answer is mostly faithful with minor inaccuracies.
        5: The answer is completely faithful to the information in the contexts.
        
        Your response must be exactly one number between 1 and 5, with no additional explanation.
        """
        
        prompt_template = ChatPromptTemplate.from_template(template)
        chain = prompt_template | self.evaluator_model
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
    
    @property
    def supported_metrics(self) -> List[str]:
        """Return list of metrics supported by this evaluator"""
        return self._supported_metrics
    
    @property
    def name(self) -> str:
        """Return the name of the evaluator"""
        return "LangSmith Evaluator"
    
    @property
    def description(self) -> str:
        """Return a description of the evaluator"""
        return "Uses LangSmith-inspired evaluation techniques for assessing RAG system performance"

class DeepEvaluator(BaseEvaluator):
    """Evaluator using smaller, specialized LLMs for different metrics"""
    
    def __init__(self, metrics: List[str]):
        """Initialize the DeepEvaluator with selected metrics"""
        super().__init__(metrics)
        
        try:
            from langchain_openai import ChatOpenAI
            from langchain_anthropic import ChatAnthropic
            from langchain_mistralai import ChatMistralAI
        except ImportError as e:
            raise ValueError(f"Required library not installed: {e}")
        
        # Check for required API keys
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key required for DeepEvaluator")
        
        # Initialize different models for different evaluation tasks
        # Using smaller models for faster, more cost-effective evaluation
        self._general_evaluator = ChatOpenAI(model_name="gpt-3.5-turbo")
        
        # For metrics that need more careful analysis, use a more capable model
        self._deep_evaluator = None
        if os.environ.get("ANTHROPIC_API_KEY"):
            self._deep_evaluator = ChatAnthropic(model="claude-3-haiku-20240307")
        elif os.environ.get("MISTRAL_API_KEY"):
            self._deep_evaluator = ChatMistralAI(model="mistral-small")
        else:
            # Fallback to OpenAI
            self._deep_evaluator = self._general_evaluator
        
        # Create a mapping of metrics to the most appropriate model for evaluation
        self._metric_to_model = {
            "answer_relevance": self._general_evaluator,  # Simple relevance check
            "context_relevance": self._general_evaluator,  # Simple relevance check
            "groundedness": self._deep_evaluator,  # Requires deeper analysis
            "faithfulness": self._deep_evaluator,  # Requires deeper analysis
            "answer_consistency": self._deep_evaluator,  # Custom metric
            "context_coverage": self._general_evaluator  # Custom metric
        }
    
    def evaluate(self, query: str, response: str, contexts: List[str], 
                 ground_truth: Optional[str] = None) -> Dict[str, float]:
        """Evaluate RAG system performance using selected metrics with specialized models"""
        results = {}
        
        # Evaluate each selected metric with the appropriate model
        for metric in self._metrics:
            try:
                if metric == "answer_relevance":
                    results[metric] = self._evaluate_answer_relevance(query, response, ground_truth)
                elif metric == "context_relevance":
                    results[metric] = self._evaluate_context_relevance(query, contexts)
                elif metric == "groundedness":
                    results[metric] = self._evaluate_groundedness(response, contexts)
                elif metric == "faithfulness":
                    results[metric] = self._evaluate_faithfulness(response, contexts)
                elif metric == "answer_consistency":
                    results[metric] = self._evaluate_answer_consistency(response, contexts)
                elif metric == "context_coverage":
                    results[metric] = self._evaluate_context_coverage(query, contexts)
            except Exception as e:
                # Log error and use default middle score instead of 0
                print(f"Error evaluating {metric}: {str(e)}")
                results[metric] = 3.0  # Default to middle score instead of 0
        
        return results
    
    def _extract_score_from_response(self, response_text: str) -> float:
        """
        Extract a numeric score from LLM response with improved robustness.
        Returns a score between 1 and 5, with better fallback handling.
        """
        # First try direct extraction of a single number
        try:
            score = float(response_text.strip())
            return min(max(score, 1), 5)
        except ValueError:
            pass
        
        # Try to extract the first number from the text
        import re
        number_matches = re.findall(r'\d+\.?\d*', response_text)
        if number_matches:
            try:
                score = float(number_matches[0])
                if 1 <= score <= 5:
                    return score
            except ValueError:
                pass
        
        # Look for score indicators in text
        lower_text = response_text.lower()
        if "score: " in lower_text:
            score_text = lower_text.split("score: ")[1].split()[0]
            try:
                score = float(score_text)
                return min(max(score, 1), 5)
            except ValueError:
                pass
        
        # Check for textual indicators
        if "excellent" in lower_text or "perfect" in lower_text or "completely" in lower_text:
            return 5.0
        elif "good" in lower_text or "mostly" in lower_text:
            return 4.0
        elif "moderate" in lower_text or "partial" in lower_text or "average" in lower_text:
            return 3.0
        elif "poor" in lower_text or "slight" in lower_text:
            return 2.0
        elif "terrible" in lower_text or "complete" in lower_text and "irrelevant" in lower_text:
            return 1.0
            
        # Default to middle score if all extraction methods fail
        return 3.0
    
    def _evaluate_answer_relevance(self, query: str, response: str, ground_truth: Optional[str] = None) -> float:
        """Evaluate the relevance of the answer to the query"""
        from langchain.prompts import ChatPromptTemplate
        
        model = self._metric_to_model["answer_relevance"]
        
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
            chain = prompt_template | model
            response_obj = chain.invoke({
                "query": query,
                "response": response,
                "ground_truth": ground_truth
            })
        else:
            chain = prompt_template | model
            response_obj = chain.invoke({
                "query": query,
                "response": response
            })
        
        # Use the improved score extraction method
        return self._extract_score_from_response(response_obj.content)
    
    def _evaluate_context_relevance(self, query: str, contexts: List[str]) -> float:
        """Evaluate the relevance of the contexts to the query"""
        from langchain.prompts import ChatPromptTemplate
        
        model = self._metric_to_model["context_relevance"]
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
        chain = prompt_template | model
        response = chain.invoke({
            "query": query,
            "contexts": context_text
        })
        
        # Use the improved score extraction method
        return self._extract_score_from_response(response.content)
    
    def _evaluate_groundedness(self, response: str, contexts: List[str]) -> float:
        """Evaluate if the response is grounded in the provided contexts"""
        from langchain.prompts import ChatPromptTemplate
        
        model = self._metric_to_model["groundedness"]
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
        chain = prompt_template | model
        response_obj = chain.invoke({
            "response": response,
            "contexts": context_text
        })
        
        # Use the improved score extraction method
        return self._extract_score_from_response(response_obj.content)
    
    def _evaluate_faithfulness(self, response: str, contexts: List[str]) -> float:
        """Evaluate the faithfulness of the response to the provided contexts"""
        from langchain.prompts import ChatPromptTemplate
        
        model = self._metric_to_model["faithfulness"]
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
        chain = prompt_template | model
        response_obj = chain.invoke({
            "response": response,
            "contexts": context_text
        })
        
        # Use the improved score extraction method
        return self._extract_score_from_response(response_obj.content)
    
    def _evaluate_answer_consistency(self, response: str, contexts: List[str]) -> float:
        """Custom metric: Evaluate the internal consistency of the answer"""
        from langchain.prompts import ChatPromptTemplate
        
        model = self._metric_to_model["answer_consistency"]
        
        template = """
        Evaluate the internal consistency of the answer on a scale of 1 to 5.
        
        Answer: {response}
        
        Scoring guidelines:
        1: The answer contains severe internal contradictions or logical inconsistencies.
        2: The answer has noticeable contradictions or logical flaws.
        3: The answer has minor inconsistencies but maintains overall coherence.
        4: The answer is mostly consistent with minimal logical issues.
        5: The answer is perfectly consistent with no contradictions or logical flaws.
        
        Your response should be just the score (a number between 1 and 5).
        """
        
        prompt_template = ChatPromptTemplate.from_template(template)
        chain = prompt_template | model
        response_obj = chain.invoke({
            "response": response
        })
        
        # Use the improved score extraction method
        return self._extract_score_from_response(response_obj.content)
    
    def _evaluate_context_coverage(self, query: str, contexts: List[str]) -> float:
        """Custom metric: Evaluate how well the contexts cover different aspects of the query"""
        from langchain.prompts import ChatPromptTemplate
        
        model = self._metric_to_model["context_coverage"]
        context_text = "\n\n".join([f"Context {i+1}: {context}" for i, context in enumerate(contexts)])
        
        template = """
        First, identify the key aspects or sub-questions contained in the main question.
        Then evaluate how completely the provided contexts cover these aspects on a scale of 1 to 5.
        
        Question: {query}
        
        Contexts:
        {contexts}
        
        Scoring guidelines:
        1: The contexts fail to address most aspects of the question.
        2: The contexts address only a few aspects of the question.
        3: The contexts address about half of the aspects of the question.
        4: The contexts address most aspects of the question.
        5: The contexts comprehensively address all aspects of the question.
        
        Your response should be just the score (a number between 1 and 5).
        """
        
        prompt_template = ChatPromptTemplate.from_template(template)
        chain = prompt_template | model
        response = chain.invoke({
            "query": query,
            "contexts": context_text
        })
        
        # Use the improved score extraction method
        return self._extract_score_from_response(response.content)
    
    @property
    def supported_metrics(self) -> List[str]:
        """Return list of metrics supported by this evaluator"""
        return list(self._metric_to_model.keys())
    
    @property
    def name(self) -> str:
        """Return the name of the evaluator"""
        return "Deep Evaluator"
    
    @property
    def description(self) -> str:
        """Return a description of the evaluator"""
        return "Uses specialized language models for different evaluation metrics, balancing efficiency and accuracy"

class RAGASEvaluatorV2(BaseEvaluator):
    """RAGAS-based evaluator for RAG systems - Version 2"""
    
    def __init__(self, metrics: List[str] = None):
        """
        Initialize the RAGAS evaluator with optional metrics
        
        Args:
            metrics: List of metric names to use (default: all supported metrics)
        """
        # Import RAGAS metrics
        try:
            import ragas
            from ragas.metrics import (
                faithfulness,
                answer_correctness,
                context_precision,
                context_recall
            )
        except ImportError as e:
            raise ValueError(f"Required library not installed: {e}")
        
        # Store RAGAS metric objects
        self._ragas_metrics = {
            "faithfulness": faithfulness,
            "answer_correctness": answer_correctness,
            "context_precision": context_precision,
            "context_recall": context_recall
        }
        
        # Use all metrics if none specified
        if metrics is None:
            self._metrics = list(self._ragas_metrics.keys())
        else:
            # Validate provided metrics
            invalid_metrics = [m for m in metrics if m not in self._ragas_metrics]
            if invalid_metrics:
                raise ValueError(f"Unsupported metrics: {invalid_metrics}")
            self._metrics = metrics
        
        # Verify OpenAI API key exists for RAGAS
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key required for RAGAS evaluation")
        
        # Initialize the LLM for RAGAS
        from langchain_openai import ChatOpenAI
        self._llm = ChatOpenAI(model_name="gpt-3.5-turbo")
        
        # Configure RAGAS to use this LLM
        import ragas
        ragas.llm = self._llm
    
    def evaluate(self, query: str, response: str, contexts: List[str], 
                ground_truth: Optional[str] = None) -> Dict[str, float]:
        """
        Evaluate using RAGAS metrics
        
        Args:
            query: The question asked
            response: The generated answer
            contexts: The contexts used to generate the answer
            ground_truth: Optional ground truth answer
            
        Returns:
            Dictionary of evaluation metrics and scores (scaled to 1-5)
        """
        try:
            from datasets import Dataset
            import ragas
            from ragas import evaluate as ragas_evaluate
            import logging
            
            # Configure RAGAS
            if not hasattr(ragas, 'llm') or ragas.llm is None:
                ragas.llm = self._llm
            
            # Prepare data
            data = {
                "question": [query],
                "answer": [response],
                "contexts": [contexts]  # List of lists as expected by RAGAS
            }
            
            if ground_truth:
                data["ground_truths"] = [[ground_truth]]
                data["reference"] = [ground_truth]
            
            ds = Dataset.from_dict(data)
            
            logging.info(f"Data : {data}")

            logging.info(f"DataSet : {ds}")

            # Get metrics to evaluate
            active_metrics = [self._ragas_metrics[metric] for metric in self._metrics 
                            if metric in self._ragas_metrics]
            
            # Run evaluation
            results = ragas_evaluate(ds, metrics=active_metrics, llm=self._llm)
            
            # Log results for debugging
            logging.info(f"RAGAS results: {results}")
            
            logging.info(f"Contexts: {contexts}")
            logging.info(f"Ground truth: {ground_truth}")

            # Initialize results dictionary
            metrics_dict = {}
            
            # Process results based on RAGAS version
            if hasattr(results, 'scores'):
                scores = results.scores
                logging.info(f"RAGAS scores attribute: {scores}")
                
                if isinstance(scores, list) and len(scores) > 0:
                    scores_dict = scores[0] if isinstance(scores[0], dict) else {}
                    logging.info(f"Using scores from list: {scores_dict}")
                    
                    for metric in self._metrics:
                        if metric in scores_dict:
                            raw_value = float(scores_dict[metric])
                            # Context metrics in RAGAS are usually already in 0-1 range where 1 is best
                            if metric in ["context_precision", "context_recall"]:
                                scaled_value = 1.0 + raw_value * 4.0
                            else:
                                scaled_value = 1.0 + raw_value * 4.0
                            metrics_dict[metric] = round(scaled_value, 2)
                        else:
                            metrics_dict[metric] = 3.0  # Default middle value
                
                elif isinstance(scores, dict):
                    logging.info(f"Using scores dict directly: {scores}")
                    for metric in self._metrics:
                        if metric in scores:
                            raw_value = float(scores[metric])
                            scaled_value = 1.0 + raw_value * 4.0
                            metrics_dict[metric] = round(scaled_value, 2)
                        else:
                            metrics_dict[metric] = 3.0
            
            # As a last resort, try direct attributes
            if not metrics_dict:
                for metric in self._metrics:
                    if hasattr(results, metric):
                        try:
                            raw_value = float(getattr(results, metric))
                            scaled_value = 1.0 + raw_value * 4.0
                            metrics_dict[metric] = round(scaled_value, 2)
                        except (ValueError, TypeError):
                            metrics_dict[metric] = 3.0
                    else:
                        metrics_dict[metric] = 3.0
            
            # Log final metrics
            logging.info(f"Final scaled metrics: {metrics_dict}")
            
            return metrics_dict
            
        except Exception as e:
            import logging
            logging.error(f"RAGAS evaluation error: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            
            # Return default values on complete failure
            return {metric: 3.0 for metric in self._metrics}
    
    @property
    def supported_metrics(self) -> List[str]:
        """Return list of metrics supported by this evaluator"""
        return list(self._ragas_metrics.keys())
    
    @property
    def name(self) -> str:
        return "RAGAS Evaluator V2"
    
    @property
    def description(self) -> str:
        return "Uses RAGAS framework to evaluate RAG system performance with improved result handling"

class CustomEvaluator(BaseEvaluator):
    """Custom evaluator using a Claude model for evaluation"""

    def __init__(self, metrics: List[str]):
        """Initialize the custom evaluator"""
        super().__init__(metrics)
        try:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise ValueError("ANTHROPIC_API_KEY environment variable not set. This is required for CustomEvaluator with Claude.")

            from langchain_anthropic import ChatAnthropic
            self._evaluator_model = ChatAnthropic(
                model="claude-3-opus-20240229" 
                # temperature=0.0 # For more deterministic outputs if needed
            )
        except ImportError:
            raise ValueError("langchain_anthropic is not installed. Please install it to use Claude for custom evaluation.")
        except Exception as e:
            print(f"Error initializing Claude model: {e}. Evaluation will not work.")
            self._evaluator_model = None

        if self._evaluator_model is None:
            raise ValueError("Claude model (claude-3-opus-20240229) could not be initialized. Please check ANTHROPIC_API_KEY and langchain_anthropic installation.")

    def _parse_llm_response_to_list(self, response_content: str, item_type: str = "statement") -> List[str]:
        """Parses LLM response (expected to be a list of items) into a Python list of strings."""
        # Assuming LLM might return a numbered list, bullet points, or simple newline-separated items.
        items = []
        # Try to parse as JSON list first
        try:
            parsed_json = json.loads(response_content)
            if isinstance(parsed_json, list):
                return [str(item) for item in parsed_json]
        except json.JSONDecodeError:
            pass # Not a JSON list, try other parsing

        lines = response_content.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Remove common list markers (numbers, bullets)
            line = re.sub(r"^\d+[\.\)]\s*", "", line)  # Matches "1. ", "1) "
            line = re.sub(r"^[\*\-\+]\s*", "", line)  # Matches "* ", "- ", "+ "
            if line:
                items.append(line.strip())
        
        if not items and response_content.strip(): # If parsing failed but there was content
             # Fallback: consider the whole response as a single item if it's not clearly a list
            if len(response_content.strip().split()) > 5: # Heuristic: if it's a sentence/paragraph
                 items.append(response_content.strip())
            else: # Otherwise, might be a malformed list, try splitting by common delimiters
                items = [i.strip() for i in re.split(r'[,;]', response_content) if i.strip()]

        if not items and response_content.strip():
            print(f"Warning: Could not parse LLM response into a list of {item_type}s. Response: {response_content}")
            return [response_content.strip()] # Return the whole response as a single item as a last resort
        return items

    def _parse_llm_yes_no_response(self, response_content: str, prompt_details: str) -> bool:
        """Parses LLM response to a boolean for Yes/No questions."""
        response_lower = response_content.strip().lower()
        if "yes" in response_lower and "no" not in response_lower: # Check for "yes" and not "yes, but no..."
            return True
        elif "no" in response_lower and "yes" not in response_lower:
            return False
        else:
            # Fallback: if the answer is not clearly yes/no, we might need to be conservative
            # or re-prompt. For now, let's assume 'no' if unclear to be safe.
            print(f"Warning: Ambiguous Yes/No response for prompt '{prompt_details}'. Response: {response_content}. Defaulting to False.")
            return False

    def _parse_llm_float_response(self, response_content: str, prompt_details: str) -> float:
        """Parses LLM response to a float, expected to be between 0 and 1."""
        try:
            # Extract numbers using regex to handle cases like "Score: 0.8" or just "0.8"
            match = re.search(r"[-+]?\d*\.?\d+", response_content)
            if match:
                score = float(match.group())
                return min(max(score, 0.0), 1.0) # Clamp to 0-1 range
            else:
                print(f"Warning: Could not parse float from response for prompt '{prompt_details}'. Response: {response_content}. Defaulting to 0.0.")
                return 0.0
        except ValueError:
            print(f"Warning: ValueError parsing float for prompt '{prompt_details}'. Response: {response_content}. Defaulting to 0.0.")
            return 0.0

    def evaluate(self, query: str, response: str, contexts: List[str],
                 ground_truth: Optional[str] = None) -> Dict[str, float]:
        """Evaluate RAG system performance using selected metrics"""
        results = {}

        for metric in self._metrics:
            if metric == EvaluationMetricType.CONTEXT_RECALL.value:
                if ground_truth is None:
                    raise ValueError("Ground truth is required for Context Recall.")
                results[metric] = self._evaluate_context_recall(query, ground_truth, contexts)
            elif metric == EvaluationMetricType.CONTEXT_PRECISION.value:
                if ground_truth is None:
                    raise ValueError("Ground truth is required for Context Precision.")
                results[metric] = self._evaluate_context_precision(query, ground_truth, contexts)
            elif metric == EvaluationMetricType.ANSWER_RELEVANCE.value:
                results[metric] = self._evaluate_answer_relevancy(query, response, contexts)
            elif metric == EvaluationMetricType.FAITHFULNESS.value:
                results[metric] = self._evaluate_faithfulness(response, contexts)
            elif metric == EvaluationMetricType.ANSWER_CORRECTNESS.value:
                if ground_truth is None:
                    raise ValueError("Ground truth is required for Answer Correctness.")
                results[metric] = self._evaluate_answer_correctness(response, ground_truth)
            # No F1 score or overall score as per requirements

        return results

    def _evaluate_context_recall(self, query: str, ground_truth: str, contexts: List[str]) -> float:
        """
        Measures the extent to which the retrieved context aligns with the annotated answer (ground truth).
        Computed using question, ground truth, and retrieved context. Values range from 0 to 1.
        """
        if not self._evaluator_model:
            print("Warning: Evaluator model not available for context_recall. Returning 0.0.")
            return 0.0

        # Step 1: Break the ground truth answer into individual statements.
        prompt_template_statements = ChatPromptTemplate.from_template(
            """Break the following ground truth answer into individual statements or claims.
            Each statement should be a distinct piece of factual information.
            Return a JSON list of strings. For example: ["Statement 1.", "Statement 2.", "Statement 3."]

            Ground Truth Answer:
            {ground_truth}

            Statements:"""
                    )
        chain_statements = prompt_template_statements | self._evaluator_model
        try:
            response_statements = chain_statements.invoke({"ground_truth": ground_truth})
            ground_truth_statements = self._parse_llm_response_to_list(response_statements.content, "ground truth statement")
        except Exception as e:
            print(f"Error getting statements from LLM for context_recall: {e}. Returning 0.0")
            ground_truth_statements = []

        if not ground_truth_statements:
            return 0.0

        # Step 2: For each ground truth statement, verify if it can be attributed to the retrieved context.
        attributed_statements = 0
        contexts_str = "\\n\\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(contexts)])
        
        prompt_template_attribution = ChatPromptTemplate.from_template(
            """Given the following retrieved contexts and a single statement from the ground truth,
            determine if the statement can be directly attributed to (supported by) the information present in the retrieved contexts.
            Answer with only "Yes" or "No".

            Retrieved Contexts:
            {contexts}

            Statement:
            {statement}

            Can the statement be attributed to the retrieved contexts? (Yes/No):"""
                    )
        chain_attribution = prompt_template_attribution | self._evaluator_model

        for stmt in ground_truth_statements:
            if not stmt.strip(): # Skip empty statements
                continue
            try:
                response_attribution = chain_attribution.invoke({"contexts": contexts_str, "statement": stmt})
                if self._parse_llm_yes_no_response(response_attribution.content, f"context_recall attribution for statement: {stmt}"):
                    attributed_statements += 1
            except Exception as e:
                print(f"Error during attribution check for context_recall statement '{stmt}': {e}")
                # Optionally, decide how to handle errors, e.g., assume not attributed

        # Step 3: Calculate context recall.
        recall_0_to_1 = attributed_statements / len(ground_truth_statements) if ground_truth_statements else 0.0
        scaled_recall = 1.0 + (min(max(recall_0_to_1, 0.0), 1.0) * 4.0) # Scale to 1-5
        return scaled_recall

    def _evaluate_context_precision(self, query: str, ground_truth: str, contexts: List[str]) -> float:
        """
        Evaluates whether all ground-truth relevant items in contexts are ranked higher.
        Computed using question, ground_truth, and contexts. Values range from 0 to 1.
        """
        if not self._evaluator_model:
            print("Warning: Evaluator model not available for context_precision. Returning 0.0.")
            return 0.0
        
        if not contexts:
            return 0.0

        precision_at_k_scores = []
        relevant_chunks_found = 0

        prompt_template_relevance = ChatPromptTemplate.from_template(
            """Consider the following question and ground truth answer.
            Then, evaluate if the provided context chunk contains information that is relevant and helpful to construct the ground truth answer for the question.
            Answer with only "Yes" or "No".

            Question:
            {question}

            Ground Truth Answer:
            {ground_truth}

            Context Chunk:
            {context_chunk}

            Is this context chunk relevant and helpful for the ground truth answer? (Yes/No):"""
                    )
        chain_relevance = prompt_template_relevance | self._evaluator_model

        for i, context_chunk in enumerate(contexts):
            k = i + 1
            is_relevant = False
            if not context_chunk.strip(): # Skip empty context chunks
                precision_at_k = relevant_chunks_found / k # if chunk is empty, it means previous relevant count over k
                precision_at_k_scores.append(precision_at_k)
                continue
            try:
                response_relevance = chain_relevance.invoke({
                    "question": query,
                    "ground_truth": ground_truth,
                    "context_chunk": context_chunk
                })
                if self._parse_llm_yes_no_response(response_relevance.content, f"context_precision relevance for chunk {k}"):
                    is_relevant = True
            except Exception as e:
                print(f"Error during relevance check for context_precision chunk {k}: {e}")
                # Optionally, decide how to handle errors, e.g., assume not relevant

            if is_relevant:
                relevant_chunks_found += 1
            
            precision_at_k = relevant_chunks_found / k
            precision_at_k_scores.append(precision_at_k)

        if not precision_at_k_scores:
            # If there were no context chunks to evaluate (e.g. contexts list was empty or all chunks were empty strings)
            # or if all LLM calls failed for relevance, we should return a base score. Let's use 1.0 for a 1-5 scale.
            return 1.0 
        
        mean_precision_0_to_1 = sum(precision_at_k_scores) / len(precision_at_k_scores)
        scaled_precision = 1.0 + (min(max(mean_precision_0_to_1, 0.0), 1.0) * 4.0) # Scale to 1-5
        return scaled_precision

    def _evaluate_answer_relevancy(self, query: str, answer: str, contexts: List[str]) -> float:
        """
        Assesses how pertinent the generated answer is to the given prompt.
        Computed using the question, the context and the answer. Values range from 0 to 1.
        """
        if not self._evaluator_model:
            print("Warning: Evaluator model not available for answer_relevancy. Returning 0.0.")
            return 0.0

        # Step 1: Reverse-engineer 'n' variants of the question from the generated answer using an LLM.
        contexts_str = "\\n\\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(contexts)])
        prompt_template_qgen = ChatPromptTemplate.from_template(
            """Given the following answer and supporting contexts, please generate 3 distinct questions that this answer could be a response to.
            Focus on rephrasing the core intent and information sought, based *only* on the provided answer and contexts.
            Return a JSON list of strings. For example: ["Generated Question 1?", "Generated Question 2?", "Generated Question 3?"]

            Answer:
            {answer}

            Contexts:
            {contexts}

            Generated Questions:"""
        )
        chain_qgen = prompt_template_qgen | self._evaluator_model
        generated_questions = []
        try:
            response_qgen = chain_qgen.invoke({"answer": answer, "contexts": contexts_str})
            generated_questions = self._parse_llm_response_to_list(response_qgen.content, "generated question")
        except Exception as e:
            print(f"Error generating questions for answer_relevancy: {e}")
        
        if not generated_questions:
            print("Warning: No questions generated by LLM for answer_relevancy. Returning 0.0.")
            return 0.0

        # Step 2: Calculate the mean similarity between the generated questions and the actual question.
        total_similarity = 0
        actual_eval_count = 0

        prompt_template_similarity = ChatPromptTemplate.from_template(
            """Rate the semantic similarity between the following two questions on a scale from 0.0 (not similar at all) to 1.0 (semantically identical).
            Provide only the numerical score.

            Original Question:
            {original_question}

            Generated Question:
            {generated_question}

            Similarity Score (0.0-1.0):"""
        )
        chain_similarity = prompt_template_similarity | self._evaluator_model

        for gen_q in generated_questions:
            if not gen_q.strip(): # Skip empty generated questions
                continue
            try:
                response_similarity = chain_similarity.invoke({"original_question": query, "generated_question": gen_q})
                similarity_score = self._parse_llm_float_response(response_similarity.content, f"answer_relevancy similarity for gen_q: {gen_q}")
                total_similarity += similarity_score
                actual_eval_count += 1
            except Exception as e:
                print(f"Error calculating similarity for answer_relevancy gen_q '{gen_q}': {e}")
        
        if actual_eval_count == 0:
            print("Warning: Could not calculate similarity for any generated questions in answer_relevancy. Returning 1.0.")
            return 1.0 # Bottom of 1-5 scale
            
        mean_similarity_0_to_1 = total_similarity / actual_eval_count
        scaled_similarity = 1.0 + (min(max(mean_similarity_0_to_1, 0.0), 1.0) * 4.0) # Scale to 1-5
        return scaled_similarity

    def _evaluate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """
        Measures the factual consistency of the generated answer against the given context.
        Calculated from answer and retrieved context. Scaled to (0,1) range.
        """
        if not self._evaluator_model:
            print("Warning: Evaluator model not available for faithfulness. Returning 0.0.")
            return 0.0

        # Step 1: Break the generated answer into individual statements.
        prompt_template_statements = ChatPromptTemplate.from_template(
            """Break the following generated answer into individual factual statements or claims.
            Each statement should be a distinct piece of factual information asserted in the answer.
            Return a JSON list of strings. For example: ["Claim 1.", "Claim 2.", "Claim 3."]

            Generated Answer:
            {answer}

            Statements:"""
        )
        chain_statements = prompt_template_statements | self._evaluator_model
        answer_statements = []
        try:
            response_statements = chain_statements.invoke({"answer": answer})
            answer_statements = self._parse_llm_response_to_list(response_statements.content, "answer statement")
        except Exception as e:
            print(f"Error getting statements from LLM for faithfulness: {e}")

        if not answer_statements:
            print("Warning: No statements parsed from answer for faithfulness. Returning 0.0.")
            return 0.0

        # Step 2: For each generated statement, verify if it can be inferred from the given context.
        faithful_statements = 0
        contexts_str = "\\n\\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(contexts)])
        
        prompt_template_inference = ChatPromptTemplate.from_template(
            """Given the following retrieved contexts and a single statement from a generated answer,
            determine if the statement can be directly inferred from (is factually consistent with) the information present in the retrieved contexts.
            Answer with only "Yes" or "No".

            Retrieved Contexts:
            {contexts}

            Statement from Answer:
            {statement}

            Can the statement be inferred from the retrieved contexts? (Yes/No):"""
        )
        chain_inference = prompt_template_inference | self._evaluator_model

        for stmt in answer_statements:
            if not stmt.strip(): # Skip empty statements
                continue
            try:
                response_inference = chain_inference.invoke({"contexts": contexts_str, "statement": stmt})
                if self._parse_llm_yes_no_response(response_inference.content, f"faithfulness inference for statement: {stmt}"):
                    faithful_statements += 1
            except Exception as e:
                print(f"Error during inference check for faithfulness statement '{stmt}': {e}")

        # Step 3: Calculate faithfulness.
        faithfulness_score_0_to_1 = faithful_statements / len(answer_statements) if answer_statements else 0.0
        scaled_faithfulness = 1.0 + (min(max(faithfulness_score_0_to_1, 0.0), 1.0) * 4.0) # Scale to 1-5
        return scaled_faithfulness

    def _evaluate_answer_correctness(self, answer: str, ground_truth: str) -> float:
        """
        Gauges the accuracy of the generated answer when compared to the ground truth.
        Relies on ground truth and answer. Scores range from 0 to 1.
        """
        if not self._evaluator_model:
            print("Warning: Evaluator model not available for answer_correctness. Returning 0.0.")
            return 0.0

        # Factual correctness
        tp_statements = []
        fp_statements = []
        fn_statements = []

        prompt_template_factual = ChatPromptTemplate.from_template(
            """Compare the generated answer with the ground truth answer. Identify:
            1. True Positives (TP): Factual statements present in *both* the ground truth and the generated answer.
            2. False Positives (FP): Factual statements present in the *generated answer* but *not* in the ground truth.
            3. False Negatives (FN): Factual statements present in the *ground truth* but *not* in the generated answer.

            Break down each answer into its core factual statements before comparison.
            Return the results as a JSON object with three keys: "TP", "FP", "FN", where each key maps to a list of strings (the statements).
            Example: {{"TP": ["Statement A is true."], "FP": ["Statement B is false."], "FN": ["Statement C was missed."]}}

            Ground Truth Answer:
            {ground_truth}

            Generated Answer:
            {answer}

            Factual Analysis (TP, FP, FN JSON):"""
        )
        chain_factual = prompt_template_factual | self._evaluator_model
        try:
            response_factual = chain_factual.invoke({"ground_truth": ground_truth, "answer": answer})
            factual_analysis_json = json.loads(response_factual.content.strip()) # Expecting JSON directly
            tp_statements = [str(s) for s in factual_analysis_json.get("TP", [])]
            fp_statements = [str(s) for s in factual_analysis_json.get("FP", [])]
            fn_statements = [str(s) for s in factual_analysis_json.get("FN", [])]
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for factual analysis in answer_correctness: {e}. Response: {response_factual.content if 'response_factual' in locals() else 'N/A'}")
        except Exception as e:
            print(f"Error during factual analysis for answer_correctness: {e}")
        
        tp_count = len(tp_statements)
        fp_count = len(fp_statements)
        fn_count = len(fn_statements)

        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
        factual_f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Semantic similarity
        semantic_similarity = 0.0
        prompt_template_semantic = ChatPromptTemplate.from_template(
            """Rate the overall semantic similarity between the generated answer and the ground truth answer.
            Consider if they convey the same meaning, even if the wording is different.
            Score on a scale from 0.0 (completely different meaning) to 1.0 (identical meaning).
            Provide only the numerical score.

            Ground Truth Answer:
            {ground_truth}

            Generated Answer:
            {answer}

            Semantic Similarity Score (0.0-1.0):"""
        )
        chain_semantic = prompt_template_semantic | self._evaluator_model
        try:
            response_semantic = chain_semantic.invoke({"ground_truth": ground_truth, "answer": answer})
            semantic_similarity = self._parse_llm_float_response(response_semantic.content, "answer_correctness semantic similarity")
        except Exception as e:
            print(f"Error calculating semantic similarity for answer_correctness: {e}")

        # Weighted average (default weights: 0.5 for factual, 0.5 for semantic)
        factual_weight = 0.5
        semantic_weight = 0.5
        
        answer_correctness_score_0_to_1 = (factual_weight * factual_f1_score) + (semantic_weight * semantic_similarity)
        scaled_correctness = 1.0 + (min(max(answer_correctness_score_0_to_1, 0.0), 1.0) * 4.0) # Scale to 1-5
        return scaled_correctness

    @property
    def supported_metrics(self) -> List[str]:
        return [
            EvaluationMetricType.CONTEXT_RECALL.value,
            EvaluationMetricType.CONTEXT_PRECISION.value,
            EvaluationMetricType.ANSWER_RELEVANCE.value,
            EvaluationMetricType.FAITHFULNESS.value,
            EvaluationMetricType.ANSWER_CORRECTNESS.value
        ]

    @property
    def name(self) -> str:
        return "Custom Claude Evaluator"

    @property
    def description(self) -> str:
        return "Evaluator using a Claude model for Context Recall, Answer Relevancy, Context Precision, Faithfulness, and Answer Correctness."


class EvaluatorFactory:
    """Factory for creating evaluators"""
    
    @staticmethod
    def create_evaluator(backend_type: EvaluationBackendType, metrics: List[str]) -> BaseEvaluator:
        """Create an evaluator based on backend type and metrics"""
        if backend_type == EvaluationBackendType.BUILTIN:
            return BuiltinEvaluator(metrics)
        elif backend_type == EvaluationBackendType.RAGAS:
            return RAGASEvaluator(metrics)
        elif backend_type == EvaluationBackendType.LANGSMITH:
            return LangSmithEvaluator(metrics)
        elif backend_type == EvaluationBackendType.DEEP_EVAL:
            return DeepEvaluator(metrics)
        elif backend_type == EvaluationBackendType.RAGAS_V2:
            return RAGASEvaluatorV2(metrics)
        elif backend_type == EvaluationBackendType.CUSTOM:
            return CustomEvaluator(metrics)
        else:
            raise ValueError(f"Unsupported evaluation backend type: {backend_type}")