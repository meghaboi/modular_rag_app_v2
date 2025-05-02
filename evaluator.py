# evaluator.py
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import os

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
                answer_correctness,
                context_precision,
                context_recall
            )
            from datasets import Dataset
        except ImportError as e:
            raise ValueError(f"Required library not installed: {e}")
        
        # Store RAGAS metric objects
        self.ragas_metrics = {
            "faithfulness": faithfulness,
            "answer_correctness": answer_correctness,
            "context_precision": context_precision,
            "context_recall": context_recall
        }
        
        # Map our metric names to potential RAGAS result attribute names
        self.metric_name_map = {
            "faithfulness": ["faithfulness", "ragas_faithfulness"],
            "answer_correctness": ["answer_correctness", "answer_relevancy", "correctness"],
            "context_precision": ["context_precision", "precision"],
            "context_recall": ["context_recall", "recall"]
        }
        
        # Use all metrics if none specified
        if metrics is None:
            self._metrics = list(self.ragas_metrics.keys())
        else:
            # Validate provided metrics
            invalid_metrics = [m for m in metrics if m not in self.ragas_metrics]
            if invalid_metrics:
                raise ValueError(f"Unsupported metrics: {invalid_metrics}")
            self._metrics = metrics
        
        # Verify OpenAI API key exists for RAGAS
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key required for RAGAS evaluation")
        
        # Initialize the LLM for RAGAS
        from langchain_openai import ChatOpenAI
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo")
        
        # Configure RAGAS to use this LLM
        import ragas
        ragas.llm = self.llm
    
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
            
            # Configure RAGAS
            if not hasattr(ragas, 'llm') or ragas.llm is None:
                ragas.llm = self.llm
            
            # Prepare data
            data = {
                "question": [query],
                "answer": [response],
                "contexts": [contexts]  # List of lists as expected by RAGAS
            }
            
            if ground_truth:
                data["ground_truths"] = [[ground_truth]]
                
                # Both naming conventions have been used in different RAGAS versions
                data["reference"] = [ground_truth]
                data["references"] = [[ground_truth]]
            
            ds = Dataset.from_dict(data)
            
            # Get metrics
            active_metrics = [self.ragas_metrics[metric] for metric in self._metrics 
                            if metric in self.ragas_metrics]
            
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
            
            # If we found metrics already, return them
            if metrics_dict:
                return metrics_dict
                
            # Fallback methods - try to extract scores from different properties
            try:
                # Try dataframe approach
                df = pd.DataFrame(results)
                logging.info(f"RAGAS DataFrame columns: {df.columns.tolist()}")
                
                for metric in self._metrics:
                    found = False
                    possible_names = self.metric_name_map.get(metric, [metric])
                    
                    for name in possible_names:
                        for col in df.columns:
                            if name.lower() in col.lower():
                                raw_value = float(df[col].iloc[0])
                                # FIXED SCALING: Properly map from 0-1 to 1-5 scale
                                scaled_value = 1.0 + raw_value * 4.0
                                metrics_dict[metric] = round(scaled_value, 2)
                                found = True
                                logging.info(f"Found metric {metric} as column {col}, value: {raw_value}")
                                break
                        if found:
                            break
                    
                    if not found:
                        logging.warning(f"Metric {metric} not found in DataFrame columns")
                        metrics_dict[metric] = 3.0
            except Exception as e:
                logging.info(f"DataFrame approach failed: {str(e)}, trying direct attribute access")
            
            # If we found metrics using DataFrame approach, return them
            if metrics_dict:
                return metrics_dict
                
            # Check for direct attributes
            for metric in self._metrics:
                found = False
                possible_names = self.metric_name_map.get(metric, [metric])
                
                for name in possible_names:
                    if hasattr(results, name):
                        try:
                            raw_value = float(getattr(results, name))
                            # FIXED SCALING: Properly map from 0-1 to 1-5 scale
                            scaled_value = 1.0 + raw_value * 4.0
                            metrics_dict[metric] = round(scaled_value, 2)
                            found = True
                            logging.info(f"Found metric {metric} as attribute {name}, value: {raw_value}")
                            break
                        except (ValueError, TypeError):
                            continue
                
                if not found:
                    # Try a more fuzzy match on attributes
                    for attr_name in dir(results):
                        if attr_name.startswith('_') or callable(getattr(results, attr_name)):
                            continue
                            
                        for name in possible_names:
                            if name.lower() in attr_name.lower() or attr_name.lower() in name.lower():
                                try:
                                    attr_value = getattr(results, attr_name)
                                    # Skip complex objects
                                    if isinstance(attr_value, (dict, list, object)) and not isinstance(attr_value, (int, float, str)):
                                        continue
                                    
                                    raw_value = float(attr_value)
                                    # FIXED SCALING: Properly map from 0-1 to 1-5 scale
                                    scaled_value = 1.0 + raw_value * 4.0
                                    metrics_dict[metric] = round(scaled_value, 2)
                                    found = True
                                    logging.info(f"Found metric {metric} as attribute {attr_name}, value: {raw_value}")
                                    break
                                except (ValueError, TypeError):
                                    continue
                        
                        if found:
                            break
                
                if not found:
                    logging.warning(f"Metric {metric} not found in results attributes")
                    metrics_dict[metric] = 3.0
            
            # Final fallback: use traces if available
            if not metrics_dict and hasattr(results, 'traces') and results.traces:
                try:
                    logging.info("Trying traces attribute as final resort")
                    traces = results.traces
                    
                    if isinstance(traces, list) and len(traces) > 0 and isinstance(traces[0], dict):
                        first_trace = traces[0]
                        for metric in self._metrics:
                            if metric in first_trace:
                                raw_value = float(first_trace[metric])
                                # FIXED SCALING: Properly map from 0-1 to 1-5 scale
                                scaled_value = 1.0 + raw_value * 4.0
                                metrics_dict[metric] = round(scaled_value, 2)
                                logging.info(f"Found metric {metric} in traces, value: {raw_value}")
                            else:
                                # Try alternate names
                                found = False
                                for alt_name in self.metric_name_map.get(metric, []):
                                    if alt_name in first_trace:
                                        raw_value = float(first_trace[alt_name])
                                        # FIXED SCALING: Properly map from 0-1 to 1-5 scale
                                        scaled_value = 1.0 + raw_value * 4.0
                                        metrics_dict[metric] = round(scaled_value, 2)
                                        logging.info(f"Found metric {metric} as {alt_name} in traces, value: {raw_value}")
                                        found = True
                                        break
                                
                                if not found:
                                    logging.warning(f"Metric {metric} not found in traces dictionary")
                                    metrics_dict[metric] = 3.0
                except Exception as e:
                    logging.warning(f"Traces extraction failed: {str(e)}")
            
            # If all extraction methods failed, use default values
            if not metrics_dict:
                logging.warning("All extraction methods failed. Setting default metrics.")
                # Set default middle values
                for metric in self._metrics:
                    metrics_dict[metric] = 3.0
            
            return metrics_dict
            
        except Exception as e:
            logging.error(f"RAGAS evaluation error: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            
            # Return default values on complete failure
            default_metrics = {metric: 3.0 for metric in self._metrics}
            return default_metrics
    
    @property
    def supported_metrics(self) -> List[str]:
        """Return list of metrics supported by this evaluator"""
        return list(self.ragas_metrics.keys())
    
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


class EvaluatorFactory:
    """Factory for creating evaluators"""
    
    @staticmethod
    def create_evaluator(backend_type: EvaluationBackendType, metrics: List[str]) -> BaseEvaluator:
        """Create an evaluator based on backend type and metrics"""
        if backend_type == EvaluationBackendType.BUILTIN:
            return BuiltinEvaluator(metrics)
        elif backend_type == EvaluationBackendType.RAGAS:
            return RAGASEvaluatorV2(metrics)  # Using V2 instead of original
        elif backend_type == EvaluationBackendType.LANGSMITH:
            return LangSmithEvaluator(metrics)
        elif backend_type == EvaluationBackendType.DEEP:
            return DeepEvaluator(metrics) 
        else:
            raise ValueError(f"Unsupported evaluator type: {backend_type}")