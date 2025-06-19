# evaluator.py
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import os
import re
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from prompts.prompt_providers import get_provider

from utils.enums import EvaluationBackendType, EvaluationMetricType

class BaseEvaluator(ABC):
    """Abstract base class for RAG evaluators"""
    
    def __init__(self, metrics: List[str]):
        """Initialize with selected metrics"""
        self._metrics = metrics
    
    @abstractmethod
    def evaluate(self, query: str, response: str, contexts: List[str], 
                 ground_truth: Optional[str] = None, cost: Optional[float] = None) -> Dict[str, float]:
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
        self._prompt_provider = get_provider('evaluator')
    
    def evaluate(self, query: str, response: str, contexts: List[str], 
                 ground_truth: Optional[str] = None, cost: Optional[float] = None) -> Dict[str, float]:
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
        
        if cost is not None:
            results[EvaluationMetricType.COST.value] = cost
        
        return results
    
    def _evaluate_answer_relevance(self, query: str, response: str, ground_truth: Optional[str] = None) -> float:
        """Evaluate the relevance of the answer to the query"""
        prompt = self._prompt_provider.get_prompt(
            'answer_relevance',
            query=query,
            response=response,
            ground_truth=ground_truth
        )
        prompt_template = ChatPromptTemplate.from_template(prompt)
        
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
        context_text = "\n\n".join([f"Context {i+1}: {context}" for i, context in enumerate(contexts)])
        prompt = self._prompt_provider.get_prompt(
            'context_relevance',
            query=query,
            contexts=context_text
        )
        prompt_template = ChatPromptTemplate.from_template(prompt)
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
        context_text = "\n\n".join([f"Context {i+1}: {context}" for i, context in enumerate(contexts)])
        prompt = self._prompt_provider.get_prompt(
            'groundedness',
            response=response,
            contexts=context_text
        )
        prompt_template = ChatPromptTemplate.from_template(prompt)
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
        context_text = "\n\n".join([f"Context {i+1}: {context}" for i, context in enumerate(contexts)])
        prompt = self._prompt_provider.get_prompt(
            'faithfulness',
            response=response,
            contexts=context_text
        )
        prompt_template = ChatPromptTemplate.from_template(prompt)
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
        self._prompt_provider = get_provider('evaluator')
        
        # Define supported metrics
        self._supported_metrics = [
            EvaluationMetricType.ANSWER_RELEVANCE.value,
            EvaluationMetricType.CONTEXT_RELEVANCE.value,
            EvaluationMetricType.GROUNDEDNESS.value,
            EvaluationMetricType.FAITHFULNESS.value
        ]
    
    def evaluate(self, query: str, response: str, contexts: List[str], 
                 ground_truth: Optional[str] = None, cost: Optional[float] = None) -> Dict[str, float]:
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
        
        if cost is not None:
            results[EvaluationMetricType.COST.value] = cost
        
        return results
    
    def _evaluate_answer_relevance(self, query: str, response: str, ground_truth: Optional[str] = None) -> float:
        """Evaluate the relevance of the answer to the query using LangSmith-inspired prompts"""
        from langchain.prompts import ChatPromptTemplate
        
        template = self._prompt_provider.get_prompt(
            'answer_relevance',
            query=query,
            response=response,
            ground_truth=ground_truth
        )
        
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
        
        template = self._prompt_provider.get_prompt(
            'context_relevance',
            query=query,
            contexts=context_text
        )
        
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
        
        template = self._prompt_provider.get_prompt(
            'groundedness',
            response=response,
            contexts=context_text
        )
        
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
        
        template = self._prompt_provider.get_prompt(
            'faithfulness',
            response=response,
            contexts=context_text
        )
        
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

        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key required for DeepEvaluator")

        self._general_evaluator = ChatOpenAI(model_name="gpt-3.5-turbo")
        self._deep_evaluator = None
        if os.environ.get("ANTHROPIC_API_KEY"):
            self._deep_evaluator = ChatAnthropic(model="claude-3-haiku-20240307")
        elif os.environ.get("MISTRAL_API_KEY"):
            self._deep_evaluator = ChatMistralAI(model="mistral-small")
        else:
            self._deep_evaluator = self._general_evaluator

        self._metric_to_model = {
            "answer_relevance": self._general_evaluator,
            "context_relevance": self._general_evaluator,
            "groundedness": self._deep_evaluator,
            "faithfulness": self._deep_evaluator,
            "answer_consistency": self._deep_evaluator,
            "context_coverage": self._general_evaluator
        }
        
        self._prompt_provider = get_provider('evaluator')

    def evaluate(self, query: str, response: str, contexts: List[str],
                 ground_truth: Optional[str] = None, cost: Optional[float] = None) -> Dict[str, float]:
        """Evaluate RAG system performance using selected metrics with specialized models"""
        results = {}
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
                    results[metric] = self._evaluate_answer_consistency(response)
                elif metric == "context_coverage":
                    results[metric] = self._evaluate_context_coverage(query, contexts)
            except Exception as e:
                print(f"Error evaluating {metric}: {str(e)}")
                results[metric] = 3.0

        if cost is not None:
            results[EvaluationMetricType.COST.value] = cost
        return results

    def _extract_score_from_response(self, response_text: str) -> float:
        """Extract a numeric score from LLM response with improved robustness."""
        try:
            score = float(response_text.strip())
            return min(max(score, 1), 5)
        except ValueError:
            pass

        number_matches = re.findall(r'\d+\.?\d*', response_text)
        if number_matches:
            try:
                score = float(number_matches[0])
                if 1 <= score <= 5:
                    return score
            except ValueError:
                pass
        
        lower_text = response_text.lower()
        if "score: " in lower_text:
            try:
                score_text = lower_text.split("score: ")[1].split()[0]
                score = float(score_text)
                return min(max(score, 1), 5)
            except (ValueError, IndexError):
                pass

        return 3.0

    def _evaluate_answer_relevance(self, query: str, response: str, ground_truth: Optional[str] = None) -> float:
        """Evaluate the relevance of the answer to the query"""
        model = self._metric_to_model["answer_relevance"]
        prompt = self._prompt_provider.get_prompt(
            'answer_relevance',
            query=query,
            response=response,
            ground_truth=ground_truth
        )
        prompt_template = ChatPromptTemplate.from_template(prompt)
        chain = prompt_template | model
        
        invoke_params = {"query": query, "response": response}
        if ground_truth:
            invoke_params["ground_truth"] = ground_truth
            
        response_obj = chain.invoke(invoke_params)
        return self._extract_score_from_response(response_obj.content)

    def _evaluate_context_relevance(self, query: str, contexts: List[str]) -> float:
        """Evaluate the relevance of the contexts to the query"""
        model = self._metric_to_model["context_relevance"]
        context_text = "\n\n".join([f"Context {i+1}: {context}" for i, context in enumerate(contexts)])
        prompt = self._prompt_provider.get_prompt(
            'context_relevance',
            query=query,
            contexts=context_text
        )
        prompt_template = ChatPromptTemplate.from_template(prompt)
        chain = prompt_template | model
        response_obj = chain.invoke({"query": query, "contexts": context_text})
        return self._extract_score_from_response(response_obj.content)

    def _evaluate_groundedness(self, response: str, contexts: List[str]) -> float:
        """Evaluate if the response is grounded in the provided contexts"""
        model = self._metric_to_model["groundedness"]
        context_text = "\n\n".join([f"Context {i+1}: {context}" for i, context in enumerate(contexts)])
        prompt = self._prompt_provider.get_prompt(
            'groundedness',
            response=response,
            contexts=context_text
        )
        prompt_template = ChatPromptTemplate.from_template(prompt)
        chain = prompt_template | model
        response_obj = chain.invoke({"response": response, "contexts": context_text})
        return self._extract_score_from_response(response_obj.content)

    def _evaluate_faithfulness(self, response: str, contexts: List[str]) -> float:
        """Evaluate the faithfulness of the response to the provided contexts"""
        model = self._metric_to_model["faithfulness"]
        context_text = "\n\n".join([f"Context {i+1}: {context}" for i, context in enumerate(contexts)])
        prompt = self._prompt_provider.get_prompt(
            'faithfulness',
            response=response,
            contexts=context_text
        )
        prompt_template = ChatPromptTemplate.from_template(prompt)
        chain = prompt_template | model
        response_obj = chain.invoke({"response": response, "contexts": context_text})
        return self._extract_score_from_response(response_obj.content)

    def _evaluate_answer_consistency(self, response: str) -> float:
        """Custom metric: Evaluate the internal consistency of the answer"""
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
        response_obj = chain.invoke({"response": response})
        return self._extract_score_from_response(response_obj.content)

    def _evaluate_context_coverage(self, query: str, contexts: List[str]) -> float:
        """Custom metric: Evaluate how well the contexts cover different aspects of the query"""
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
        response_obj = chain.invoke({"query": query, "contexts": context_text})
        return self._extract_score_from_response(response_obj.content)

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
                ground_truth: Optional[str] = None, cost: Optional[float] = None) -> Dict[str, float]:
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

            chatLLM = ChatOpenAI(
            model="gpt-4o",
            temperature=0.0,
            )

            # Run evaluation
            results = ragas_evaluate(ds, metrics=active_metrics, llm=chatLLM)
            
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
            
            if cost is not None:
                metrics_dict[EvaluationMetricType.COST.value] = cost
            
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
            # The ClaudeLLM class checks for the API key internally
            from models.llm_models import ClaudeLLM
            self._evaluator_model = ClaudeLLM(model_name="claude-3-opus-20240229")
            self._prompt_provider = get_provider('evaluator')
        except ImportError:
            raise ValueError("models.llm_models.ClaudeLLM could not be imported.")
        except Exception as e:
            print(f"Error initializing Claude model for evaluator: {e}. Evaluation will not work.")
            self._evaluator_model = None

        if self._evaluator_model is None:
            raise ValueError("Claude model (claude-3-opus-20240229) could not be initialized. Please check ANTHROPIC_API_KEY.")

    def _parse_llm_response_to_list(self, response_content: str, item_type: str = "statement") -> List[str]:
        """Parses LLM response (expected to be a list of items) into a Python list of strings."""
        try:
            # Use regex to find the JSON list within the response text
            match = re.search(r'\[(.*?)\]', response_content, re.DOTALL)
            if match:
                json_str = match.group(0)
                parsed_list = json.loads(json_str)
                if isinstance(parsed_list, list) and all(isinstance(item, str) for item in parsed_list):
                    return parsed_list
        except json.JSONDecodeError:
            # Fallback for non-JSON-compliant but simple list-like strings
            pass
        
        # Fallback for simple newline-separated lists
        if '\n' in response_content:
            return [line.strip() for line in response_content.split('\n') if line.strip()]

        print(f"Could not parse LLM response into a list of {item_type}s. Response: {response_content}")
        return []

    def _parse_llm_yes_no_response(self, response_content: str, prompt_details: str) -> bool:
        """Parses LLM response to a boolean for Yes/No questions."""
        normalized_response = response_content.strip().upper()
        if 'YES' in normalized_response:
            return True
        if 'NO' in normalized_response:
            return False
        
        print(f"Could not parse 'Yes' or 'No' from LLM response for {prompt_details}. Response: {response_content}")
        return False # Default to 'No' on ambiguity

    def _parse_llm_float_response(self, response_content: str, prompt_details: str) -> float:
        """Parses LLM response to a float, expected to be between 0 and 1."""
        try:
            # Find a floating-point number in the response
            match = re.search(r'\d*\.\d+', response_content)
            if match:
                return float(match.group())
            # Fallback for integer '0' or '1'
            if '1' in response_content:
                return 1.0
            if '0' in response_content:
                return 0.0
        except ValueError:
            pass
        
        print(f"Could not parse float from LLM response for {prompt_details}. Response: {response_content}")
        return 0.0 # Default to 0.0 on ambiguity

    def evaluate(self, query: str, response: str, contexts: List[str],
                 ground_truth: Optional[str] = None, cost: Optional[float] = None) -> Dict[str, float]:
        """Evaluate RAG system performance using selected metrics"""
        results = {}
        for metric in self._metrics:
            metric_value = 0.0
            # Ensure ground truth is available for metrics that require it
            if metric in ["context_recall", "context_precision", "answer_correctness"] and not ground_truth:
                print(f"Skipping metric '{metric}' as it requires a ground truth answer.")
                continue

            if metric == "context_recall":
                metric_value = self._evaluate_context_recall(query, ground_truth, contexts)
            elif metric == "context_precision":
                metric_value = self._evaluate_context_precision(query, ground_truth, contexts)
            elif metric == "answer_relevancy":
                metric_value = self._evaluate_answer_relevancy(query, response, contexts)
            elif metric == "faithfulness":
                metric_value = self._evaluate_faithfulness(response, contexts)
            elif metric == "answer_correctness":
                metric_value = self._evaluate_answer_correctness(response, ground_truth)
            
            # Scale from 0-1 range to 1-5 range for consistent reporting
            results[metric] = 1.0 + metric_value * 4.0
        
        if cost is not None:
            results[EvaluationMetricType.COST.value] = cost
            
        return results

    def _evaluate_context_recall(self, query: str, ground_truth: str, contexts: List[str]) -> float:
        """
        Measures the extent to which the retrieved context aligns with the annotated answer (ground truth).
        Computed using question, ground truth, and retrieved context. Values range from 0 to 1.
        """
        # Step 1: Extract statements from the ground truth
        prompt_statements = self._prompt_provider.get_prompt(
            'context_recall_statements',
            ground_truth=ground_truth
        )
        response_statements_text, _ = self._evaluator_model.generate(prompt=prompt_statements, evaluation_mode=True)
        statements = self._parse_llm_response_to_list(response_statements_text, "statement")
        if not statements:
            return 0.0

        # Step 2: For each statement, check if it's supported by the contexts
        supported_statements = 0
        for statement in statements:
            prompt_check = self._prompt_provider.get_prompt(
                'context_recall_attribution',
                statement=statement,
                context="\n\n".join(contexts)
            )
            response_check_text, _ = self._evaluator_model.generate(prompt=prompt_check, evaluation_mode=True)
            if self._parse_llm_yes_no_response(response_check_text, f"recall check for statement: '{statement}'"):
                supported_statements += 1

        # Step 3: Calculate recall score
        recall_score = supported_statements / len(statements) if statements else 0.0
        return recall_score

    def _evaluate_context_precision(self, query: str, ground_truth: str, contexts: List[str]) -> float:
        """
        Evaluates whether all ground-truth relevant items in contexts are ranked higher.
        Computed using question, ground_truth, and contexts. Values range from 0 to 1.
        """
        # Step 1: Extract statements from the ground truth
        prompt_statements = self._prompt_provider.get_prompt(
            'context_precision_statements',
            ground_truth=ground_truth
        )
        response_statements_text, _ = self._evaluator_model.generate(prompt=prompt_statements, evaluation_mode=True)
        statements = self._parse_llm_response_to_list(response_statements_text, "statement")
        if not statements:
            return 1.0 # No statements to check, so precision is vacuously high

        # Step 2: For each context, check if it is relevant
        relevant_contexts = 0
        for context in contexts:
            prompt_check = self._prompt_provider.get_prompt(
                'context_precision_relevance',
                query=query,
                context=context
            )
            response_check_text, _ = self._evaluator_model.generate(prompt=prompt_check, evaluation_mode=True)
            if self._parse_llm_yes_no_response(response_check_text, f"precision check for context: '{context[:100]}...'"):
                relevant_contexts += 1

        # Step 3: Calculate precision score
        precision_score = relevant_contexts / len(contexts) if contexts else 0.0
        return precision_score

    def _evaluate_answer_relevancy(self, query: str, answer: str, contexts: List[str]) -> float:
        """
        Assesses how pertinent the generated answer is to the given prompt.
        Computed using the question, the context and the answer. Values range from 0 to 1.
        """
        prompt = self._prompt_provider.get_prompt(
            'answer_relevancy',
            question=query,
            answer=answer,
            context="\n\n".join(contexts)
        )
        response_text, _ = self._evaluator_model.generate(prompt=prompt, evaluation_mode=True)
        return self._parse_llm_float_response(response_text, "answer relevancy")

    def _evaluate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """
        Measures the factual consistency of the generated answer against the given context.
        Calculated from answer and retrieved context. Scaled to (0,1) range.
        """
        # Step 1: Extract statements from the answer
        prompt_statements = self._prompt_provider.get_prompt(
            'faithfulness_statements',
            answer=answer
        )
        response_statements_text, _ = self._evaluator_model.generate(prompt=prompt_statements, evaluation_mode=True)
        statements = self._parse_llm_response_to_list(response_statements_text, "statement")
        if not statements:
            return 1.0 # No statements to check, so faithfulness is vacuously high

        # Step 2: For each statement, check if it's supported by the contexts
        faithful_statements = 0
        for statement in statements:
            prompt_check = self._prompt_provider.get_prompt(
                'faithfulness_verification',
                statement=statement,
                context="\n\n".join(contexts)
            )
            response_check_text, _ = self._evaluator_model.generate(prompt=prompt_check, evaluation_mode=True)
            if self._parse_llm_yes_no_response(response_check_text, f"faithfulness check for statement: '{statement}'"):
                faithful_statements += 1

        # Step 3: Calculate faithfulness score
        faithfulness_score = faithful_statements / len(statements) if statements else 0.0
        return faithfulness_score

    def _evaluate_answer_correctness(self, answer: str, ground_truth: str) -> float:
        """
        Gauges the accuracy of the generated answer when compared to the ground truth.
        Relies on ground truth and answer. Scores range from 0 to 1.
        """
        # This is a multi-step process:
        # 1. Factual comparison (precision, recall, F1 over statements)
        # 2. Semantic similarity comparison
        # 3. Weighted average of the two scores

        # Step 1.1: Extract factual statements from both answer and ground truth
        prompt_factual = self._prompt_provider.get_prompt(
            'answer_correctness_factual',
            ground_truth=ground_truth,
            answer=answer
        )
        response_factual_text, _ = self._evaluator_model.generate(prompt=prompt_factual, evaluation_mode=True)
        
        tp_count = 0
        fp_count = 0
        fn_count = 0
        try:
            # The prompt asks for a JSON object with TP, FP, FN counts.
            analysis = json.loads(response_factual_text)
            tp_count = int(analysis.get("TP", 0))
            fp_count = int(analysis.get("FP", 0))
            fn_count = int(analysis.get("FN", 0))
        except (json.JSONDecodeError, ValueError):
             print(f"Warning: Could not parse factual analysis from LLM for answer correctness. Response: {response_factual_text}")

        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
        factual_f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Step 2: Semantic similarity comparison
        prompt_semantic = self._prompt_provider.get_prompt(
            'answer_correctness_semantic',
            ground_truth=ground_truth,
            answer=answer
        )
        response_semantic_text, _ = self._evaluator_model.generate(prompt=prompt_semantic, evaluation_mode=True)
        semantic_similarity = self._parse_llm_float_response(response_semantic_text, "answer correctness semantic similarity")
        
        # Step 3: Weighted average (giving more weight to factual correctness)
        factual_weight = 0.6
        semantic_weight = 0.4
        final_score = (factual_weight * factual_f1_score) + (semantic_weight * semantic_similarity)
        
        return final_score

    @property
    def supported_metrics(self) -> List[str]:
        return [
            EvaluationMetricType.CONTEXT_RECALL.value,
            EvaluationMetricType.CONTEXT_PRECISION.value,
            EvaluationMetricType.ANSWER_RELEVANCY.value,
            EvaluationMetricType.FAITHFULNESS.value,
            EvaluationMetricType.ANSWER_CORRECTNESS.value
        ]

    @property
    def name(self) -> str:
        return "Custom LLM Evaluator (Claude 3 Opus)"

    @property
    def description(self) -> str:
        return "Uses Claude 3 Opus to perform a series of checks for detailed, multi-step evaluation."

class EvaluatorFactory:
    """Factory for creating evaluators"""
    
    @staticmethod
    def create_evaluator(backend_type: EvaluationBackendType, metrics: List[str]) -> BaseEvaluator:
        """Create an evaluator based on backend type and metrics"""
        if backend_type == EvaluationBackendType.BUILTIN:
            return BuiltinEvaluator(metrics)
        elif backend_type == EvaluationBackendType.RAGAS:
            return RAGASEvaluatorV2(metrics)
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