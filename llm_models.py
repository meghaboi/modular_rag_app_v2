from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import os

from enums import LLMModelType

class LLM(ABC):
    """Abstract base class for LLM models following Interface Segregation Principle"""
    
    @abstractmethod
    def generate(self, prompt: str, context: Optional[str] = None) -> str:
        """Generate text from a prompt and optional context"""
        pass

class OpenAIGPT(LLM):
    """OpenAI GPT model implementation"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """Initialize the OpenAI GPT model"""
        from langchain_openai import ChatOpenAI
        
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key not found in environment variables")
        
        # Define system prompt for JEFF
        jeff_system_prompt = """You are JEFF, that cool friend everyone wishes they had the night before exams.
        You explain complex subjects in simple, relatable terms that just click when it matters most.
        Unlike formal professors, you break down academic concepts with perfect clarity, memorable examples, and occasional humor.
        You excel at finding the shortcuts, mnemonics, and "aha!" moments that make difficult material suddenly make sense.
        Your explanations focus on what's actually important to understand and remember, cutting through the noise.
        You're encouraging, patient, and have a knack for making anyone feel like they can ace their exam.
        Always respond as JEFF - casual but knowledgeable, relatable but authoritative, and above all, the friend who helps everyone pass their exams."""
        
        self._model = ChatOpenAI(model_name=model_name, system=jeff_system_prompt)
    
    def generate(self, prompt: str, context: Optional[str] = None) -> str:
        """Generate text from a prompt and optional context"""
        from langchain.prompts import ChatPromptTemplate
        
        if context:
            template = """
            Answer the question as JEFF, that cool friend who explains subjects better than professors do.
            Remember to be conversational, relatable, and break down complex topics into simple terms.
            Focus on the most important concepts, use memorable examples, and explain things the way you would
            the night before an exam - clear, concise, and actually helpful.
            
            Context:
            {context}
            
            Question:
            {question}
            
            Answer:
            """
            prompt_template = ChatPromptTemplate.from_template(template)
            chain = prompt_template | self._model
            response = chain.invoke({"context": context, "question": prompt})
            return response.content
        else:
            return self._model.invoke(prompt).content

class GeminiLLM(LLM):
    """Google Gemini model implementation"""
    
    def __init__(self):
        """Initialize the Google Gemini model"""
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        if not os.environ.get("GEMINI_API_KEY"):
            raise ValueError("Gemini API key not found in environment variables")
        
        # Define system prompt for JEFF
        jeff_system_prompt = """You are JEFF, that cool friend everyone wishes they had the night before exams.
        You explain complex subjects in simple, relatable terms that just click when it matters most.
        Unlike formal professors, you break down academic concepts with perfect clarity, memorable examples, and occasional humor.
        You excel at finding the shortcuts, mnemonics, and "aha!" moments that make difficult material suddenly make sense.
        Your explanations focus on what's actually important to understand and remember, cutting through the noise.
        You're encouraging, patient, and have a knack for making anyone feel like they can ace their exam.
        Always respond as JEFF - casual but knowledgeable, relatable but authoritative, and above all, the friend who helps everyone pass their exams."""
        
        self._model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp", 
            google_api_key=os.environ.get("GEMINI_API_KEY"),
            system_instruction=jeff_system_prompt
        )
    
    def generate(self, prompt: str, context: Optional[str] = None) -> str:
        """Generate text from a prompt and optional context"""
        from langchain.prompts import ChatPromptTemplate
        
        if context:
            template = """
            Answer the question as JEFF, that cool friend who explains subjects better than professors do.
            Remember to be conversational, relatable, and break down complex topics into simple terms.
            Focus on the most important concepts, use memorable examples, and explain things the way you would
            the night before an exam - clear, concise, and actually helpful.
            
            Context:
            {context}
            
            Question:
            {question}
            
            Answer:
            """
            prompt_template = ChatPromptTemplate.from_template(template)
            chain = prompt_template | self._model
            response = chain.invoke({"context": context, "question": prompt})
            return response.content
        else:
            return self._model.invoke(prompt).content

class ClaudeLLM(LLM):
    """Anthropic Claude model implementation"""
    
    def __init__(self, model_name: str = "claude-3-5-sonnet-20240229"):
        """Initialize the Anthropic Claude model"""
        from langchain_anthropic import ChatAnthropic
        
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError("Anthropic API key not found in environment variables")
        
        jeff_system_prompt = """You are JEFF, that cool friend everyone wishes they had the night before exams.
        You explain complex subjects in simple, relatable terms that just click when it matters most.
        Unlike formal professors, you break down academic concepts with perfect clarity, memorable examples, and occasional humor.
        You excel at finding the shortcuts, mnemonics, and "aha!" moments that make difficult material suddenly make sense.
        Your explanations focus on what's actually important to understand and remember, cutting through the noise.
        You're encouraging, patient, and have a knack for making anyone feel like they can ace their exam.
        Always respond as JEFF - casual but knowledgeable, relatable but authoritative, and above all, the friend who helps everyone pass their exams."""
        
        self._model = ChatAnthropic(model=model_name, system=jeff_system_prompt)
    
    def generate(self, prompt: str, context: Optional[str] = None) -> str:
        """Generate text from a prompt and optional context"""
        from langchain.prompts import ChatPromptTemplate
        
        if context:
            template = """
            Answer the question as JEFF, that cool friend who explains subjects better than professors do.
            Remember to be conversational, relatable, and break down complex topics into simple terms.
            Focus on the most important concepts, use memorable examples, and explain things the way you would
            the night before an exam - clear, concise, and actually helpful.
            
            Context:
            {context}
            
            Question:
            {question}
            
            Answer:
            """
            prompt_template = ChatPromptTemplate.from_template(template)
            chain = prompt_template | self._model
            response = chain.invoke({"context": context, "question": prompt})
            return response.content
        else:
            return self._model.invoke(prompt).content

class MistralLLM(LLM):
    """Mistral model implementation"""
    
    def __init__(self, model_name: str = "mistral-large-latest"):
        """Initialize the Mistral model"""
        from langchain_mistralai import ChatMistralAI
        
        if not os.environ.get("MISTRAL_API_KEY"):
            raise ValueError("Mistral API key not found in environment variables")
        
        jeff_system_prompt = """You are JEFF, that cool friend everyone wishes they had the night before exams.
        You explain complex subjects in simple, relatable terms that just click when it matters most.
        Unlike formal professors, you break down academic concepts with perfect clarity, memorable examples, and occasional humor.
        You excel at finding the shortcuts, mnemonics, and "aha!" moments that make difficult material suddenly make sense.
        Your explanations focus on what's actually important to understand and remember, cutting through the noise.
        You're encouraging, patient, and have a knack for making anyone feel like they can ace their exam.
        Always respond as JEFF - casual but knowledgeable, relatable but authoritative, and above all, the friend who helps everyone pass their exams."""
        
        self._model = ChatMistralAI(model=model_name, system=jeff_system_prompt)
    
    def generate(self, prompt: str, context: Optional[str] = None) -> str:
        """Generate text from a prompt and optional context"""
        from langchain.prompts import ChatPromptTemplate
        
        if context:
            template = """
            Answer the question as JEFF, that cool friend who explains subjects better than professors do.
            Remember to be conversational, relatable, and break down complex topics into simple terms.
            Focus on the most important concepts, use memorable examples, and explain things the way you would
            the night before an exam - clear, concise, and actually helpful.
            
            Context:
            {context}
            
            Question:
            {question}
            
            Answer:
            """
            prompt_template = ChatPromptTemplate.from_template(template)
            chain = prompt_template | self._model
            response = chain.invoke({"context": context, "question": prompt})
            return response.content
        else:
            return self._model.invoke(prompt).content

class LLMFactory:
    """Factory for creating LLM models (Factory Pattern)"""
    
    @staticmethod
    def create_llm(model_type: LLMModelType) -> LLM:
        """Create an LLM model based on the model type"""
        if model_type == LLMModelType.OPENAI_GPT35:
            return OpenAIGPT(model_name="gpt-3.5-turbo")
        elif model_type == LLMModelType.OPENAI_GPT4:
            return OpenAIGPT(model_name="gpt-4")
        elif model_type == LLMModelType.GEMINI:
            return GeminiLLM()
        elif model_type == LLMModelType.CLAUDE_3_OPUS:
            return ClaudeLLM(model_name="claude-3-opus-20240229") 
        elif model_type == LLMModelType.CLAUDE_37_SONNET:
            return ClaudeLLM(model_name="claude-3-7-sonnet-20250219")
        elif model_type == LLMModelType.MISTRAL_LARGE:
            return MistralLLM(model_name="mistral-large-latest")
        elif model_type == LLMModelType.MISTRAL_MEDIUM:
            return MistralLLM(model_name="mistral-medium-latest")
        elif model_type == LLMModelType.MISTRAL_SMALL:
            return MistralLLM(model_name="mistral-small-latest")
        else:
            raise ValueError(f"Unsupported LLM model: {model_type}")