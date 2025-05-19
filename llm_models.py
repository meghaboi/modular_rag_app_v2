from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable, Iterator
import os

from enums import LLMModelType

class StreamingLLM(ABC):
    """Abstract base class for streaming LLM models"""
    
    @abstractmethod
    def generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False) -> str:
        """Generate text from a prompt and optional context"""
        pass
    
    @abstractmethod
    def stream_generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False) -> Iterator[str]:
        """Stream generate text from a prompt and optional context"""
        pass

class OpenAIGPT(StreamingLLM):
    """OpenAI GPT model implementation with streaming support"""
    
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
        
        self._model = ChatOpenAI(
            model_name=model_name,
            streaming=False,
            model_kwargs={"system": jeff_system_prompt}
        )
        self._streaming_model = ChatOpenAI(
            model_name=model_name,
            streaming=True,
            model_kwargs={"system": jeff_system_prompt}
        )
        self._jeff_system_prompt = jeff_system_prompt
        self._cache = {}  # Initialize cache dictionary

    def generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False) -> str:
        cache_key = (prompt, context, evaluation_mode)
        if cache_key in self._cache:
            return self._cache[cache_key]
        """Generate text from a prompt and optional context"""
        from langchain.prompts import ChatPromptTemplate
        
        # In evaluation mode, don't use system prompt or JEFF persona
        if evaluation_mode:
            if context:
                template = """
                Context:
                {context}
                
                Question:
                {question}
                
                Answer:
                """
            else:
                return self._model.invoke(prompt).content
        else:
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
            else:
                return self._model.invoke(prompt).content
        
        prompt_template = ChatPromptTemplate.from_template(template)
        chain = prompt_template | self._model
        response = chain.invoke({"context": context, "question": prompt})
        self._cache[cache_key] = response.content  # Cache the result
        return response.content
    
    def stream_generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False) -> Iterator[str]:
        """Stream generate text from a prompt and optional context"""
        from langchain.prompts import ChatPromptTemplate
        
        # In evaluation mode, don't use system prompt or JEFF persona
        if evaluation_mode:
            if context:
                template = """
                Context:
                {context}
                
                Question:
                {question}
                
                Answer:
                """
            else:
                for chunk in self._streaming_model.stream(prompt):
                    yield chunk.content
                return
        else:
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
            else:
                for chunk in self._streaming_model.stream(prompt):
                    yield chunk.content
                return
        
        prompt_template = ChatPromptTemplate.from_template(template)
        chain = prompt_template | self._streaming_model
        
        for chunk in chain.stream({"context": context, "question": prompt}):
            yield chunk.content

class GeminiLLM(StreamingLLM):
    """Google Gemini model implementation with streaming support and native caching"""
    
    def __init__(self, ttl: str = "3600s"):
        """Initialize the Google Gemini model
        
        Args:
            ttl: Time to live for the cache in seconds, formatted as string (e.g. "3600s")
        """
        from langchain_google_genai import ChatGoogleGenerativeAI
        import os
        from google import genai
        
        if not os.environ.get("GEMINI_API_KEY"):
            raise ValueError("Gemini API key not found in environment variables")
        
        # Initialize Google Generative AI client
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        
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
            streaming=False
        )
        
        self._streaming_model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp", 
            google_api_key=os.environ.get("GEMINI_API_KEY"),
            streaming=True
        )
        
        self._jeff_system_prompt = jeff_system_prompt
        self._ttl = ttl
        self._cache_mapping = {}  # Maps context hash to cache name
    
    def _get_or_create_cache(self, context: str) -> str:
        """Get existing cache or create a new one for the context
        
        Args:
            context: The context text to create a cache for
            
        Returns:
            The cache name to use with the Gemini API
        """
        import hashlib
        from google.genai import types
        
        # Create a hash of the context to use as a unique identifier
        if not context:
            return None
            
        context_hash = hashlib.md5(context.encode()).hexdigest()
        
        # Check if we already have a cache for this context
        if context_hash in self._cache_mapping:
            return self._cache_mapping[context_hash]
        
        # Create a new cache with the specified TTL
        cache = self.client.caches.create(
            model="models/gemini-2.0-flash-exp",
            config=types.CreateCachedContentConfig(
                display_name=f"context-cache-{context_hash[:8]}",
                system_instruction=self._jeff_system_prompt,
                contents=[context],  # Pass context as content to cache
                ttl=self._ttl,
            )
        )
        
        # Store the cache name for future use
        self._cache_mapping[context_hash] = cache.name
        return cache.name
    
    def generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False) -> str:
        """Generate text from a prompt and optional context using Gemini's native caching"""
        from google.genai import types
        
        # For non-context queries or evaluation mode without context, use standard approach
        if not context or (evaluation_mode and not context):
            # Use the LangChain model for simplicity in these cases
            if evaluation_mode:
                # Use a clean model without the JEFF system prompt
                clean_model = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash-exp", 
                    google_api_key=os.environ.get("GEMINI_API_KEY"),
                    streaming=False
                )
                return clean_model.invoke(prompt).content
            else:
                # Use the model with JEFF system prompt
                self._model.system_instruction = self._jeff_system_prompt
                return self._model.invoke(prompt).content
        
        # For context-based queries, use Gemini's caching
        cache_name = self._get_or_create_cache(context)
        
        # Prepare the prompt based on evaluation mode
        if evaluation_mode:
            content_prompt = f"""
            Question:
            {prompt}
            
            Answer:
            """
        else:
            content_prompt = f"""
            Answer the question as JEFF, that cool friend who explains subjects better than professors do.
            Remember to be conversational, relatable, and break down complex topics into simple terms.
            Focus on the most important concepts, use memorable examples, and explain things the way you would
            the night before an exam - clear, concise, and actually helpful.
            
            Question:
            {prompt}
            
            Answer:
            """
        
        # Generate content using the cache
        response = self.client.models.generate_content(
            model="models/gemini-2.0-flash-exp",
            contents=content_prompt,
            config=types.GenerateContentConfig(cached_content=cache_name)
        )
        
        # Print token usage statistics for debugging (can be removed in production)
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            print(f"Token usage: {response.usage_metadata}")
        
        return response.text
    
    def stream_generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False) -> Iterator[str]:
        """Stream generate text from a prompt and optional context
        
        Note: For streaming, we use the standard approach since caching benefits are primarily
        for repeated queries rather than streaming display.
        """
        from langchain.prompts import ChatPromptTemplate
        
        # In evaluation mode, don't use system prompt or JEFF persona
        if evaluation_mode:
            if context:
                template = """
                Context:
                {context}
                
                Question:
                {question}
                
                Answer:
                """
            else:
                for chunk in self._streaming_model.stream(prompt):
                    yield chunk.content
                return
        else:
            # Set system prompt for non-evaluation mode
            self._streaming_model.system_instruction = self._jeff_system_prompt
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
            else:
                for chunk in self._streaming_model.stream(prompt):
                    yield chunk.content
                return
        
        prompt_template = ChatPromptTemplate.from_template(template)
        chain = prompt_template | self._streaming_model
        
        for chunk in chain.stream({"context": context, "question": prompt}):
            yield chunk.content

class ClaudeLLM(StreamingLLM):
    """Anthropic Claude model implementation with streaming support and caching"""
    
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
        
        self._model = ChatAnthropic(
            model=model_name,
            streaming=False,
            model_kwargs={"system": jeff_system_prompt}
        )
        self._streaming_model = ChatAnthropic(
            model=model_name,
            streaming=True,
            model_kwargs={"system": jeff_system_prompt}
        )
        self._jeff_system_prompt = jeff_system_prompt
        self._cache = {}  # Cache for storing generated responses
    
    def generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False) -> str:
        """Generate text from a prompt and optional context, with caching"""
        # Create a cache key based on the input parameters
        cache_key = (prompt, context, evaluation_mode)
        
        # Check if we have a cached response
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # If not in cache, generate a new response
        from langchain.prompts import ChatPromptTemplate
        
        # In evaluation mode, don't use system prompt or JEFF persona
        if evaluation_mode:
            # Create a new model instance without the system prompt
            from langchain_anthropic import ChatAnthropic
            evaluation_model = ChatAnthropic(model=self._model.model, streaming=False)
        
            if context:
                # Use cache_control for context to mark it as ephemeral
                messages = [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Context:\n{context}",
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": f"Question:\n{prompt}\n\nAnswer:",
                    },
                ]
                response = evaluation_model.invoke(messages)
                result = response.content
            else:
                result = evaluation_model.invoke(prompt).content
        else:
            if context:
                # Use cache_control for context to mark it as ephemeral
                messages = [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": self._jeff_system_prompt
                            },
                            {
                                "type": "text",
                                "text": f"Context:\n{context}",
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": f"""Answer the question as JEFF, that cool friend who explains subjects better than professors do.
                        Remember to be conversational, relatable, and break down complex topics into simple terms.
                        Focus on the most important concepts, use memorable examples, and explain things the way you would
                        the night before an exam - clear, concise, and actually helpful.
                        
                        Question:
                        {prompt}
                        
                        Answer:"""
                    },
                ]
                response = self._model.invoke(messages)
                result = response.content
            else:
                result = self._model.invoke(prompt).content
        
        # Store the result in cache
        self._cache[cache_key] = result
        return result
    
    def stream_generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False) -> Iterator[str]:
        """Stream generate text from a prompt and optional context
        
        Note: We don't cache streaming responses as they're meant for immediate display.
        """
        from langchain.prompts import ChatPromptTemplate
        
        # In evaluation mode, don't use system prompt or JEFF persona
        if evaluation_mode:
            # Create a new model instance without the system prompt
            from langchain_anthropic import ChatAnthropic
            evaluation_streaming_model = ChatAnthropic(model=self._model.model, streaming=True)
        
            if context:
                # Use cache_control for context to mark it as ephemeral
                messages = [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Context:\n{context}",
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": f"Question:\n{prompt}\n\nAnswer:",
                    },
                ]
                
                for chunk in evaluation_streaming_model.stream(messages):
                    yield chunk.content
            else:
                for chunk in evaluation_streaming_model.stream(prompt):
                    yield chunk.content
        else:
            if context:
                messages = [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": self._jeff_system_prompt
                            },
                            {
                                "type": "text",
                                "text": f"Context:\n{context}",
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": f"""Answer the question as JEFF, that cool friend who explains subjects better than professors do.
                        Remember to be conversational, relatable, and break down complex topics into simple terms.
                        Focus on the most important concepts, use memorable examples, and explain things the way you would
                        the night before an exam - clear, concise, and actually helpful.
                        
                        Question:
                        {prompt}
                        
                        Answer:"""
                    },
                ]
                
                for chunk in self._streaming_model.stream(messages):
                    yield chunk.content
            else:
                for chunk in self._streaming_model.stream(prompt):
                    yield chunk.content

class MistralLLM(StreamingLLM):
    """Mistral model implementation with streaming support"""
    
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
        
        self._model = ChatMistralAI(
            model=model_name,
            streaming=False,
            model_kwargs={"system": jeff_system_prompt}
        )
        self._streaming_model = ChatMistralAI(
            model=model_name,
            streaming=True,
            model_kwargs={"system": jeff_system_prompt}
        )
        self._jeff_system_prompt = jeff_system_prompt
        self._cache = {}  # Initialize cache dictionary

    def generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False) -> str:
        """Generate text from a prompt and optional context"""
        cache_key = (prompt, context, evaluation_mode)
        if cache_key in self._cache:
            return self._cache[cache_key]

        from langchain.prompts import ChatPromptTemplate
        
        # In evaluation mode, don't use system prompt or JEFF persona
        if evaluation_mode:
            if context:
                template = """
                Context:
                {context}
                
                Question:
                {question}
                
                Answer:
                """
            else:
                return self._model.invoke(prompt).content
        else:
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
            else:
                return self._model.invoke(prompt).content
        
        prompt_template = ChatPromptTemplate.from_template(template)
        chain = prompt_template | self._model
        response = chain.invoke({"context": context, "question": prompt})
        self._cache[cache_key] = response.content  # Cache the result
        return response.content
    
    def stream_generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False) -> Iterator[str]:
        """Stream generate text from a prompt and optional context"""
        from langchain.prompts import ChatPromptTemplate
        
        # In evaluation mode, don't use system prompt or JEFF persona
        if evaluation_mode:
            if context:
                template = """
                Context:
                {context}
                
                Question:
                {question}
                
                Answer:
                """
            else:
                for chunk in self._streaming_model.stream(prompt):
                    yield chunk.content
                return
        else:
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
            else:
                for chunk in self._streaming_model.stream(prompt):
                    yield chunk.content
                return
        
        prompt_template = ChatPromptTemplate.from_template(template)
        chain = prompt_template | self._streaming_model
        
        for chunk in chain.stream({"context": context, "question": prompt}):
            yield chunk.content

class LLMFactory:
    """Factory for creating LLM models (Factory Pattern)"""
    
    @staticmethod
    def create_llm(model_type: LLMModelType) -> StreamingLLM:
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