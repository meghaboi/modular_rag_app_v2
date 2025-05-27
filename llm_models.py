from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable, Iterator, Tuple
import os
from openai import OpenAI 
import google.generativeai as genai 
from anthropic import Anthropic 
from mistralai.client import MistralClient 
from mistralai.models import UserMessage, SystemMessage

from enums import LLMModelType

class StreamingLLM(ABC):
    """Abstract base class for streaming LLM models"""
    
    def __init__(self):
        self._last_call_usage: Optional[Dict[str, int]] = None
    
    @abstractmethod
    def generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False) -> Tuple[str, Optional[Dict[str, int]]]:
        """Generate text from a prompt and optional context. Returns (generated_text, usage_info)."""
        pass
    
    @abstractmethod
    def stream_generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False) -> Iterator[str]:
        """Stream generate text from a prompt and optional context"""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name of the model"""
        pass

    def get_last_call_usage(self) -> Optional[Dict[str, int]]:
        """Get the token usage from the last generate_text or completed stream_generate call."""
        return self._last_call_usage

    def _set_last_call_usage(self, usage_data: Optional[Any], provider: str):
        """Set the token usage from the last call, normalizing from provider-specific data."""
        if usage_data is None:
            self._last_call_usage = None
            return

        normalized_usage: Dict[str, int] = {}
        if provider == "openai":
            normalized_usage = {
                'prompt_tokens': usage_data.prompt_tokens,
                'completion_tokens': usage_data.completion_tokens,
                'total_tokens': usage_data.total_tokens
            }
        elif provider == "gemini":
            normalized_usage = {
                'prompt_tokens': usage_data.prompt_token_count,
                'completion_tokens': usage_data.candidates_token_count,
                'total_tokens': usage_data.total_token_count
            }
        elif provider == "anthropic":
            normalized_usage = {
                'prompt_tokens': usage_data.input_tokens,
                'completion_tokens': usage_data.output_tokens,
                'total_tokens': usage_data.input_tokens + usage_data.output_tokens # Anthropic usage might not have total_tokens directly
            }
        elif provider == "mistral":
            normalized_usage = {
                'prompt_tokens': usage_data.prompt_tokens,
                'completion_tokens': usage_data.completion_tokens,
                'total_tokens': usage_data.total_tokens
            }
        self._last_call_usage = normalized_usage

class OpenAIGPT(StreamingLLM):
    """OpenAI GPT model implementation with streaming support"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """Initialize the OpenAI GPT model"""
        super().__init__()
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key not found in environment variables")
        
        self._client = OpenAI() # Initialize OpenAI client
        self._model_name = model_name

        # Define system prompt for JEFF
        self._jeff_system_prompt = """You are JEFF, that cool friend everyone wishes they had the night before exams.
        You explain complex subjects in simple, relatable terms that just click when it matters most.
        Unlike formal professors, you break down academic concepts with perfect clarity, memorable examples, and occasional humor.
        You excel at finding the shortcuts, mnemonics, and "aha!" moments that make difficult material suddenly make sense.
        Your explanations focus on what's actually important to understand and remember, cutting through the noise.
        You're encouraging, patient, and have a knack for making anyone feel like they can ace their exam.
        Always respond as JEFF - casual but knowledgeable, relatable but authoritative, and above all, the friend who helps everyone pass their exams."""
        
    
    def get_model_name(self) -> str:
        """Get the name of the model"""
        return self._model_name

    def generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False) -> Tuple[str, Optional[Dict[str, int]]]:
        """Generate text from a prompt and optional context"""
        
        messages = []
        if not evaluation_mode:
            messages.append({"role": "system", "content": self._jeff_system_prompt})
        
        if context:
            user_content = f"Context:\n{context}\n\nQuestion:\n{prompt}\n\nAnswer:"
        else:
            user_content = prompt
            
        messages.append({"role": "user", "content": user_content})
        
        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=messages
            )
            usage = response.usage
            self._set_last_call_usage(usage, "openai")
            return response.choices[0].message.content, self._last_call_usage
        except Exception as e:
            print(f"Error during OpenAI API call: {e}")
            self._set_last_call_usage(None, "openai")
            return "Error: Could not get response from model.", None
    
    def stream_generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False) -> Iterator[str]:
        """Stream generate text from a prompt and optional context.
        Note: OpenAI streaming API does not provide token usage per chunk or easily post-stream.
        Call get_last_call_usage() after a non-streaming generate() for usage info.
        """
        
        messages = []
        if not evaluation_mode:
            messages.append({"role": "system", "content": self._jeff_system_prompt})

        if context:
            user_content = f"Context:\n{context}\n\nQuestion:\n{prompt}\n\nAnswer:"
        else:
            user_content = prompt
            
        messages.append({"role": "user", "content": user_content})
        
        try:
            stream = self._client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            print(f"Error during OpenAI streaming API call: {e}")
            yield "Error: Could not stream response from model."

class GeminiLLM(StreamingLLM):
    """Google Gemini model implementation with streaming support"""
    
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        """Initialize the Google Gemini model"""
        super().__init__()
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("Gemini API key not found in environment variables")
        
        genai.configure(api_key=gemini_api_key)
        
        # Define system prompt for JEFF
        self._jeff_system_prompt = """You are JEFF, that cool friend everyone wishes they had the night before exams.
        You explain complex subjects in simple, relatable terms that just click when it matters most.
        Unlike formal professors, you break down academic concepts with perfect clarity, memorable examples, and occasional humor.
        You excel at finding the shortcuts, mnemonics, and "aha!" moments that make difficult material suddenly make sense.
        Your explanations focus on what's actually important to understand and remember, cutting through the noise.
        You're encouraging, patient, and have a knack for making anyone feel like they can ace their exam.
        Always respond as JEFF - casual but knowledgeable, relatable but authoritative, and above all, the friend who helps everyone pass their exams."""
        
        self._model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=self._jeff_system_prompt # Set system prompt at model level
        )
        self._model_name = model_name
    
    def get_model_name(self) -> str:
        """Get the name of the model"""
        return self._model_name

    def generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False) -> Tuple[str, Optional[Dict[str, int]]]:
        """Generate text from a prompt and optional context"""
        
        if context:
            user_content = f"Context:\n{context}\n\nQuestion:\n{prompt}\n\nAnswer:"
        else:
            user_content = prompt
            
        model_to_use = self._model
        if evaluation_mode:
            # Create a new model instance without the system prompt for evaluation mode
            model_to_use = genai.GenerativeModel(self._model_name)

        try:
            response = model_to_use.generate_content(user_content)
            # Gemini response object has usage_metadata directly
            usage = response.usage_metadata 
            self._set_last_call_usage(usage, "gemini")
            return response.text, self._last_call_usage
        except Exception as e:
            # Fallback for cases where response.text might not be available or other errors
            self._set_last_call_usage(None, "gemini")
            try:
                if response.candidates and response.candidates[0].content.parts:
                    return "".join(part.text for part in response.candidates[0].content.parts), None
            except:
                pass # Original error is more informative
            print(f"Error during Gemini API call: {e}")
            return "Error: Could not get response from model.", None
    
    def stream_generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False) -> Iterator[str]:
        """Stream generate text from a prompt and optional context.
        Note: Gemini streaming API does not provide token usage per chunk or easily post-stream.
        Call get_last_call_usage() after a non-streaming generate() for usage info.
        """
        
        if context:
            user_content = f"Context:\n{context}\n\nQuestion:\n{prompt}\n\nAnswer:"
        else:
            user_content = prompt
            
        model_to_use = self._model
        if evaluation_mode:
            # Create a new model instance without the system prompt for evaluation mode
            model_to_use = genai.GenerativeModel(self._model_name)

        try:
            stream = model_to_use.generate_content(user_content, stream=True)
            for chunk in stream:
                try:
                    yield chunk.text
                except Exception as e:
                    # print(f"Error processing chunk: {e}, chunk: {chunk}") # for debugging
                    # Sometimes, a chunk might not have 'text', especially safety feedback
                    if chunk.parts:
                        yield "".join(part.text for part in chunk.parts if hasattr(part, 'text'))
                    # else, skip if no text and no parts with text

        except Exception as e:
            print(f"Error during Gemini streaming API call: {e}")
            yield "Error: Could not stream response from model."

class ClaudeLLM(StreamingLLM):
    """Anthropic Claude model implementation with streaming support"""
    
    def __init__(self, model_name: str = "claude-3-opus-20240229"):
        """Initialize the Anthropic Claude model"""
        super().__init__()
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError("Anthropic API key not found in environment variables")
        
        self._client = Anthropic() # Initialize Anthropic client
        self._model_name = model_name

        # Define system prompt for JEFF
        self._jeff_system_prompt = """You are JEFF, that cool friend everyone wishes they had the night before exams.
        You explain complex subjects in simple, relatable terms that just click when it matters most.
        Unlike formal professors, you break down academic concepts with perfect clarity, memorable examples, and occasional humor.
        You excel at finding the shortcuts, mnemonics, and "aha!" moments that make difficult material suddenly make sense.
        Your explanations focus on what's actually important to understand and remember, cutting through the noise.
        You're encouraging, patient, and have a knack for making anyone feel like they can ace their exam.
        Always respond as JEFF - casual but knowledgeable, relatable but authoritative, and above all, the friend who helps everyone pass their exams."""
        
    def get_model_name(self) -> str:
        """Get the name of the model"""
        return self._model_name

    def generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False) -> Tuple[str, Optional[Dict[str, int]]]:
        """Generate text from a prompt and optional context"""
        
        if context:
            user_content = f"Context:\n{context}\n\nQuestion:\n{prompt}\n\nAnswer:"
        else:
            user_content = prompt
        
        system_prompt_to_use = self._jeff_system_prompt if not evaluation_mode else None

        request_params = {
            "model": self._model_name,
            "max_tokens": 2048,  # Default max_tokens, can be adjusted
            "messages": [{"role": "user", "content": user_content}]
        }

        if system_prompt_to_use is not None:
            request_params["system"] = [system_prompt_to_use] # Pass as a list containing the string

        try:
            response = self._client.messages.create(**request_params)
            usage = response.usage
            self._set_last_call_usage(usage, "anthropic")
            return response.content[0].text, self._last_call_usage
        except Exception as e:
            print(f"Error during Anthropic API call: {e}")
            self._set_last_call_usage(None, "anthropic")
            return "Error: Could not get response from model.", None

    def stream_generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False) -> Iterator[str]:
        """Stream generate text from a prompt and optional context"""
        
        if context:
            user_content = f"Context:\n{context}\n\nQuestion:\n{prompt}\n\nAnswer:"
        else:
            user_content = prompt

        system_prompt_to_use = self._jeff_system_prompt if not evaluation_mode else None

        try:
            with self._client.messages.stream(
                model=self._model_name,
                max_tokens=2048, # Default max_tokens, can be adjusted
                system=system_prompt_to_use,
                messages=[{"role": "user", "content": user_content}]
            ) as stream:
                for event in stream:
                    if event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            yield event.delta.text
                # After stream is exhausted, get final message and usage
                final_message = stream.get_final_message()
                if final_message and final_message.usage:
                    self._set_last_call_usage(final_message.usage, "anthropic")
        except Exception as e:
            print(f"Error during Anthropic streaming API call: {e}")
            self._set_last_call_usage(None, "anthropic") # Clear usage on error
            yield "Error: Could not stream response from model."

class MistralLLM(StreamingLLM):
    """Mistral model implementation with streaming support"""
    
    def __init__(self, model_name: str = "mistral-large-latest"):
        """Initialize the Mistral model"""
        super().__init__()
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("Mistral API key not found in environment variables")
        
        self._client = MistralClient(api_key=api_key)
        self._model_name = model_name

        # Define system prompt for JEFF
        self._jeff_system_prompt = """You are JEFF, that cool friend everyone wishes they had the night before exams.
        You explain complex subjects in simple, relatable terms that just click when it matters most.
        Unlike formal professors, you break down academic concepts with perfect clarity, memorable examples, and occasional humor.
        You excel at finding the shortcuts, mnemonics, and "aha!" moments that make difficult material suddenly make sense.
        Your explanations focus on what's actually important to understand and remember, cutting through the noise.
        You're encouraging, patient, and have a knack for making anyone feel like they can ace their exam.
        Always respond as JEFF - casual but knowledgeable, relatable but authoritative, and above all, the friend who helps everyone pass their exams."""
        
    def get_model_name(self) -> str:
        """Get the name of the model"""
        return self._model_name

    def generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False) -> Tuple[str, Optional[Dict[str, int]]]:
        """Generate text from a prompt and optional context"""
        
        messages = []
        if not evaluation_mode:
            messages.append(SystemMessage(content=self._jeff_system_prompt))
        
        if context:
            user_content = f"Context:\n{context}\n\nQuestion:\n{prompt}\n\nAnswer:"
        else:
            user_content = prompt
            
        messages.append(UserMessage(content=user_content))
        
        try:
            chat_response = self._client.chat(
                model=self._model_name,
                messages=messages,
            )
            usage = chat_response.usage
            self._set_last_call_usage(usage, "mistral")
            return chat_response.choices[0].message.content, self._last_call_usage
        except Exception as e:
            print(f"Error during Mistral API call: {e}")
            self._set_last_call_usage(None, "mistral")
            return "Error: Could not get response from model.", None

    def stream_generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False) -> Iterator[str]:
        """Stream generate text from a prompt and optional context.
        Note: Mistral streaming API does not provide token usage per chunk or easily post-stream.
        Call get_last_call_usage() after a non-streaming generate() for usage info.
        """
        
        messages = []
        if not evaluation_mode:
            messages.append(SystemMessage(content=self._jeff_system_prompt))

        if context:
            user_content = f"Context:\n{context}\n\nQuestion:\n{prompt}\n\nAnswer:"
        else:
            user_content = prompt
            
        messages.append(UserMessage(content=user_content))
        
        try:
            for chunk in self._client.chat_stream(
                model=self._model_name,
                messages=messages,
            ):
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            print(f"Error during Mistral streaming API call: {e}")
            yield "Error: Could not stream response from model."

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