from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable, Iterator, Tuple
import os
import logging
from openai import OpenAI 
import google.generativeai as genai 
from anthropic import Anthropic 
from mistralai import Mistral 
from mistralai.models import UserMessage, SystemMessage

from utils.enums import LLMModelType
from prompts import get_provider

# Added logging for better debugging (safe improvement)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StreamingLLM(ABC):
    """Abstract base class for streaming LLM models"""
    
    def __init__(self):
        self._last_call_usage: Optional[Dict[str, int]] = None
        self._prompt_provider = get_provider('llm')
        self._system_prompt = self._prompt_provider.get_prompt('system')
    
    @abstractmethod
    def generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False, system_prompt_override: Optional[str] = None) -> Tuple[str, Optional[Dict[str, int]]]:
        """Generate text from a prompt and optional context. Returns (generated_text, usage_info)."""
        pass
    
    @abstractmethod
    def stream_generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False, system_prompt_override: Optional[str] = None) -> Iterator[str]:
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
    
    def __init__(self, model_name: str = "gpt-4-turbo-preview"):
        """Initialize the OpenAI GPT model"""
        super().__init__()
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key not found in environment variables")
        
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) # Initialize OpenAI client
        self._model_name = model_name

    def get_model_name(self) -> str:
        """Get the name of the model"""
        return self._model_name

    def generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False, system_prompt_override: Optional[str] = None) -> Tuple[str, Optional[Dict[str, int]]]:
        """Generate text from a prompt and optional context"""
        
        messages = []
        if system_prompt_override is not None:
            if system_prompt_override != "":
                messages.append({"role": "system", "content": system_prompt_override})
        elif not evaluation_mode:
            messages.append({"role": "system", "content": self._system_prompt})
        
        if context:
            user_content = self._prompt_provider.get_prompt('query', context=context, question=prompt)
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
            logger.error(f"Error during OpenAI API call: {e}")  # Changed from print to logger
            self._set_last_call_usage(None, "openai")
            return "Error: Could not get response from model.", None
    
    def stream_generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False, system_prompt_override: Optional[str] = None) -> Iterator[str]:
        """Stream generate text from a prompt and optional context.
        Note: OpenAI streaming API does not provide token usage per chunk or easily post-stream.
        Call get_last_call_usage() after a non-streaming generate() for usage info.
        """
        
        messages = []
        if system_prompt_override is not None:
            if system_prompt_override != "":
                messages.append({"role": "system", "content": system_prompt_override})
        elif not evaluation_mode:
            messages.append({"role": "system", "content": self._system_prompt})

        if context:
            user_content = self._prompt_provider.get_prompt('query', context=context, question=prompt)
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
            logger.error(f"Error during OpenAI streaming API call: {e}")  # Changed from print to logger
            yield "Error: Could not stream response from model."

class GeminiLLM(StreamingLLM):
    """Google Gemini model implementation with streaming support"""

    def __init__(self, model_name: str = "gemini-2.5-pro-preview-06-05"):
        """Initialize the Google Gemini model"""
        super().__init__()
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("Gemini API key not found in environment variables")

        genai.configure(api_key=gemini_api_key)

        self._model_name = model_name
        # Initialize model without system_instruction initially
        self._model = genai.GenerativeModel(model_name=self._model_name)

        # Check if system_instruction is supported
        self._supports_system_instruction = self._check_system_instruction_support()

        # If supported, reinitialize with system instruction
        if self._supports_system_instruction and hasattr(self, '_system_prompt') and self._system_prompt:
            try:
                self._model = genai.GenerativeModel(
                    model_name=self._model_name,
                    system_instruction=self._system_prompt
                )
            except Exception:
                # Fallback if there's still an issue
                self._model = genai.GenerativeModel(model_name=self._model_name)
                self._supports_system_instruction = False

    def _check_system_instruction_support(self) -> bool:
        """Check if the current version supports system_instruction parameter"""
        try:
            # Try creating a model with system_instruction to test support
            test_model = genai.GenerativeModel(
                model_name=self._model_name,
                system_instruction="test"
            )
            return True
        except TypeError:
            return False
        except Exception:
            # Other exceptions might indicate API issues, assume supported
            return True

    def get_model_name(self) -> str:
        """Get the name of the model"""
        return self._model_name

    def _create_model_with_system_instruction(self, system_instruction: Optional[str]) -> Any:
        """Create a model with the specified system instruction"""
        if self._supports_system_instruction and system_instruction is not None:
            try:
                return genai.GenerativeModel(
                    model_name=self._model_name,
                    system_instruction=system_instruction
                )
            except Exception:
                # Fallback to model without system instruction
                return genai.GenerativeModel(model_name=self._model_name)
        else:
            return genai.GenerativeModel(model_name=self._model_name)

    def _prepare_content_with_system_prompt(self, user_content: str, system_instruction: Optional[str]) -> str:
        """Prepare content by prepending system instruction if not natively supported"""
        if not self._supports_system_instruction and system_instruction:
            return f"{system_instruction}\n\n{user_content}"
        return user_content

    def generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False,
                 system_prompt_override: Optional[str] = None) -> Tuple[str, Optional[Dict[str, int]]]:
        """Generate text from a prompt and optional context"""

        if context:
            user_content = self._prompt_provider.get_prompt('query', context=context, question=prompt)
        else:
            user_content = prompt

        # Determine the system instruction to use
        current_system_instruction = self._system_prompt
        if system_prompt_override is not None:
            if system_prompt_override == "":
                current_system_instruction = None
            else:
                current_system_instruction = system_prompt_override
        elif evaluation_mode:
            current_system_instruction = None

        # Create model with appropriate system instruction
        model_to_use = self._create_model_with_system_instruction(current_system_instruction)

        # Prepare content (add system instruction to content if not natively supported)
        final_content = self._prepare_content_with_system_prompt(user_content, current_system_instruction)

        try:
            response = model_to_use.generate_content(final_content)

            # Handle usage metadata if available
            usage = None
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = response.usage_metadata
                self._set_last_call_usage(usage, "gemini")
            else:
                self._set_last_call_usage(None, "gemini")

            return response.text, self._last_call_usage

        except Exception as e:
            # Fallback for cases where response.text might not be available or other errors
            self._set_last_call_usage(None, "gemini")
            try:
                if hasattr(response, 'candidates') and response.candidates and response.candidates[0].content.parts:
                    return "".join(part.text for part in response.candidates[0].content.parts), None
            except:
                pass
            logger.error(f"Error during Gemini API call: {e}")  # Changed from print to logger
            return "Error: Could not get response from model.", None

    def stream_generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False,
                        system_prompt_override: Optional[str] = None) -> Iterator[str]:
        """Stream generates text from a prompt and optional context.
        Note: Gemini streaming API does not provide token usage per chunk or easily post-stream.
        Call get_last_call_usage() after a non-streaming generate() for usage info.
        """

        if context:
            user_content = self._prompt_provider.get_prompt('query', context=context, question=prompt)
        else:
            user_content = prompt

        # Determine the system instruction to use
        current_system_instruction = self._system_prompt
        if system_prompt_override is not None:
            if system_prompt_override == "":
                current_system_instruction = None
            else:
                current_system_instruction = system_prompt_override
        elif evaluation_mode:
            current_system_instruction = None

        # Create model with appropriate system instruction
        model_to_use = self._create_model_with_system_instruction(current_system_instruction)

        # Prepare content (add system instruction to content if not natively supported)
        final_content = self._prepare_content_with_system_prompt(user_content, current_system_instruction)

        try:
            stream = model_to_use.generate_content(final_content, stream=True)
            for chunk in stream:
                try:
                    if hasattr(chunk, 'text') and chunk.text:
                        yield chunk.text
                except Exception as e:
                    # Sometimes, a chunk might not have 'text', especially safety feedback
                    try:
                        if hasattr(chunk, 'parts') and chunk.parts:
                            text_parts = []
                            for part in chunk.parts:
                                if hasattr(part, 'text') and part.text:
                                    text_parts.append(part.text)
                            if text_parts:
                                yield "".join(text_parts)
                    except Exception:
                        # Skip chunks that can't be processed
                        continue

        except Exception as e:
            logger.error(f"Error during Gemini streaming API call: {e}")  # Changed from print to logger
            yield "Error: Could not stream response from model."

class ClaudeLLM(StreamingLLM):
    """Anthropic Claude model implementation with streaming support"""
    
    def __init__(self, model_name: str = "claude-sonnet-4-20250514"):
        """Initialize the Anthropic Claude model"""
        super().__init__()
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError("Anthropic API key not found in environment variables")
        
        self._client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")) # Initialize Anthropic client
        self._model_name = model_name

    def get_model_name(self) -> str:
        """Get the name of the model"""
        return self._model_name

    def generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False, system_prompt_override: Optional[str] = None) -> Tuple[str, Optional[Dict[str, int]]]:
        """Generate text from a prompt and optional context"""
        
        if context:
            user_content = self._prompt_provider.get_prompt('query', context=context, question=prompt)
        else:
            user_content = prompt
        
        system_param_value: Optional[str] = None
        if system_prompt_override is not None:
            if system_prompt_override != "":
                system_param_value = system_prompt_override
        elif not evaluation_mode:
            system_param_value = self._system_prompt
        # If evaluation_mode is True and system_prompt_override is None, system_param_value remains None

        request_params = {
            "model": self._model_name,
            "max_tokens": 2048,  # Default max_tokens, can be adjusted
            "messages": [{"role": "user", "content": user_content}]
        }

        if system_param_value is not None:
            request_params["system"] = system_param_value

        try:
            response = self._client.messages.create(**request_params)
            usage = response.usage
            self._set_last_call_usage(usage, "anthropic")
            return response.content[0].text, self._last_call_usage
        except Exception as e:
            logger.error(f"Error during Anthropic API call: {e}")  # Changed from print to logger
            self._set_last_call_usage(None, "anthropic")
            return "Error: Could not get response from model.", None

    def stream_generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False, system_prompt_override: Optional[str] = None) -> Iterator[str]:
        """Stream generates text from a prompt and optional context"""
        
        if context:
            user_content = self._prompt_provider.get_prompt('query', context=context, question=prompt)
        else:
            user_content = prompt

        system_param_value: Optional[str] = None
        if system_prompt_override is not None:
            if system_prompt_override != "":
                system_param_value = system_prompt_override
        elif not evaluation_mode:
            system_param_value = self._system_prompt
        # If evaluation_mode is True and system_prompt_override is None, system_param_value remains None

        request_params = {
            "model": self._model_name,
            "max_tokens": 2048, # Default max_tokens can be adjusted
            "messages": [{"role": "user", "content": user_content}]
        }
        if system_param_value is not None:
            request_params["system"] = system_param_value

        try:
            with self._client.messages.stream(**request_params) as stream:
                for event in stream:
                    if event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            yield event.delta.text
                # After stream is exhausted, get final message and usage
                final_message = stream.get_final_message()
                if final_message and final_message.usage:
                    self._set_last_call_usage(final_message.usage, "anthropic")
        except Exception as e:
            logger.error(f"Error during Anthropic streaming API call: {e}")  # Changed from print to logger
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
        
        self._client = Mistral(api_key=api_key)
        self._model_name = model_name

    def get_model_name(self) -> str:
        """Get the name of the model"""
        return self._model_name

    def generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False, system_prompt_override: Optional[str] = None) -> Tuple[str, Optional[Dict[str, int]]]:
        """Generate text from a prompt and optional context"""
        
        messages = []
        if system_prompt_override is not None:
            if system_prompt_override != "":
                messages.append(SystemMessage(content=system_prompt_override))
        elif not evaluation_mode:
            messages.append(SystemMessage(content=self._system_prompt))
        
        if context:
            user_content = self._prompt_provider.get_prompt('query', context=context, question=prompt)
        else:
            user_content = prompt
            
        messages.append(UserMessage(content=user_content))
        
        try:
            chat_response = self._client.chat.complete(
                model=self._model_name,
                messages=messages,
            )
            usage = chat_response.usage
            self._set_last_call_usage(usage, "mistral")
            return chat_response.choices[0].message.content, self._last_call_usage
        except Exception as e:
            logger.error(f"Error during Mistral API call: {e}")  # Changed from print to logger
            self._set_last_call_usage(None, "mistral")
            return "Error: Could not get response from model.", None

    def stream_generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False, system_prompt_override: Optional[str] = None) -> Iterator[str]:
        """Stream generate text from a prompt and optional context.
        Note: Mistral streaming API does not provide token usage per chunk or easily post-stream.
        Call get_last_call_usage() after a non-streaming generate() for usage info.
        """
        
        messages = []
        if system_prompt_override is not None:
            if system_prompt_override != "":
                messages.append(SystemMessage(content=system_prompt_override))
        elif not evaluation_mode:
            messages.append(SystemMessage(content=self._system_prompt))

        if context:
            user_content = self._prompt_provider.get_prompt('query', context=context, question=prompt)
        else:
            user_content = prompt
            
        messages.append(UserMessage(content=user_content))
        
        try:
            for chunk in self._client.chat.stream(
                model=self._model_name,
                messages=messages,
            ):
                if chunk.data.choices[0].delta.content is not None:
                    yield chunk.data.choices[0].delta.content
        except Exception as e:
            logger.error(f"Error during Mistral streaming API call: {e}")  # Changed from print to logger
            yield "Error: Could not stream response from model."

class CerebrasLLM(StreamingLLM):
    """Cerebras model implementation (OpenAI-compatible) with streaming support"""

    def __init__(self, model_name: str = "llama-3.2-3b"):
        """Initialize the Cerebras model via OpenAI-compatible API"""
        super().__init__()
        api_key = os.environ.get("CEREBRAS_API_KEY")
        if not api_key:
            raise ValueError("Cerebras API key not found in environment variables")

        # Use OpenAI client with Cerebras base_url
        self._client = OpenAI(api_key=api_key, base_url=os.environ.get("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1"))
        self._model_name = model_name

    def get_model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False, system_prompt_override: Optional[str] = None) -> Tuple[str, Optional[Dict[str, int]]]:
        """Generate text from a prompt and optional context"""

        messages = []
        if system_prompt_override is not None:
            if system_prompt_override != "":
                messages.append({"role": "system", "content": system_prompt_override})
        elif not evaluation_mode:
            messages.append({"role": "system", "content": self._system_prompt})

        if context:
            user_content = self._prompt_provider.get_prompt('query', context=context, question=prompt)
        else:
            user_content = prompt

        messages.append({"role": "user", "content": user_content})

        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=messages
            )
            usage = response.usage
            # Cerebras is OpenAI-compatible, reuse OpenAI usage normalization
            self._set_last_call_usage(usage, "openai")
            return response.choices[0].message.content, self._last_call_usage
        except Exception as e:
            logger.error(f"Error during Cerebras API call: {e}")
            self._set_last_call_usage(None, "openai")
            return "Error: Could not get response from model.", None

    def stream_generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False, system_prompt_override: Optional[str] = None) -> Iterator[str]:
        """Stream generate text from a prompt and optional context"""

        messages = []
        if system_prompt_override is not None:
            if system_prompt_override != "":
                messages.append({"role": "system", "content": system_prompt_override})
        elif not evaluation_mode:
            messages.append({"role": "system", "content": self._system_prompt})

        if context:
            user_content = self._prompt_provider.get_prompt('query', context=context, question=prompt)
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
                if chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Error during Cerebras streaming API call: {e}")
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
        elif model_type == LLMModelType.CLAUDE_3_5_HAIKU:
            return ClaudeLLM(model_name="claude-3-5-haiku-20241022")
        elif model_type == LLMModelType.CLAUDE_4_OPUS:
            return ClaudeLLM(model_name="claude-opus-4-20250514")
        elif model_type == LLMModelType.CLAUDE_4_SONNET:
            return ClaudeLLM(model_name="claude-sonnet-4-20250514")
        elif model_type == LLMModelType.MISTRAL_LARGE:
            return MistralLLM(model_name="mistral-large-latest")
        elif model_type == LLMModelType.MISTRAL_MEDIUM:
            return MistralLLM(model_name="mistral-medium-latest")
        elif model_type == LLMModelType.MISTRAL_SMALL:
            return MistralLLM(model_name="mistral-small-latest")
        elif model_type == LLMModelType.CEREBRAS_LLAMA3_3B:
            return CerebrasLLM(model_name="llama-3.2-3b")
        else:
            raise ValueError(f"Unsupported LLM model: {model_type}")