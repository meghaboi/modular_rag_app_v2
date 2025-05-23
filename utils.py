import os
import re
import logging
import tempfile
import base64
import httpx
from typing import Optional, List, Dict
from openai import OpenAI
from anthropic import Anthropic
import random
from enums import (
    EmbeddingModelType,
    RerankerModelType,
    LLMModelType,
    VectorStoreType,
    ChunkingStrategyType
)

def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary location and return the path"""
    try:
        file_suffix = os.path.splitext(uploaded_file.name)[1] if '.' in uploaded_file.name else '.txt'
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp:
            temp.write(uploaded_file.getvalue())
            logging.info(f"Saved uploaded file '{uploaded_file.name}' to temporary path: {temp.name}")
            return temp.name
    except Exception as e:
        logging.error(f"Error saving uploaded file: {e}")
        return None

def get_csv_download_link(df, filename="permutation_results.csv"):
    """Generate a download link for a pandas dataframe as CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Results as CSV</a>'
    return href

def check_api_keys(embedding_model_enum, vector_store_enum, reranker_enum, llm_enum):
    """Check if required API keys are available in environment"""
    api_keys_status = {}
    missing_keys_list = []

    # Determine required keys based on selections
    openai_needed = (embedding_model_enum == EmbeddingModelType.OPENAI or
                     llm_enum in [LLMModelType.OPENAI_GPT35, LLMModelType.OPENAI_GPT4] or
                     True) # OpenAI TTS always needs it
    cohere_needed = (embedding_model_enum == EmbeddingModelType.COHERE or
                     reranker_enum in [RerankerModelType.COHERE_V2, RerankerModelType.COHERE_V3, RerankerModelType.COHERE_MULTILINGUAL])
    gemini_needed = (embedding_model_enum == EmbeddingModelType.GEMINI or
                     llm_enum == LLMModelType.GEMINI)
    anthropic_needed = (llm_enum in [LLMModelType.CLAUDE_3_OPUS, LLMModelType.CLAUDE_37_SONNET])
    mistral_needed = (embedding_model_enum == EmbeddingModelType.MISTRAL or
                      llm_enum in [LLMModelType.MISTRAL_LARGE, LLMModelType.MISTRAL_MEDIUM, LLMModelType.MISTRAL_SMALL])
    voyage_needed = (embedding_model_enum == EmbeddingModelType.VOYAGE or
                     reranker_enum in [RerankerModelType.VOYAGE, RerankerModelType.VOYAGE_2])

    # Check and record status
    if openai_needed:
        key_name = "OpenAI API Key"
        is_available = bool(os.getenv("OPENAI_API_KEY"))
        api_keys_status[key_name] = "Available" if is_available else "Missing"
        if not is_available: missing_keys_list.append(key_name)

    if cohere_needed:
        key_name = "Cohere API Key"
        is_available = bool(os.getenv("COHERE_API_KEY"))
        api_keys_status[key_name] = "Available" if is_available else "Missing"
        if not is_available: missing_keys_list.append(key_name)

    if gemini_needed:
        key_name = "Gemini API Key"
        is_available = bool(os.getenv("GEMINI_API_KEY"))
        api_keys_status[key_name] = "Available" if is_available else "Missing"
        if not is_available: missing_keys_list.append(key_name)

    if anthropic_needed:
        key_name = "Anthropic API Key"
        is_available = bool(os.getenv("ANTHROPIC_API_KEY"))
        api_keys_status[key_name] = "Available" if is_available else "Missing"
        if not is_available: missing_keys_list.append(key_name)

    if mistral_needed:
        key_name = "Mistral API Key"
        is_available = bool(os.getenv("MISTRAL_API_KEY"))
        api_keys_status[key_name] = "Available" if is_available else "Missing"
        if not is_available: missing_keys_list.append(key_name)

    if voyage_needed:
        key_name = "Voyage AI API Key"
        is_available = bool(os.getenv("VOYAGE_API_KEY"))
        api_keys_status[key_name] = "Available" if is_available else "Missing"
        if not is_available: missing_keys_list.append(key_name)

    return missing_keys_list

def text_to_speech(text: str) -> Optional[bytes]:
    """Generates speech from text using OpenAI TTS and returns audio bytes."""
    if not text or not isinstance(text, str):
        logging.warning("TTS skipped: Input text is empty or not a string.")
        return None

    cleaned_text = re.sub(r'[#*]', '', text) 
    cleaned_text = re.sub(r'http[s]?://\S+', '', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    if not cleaned_text:
        logging.warning("TTS skipped: Text is empty after cleaning.")
        return None

    if not os.getenv("OPENAI_API_KEY"):
        logging.error("OpenAI API Key not found. Cannot generate audio.")
        return None

    try:
        client = OpenAI(timeout=httpx.Timeout(45.0, connect=10.0)) 
        selected_voice = "fable"
        selected_model = "tts-1"

        logging.info(f"Requesting OpenAI TTS: voice='{selected_voice}', model='{selected_model}', text length (cleaned): {len(cleaned_text)}")

        response = client.audio.speech.create(
            model=selected_model,
            voice=selected_voice,
            input=cleaned_text,
            response_format="mp3"
        )

        audio_bytes = response.read()
        logging.info(f"OpenAI TTS audio generated successfully ({len(audio_bytes)} bytes).")
        return audio_bytes

    except ImportError:
        logging.error("OpenAI library not installed. Cannot generate audio.")
        return None
    except Exception as e:
        logging.error(f"Error generating OpenAI TTS audio: {e}", exc_info=True)
        return None

def is_greeting(query: str) -> tuple[bool, str]:
    """Detect if the query is a greeting using Anthropic's function calling and get the response."""
    try:
        client = Anthropic()
        
        # Define the function for greeting detection
        greeting_function = {
            "name": "detect_greeting",
            "description": "Detect if the input text is a greeting or small talk and provide a friendly response",
            "input_schema": {
                "type": "object",
                "properties": {
                    "is_greeting": {
                        "type": "boolean",
                        "description": "Whether the input is a greeting or small talk"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence score between 0 and 1"
                    },
                    "response": {
                        "type": "string",
                        "description": "A friendly response to the greeting"
                    }
                },
                "required": ["is_greeting", "confidence", "response"]
            }
        }

        # Call Anthropic with function calling
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"Analyze if this is a greeting or small talk and provide a friendly response: {query}"
            }],
            tools=[greeting_function]
        )

        # Extract the function call result
        tool_calls = [content for content in response.content if content.type == "tool_use"]
        if tool_calls:
            result = tool_calls[0].input
            is_greeting = result.get("is_greeting", False)
            confidence = result.get("confidence", 0.0)
            greeting_response = result.get("response", "")
            
            # Only consider it a greeting if confidence is high enough
            return (is_greeting and confidence > 0.7, greeting_response)
            
        return (False, "")
    except Exception as e:
        logging.error(f"Error in greeting detection: {e}")
        return (False, "")

def get_greeting_response() -> str:
    """Generate a friendly greeting response."""
    greetings = [
        "Hey there! How can I help you with your studies today?",
        "Hi! Ready to tackle some learning together?",
        "Hello! What would you like to learn about?",
        "Hey! I'm here to help you understand your textbook better. What's on your mind?",
        "Hi there! Let's make learning fun. What would you like to know?"
    ]
    return random.choice(greetings) 