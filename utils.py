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
from PyPDF2 import PdfReader
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
            temp_path = temp.name
            logging.info(f"Saved uploaded file '{uploaded_file.name}' to temporary path: {temp_path}")
            
            # If it's a PDF file, convert it to text
            if file_suffix.lower() == '.pdf':
                try:
                    # Create a new temporary file for the text content
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as text_temp:
                        # Read the PDF and extract text
                        pdf_reader = PdfReader(temp_path)
                        text_content = ""
                        for page in pdf_reader.pages:
                            text_content += page.extract_text() + "\n"
                        
                        # Write the extracted text to the new temporary file
                        text_temp.write(text_content.encode('utf-8'))
                        text_temp_path = text_temp.name
                    
                    # Close the PDF reader and delete the original PDF temporary file
                    pdf_reader = None  # Release the file handle
                    try:
                        os.unlink(temp_path)
                        logging.info(f"Converted PDF to text and saved to: {text_temp_path}")
                    except Exception as e:
                        logging.warning(f"Could not delete original PDF file: {e}")
                    
                    return text_temp_path
                except Exception as e:
                    logging.error(f"Error converting PDF to text: {e}")
                    return None
            
            return temp_path
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
        is_available = bool(get_openai_api_key())
        api_keys_status[key_name] = "Available" if is_available else "Missing"
        if not is_available: missing_keys_list.append(key_name)

    if cohere_needed:
        key_name = "Cohere API Key"
        is_available = bool(get_cohere_api_key())
        api_keys_status[key_name] = "Available" if is_available else "Missing"
        if not is_available: missing_keys_list.append(key_name)

    if gemini_needed:
        key_name = "Gemini API Key"
        is_available = bool(get_gemini_api_key())
        api_keys_status[key_name] = "Available" if is_available else "Missing"
        if not is_available: missing_keys_list.append(key_name)

    if anthropic_needed:
        key_name = "Anthropic API Key"
        is_available = bool(get_anthropic_api_key())
        api_keys_status[key_name] = "Available" if is_available else "Missing"
        if not is_available: missing_keys_list.append(key_name)

    if mistral_needed:
        key_name = "Mistral API Key"
        is_available = bool(get_mistral_api_key())
        api_keys_status[key_name] = "Available" if is_available else "Missing"
        if not is_available: missing_keys_list.append(key_name)

    if voyage_needed:
        key_name = "Voyage AI API Key"
        is_available = bool(get_voyage_api_key())
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

    if not get_openai_api_key():
        logging.error("OpenAI API Key not found. Cannot generate audio.")
        return None

    try:
        # OpenAI client uses OPENAI_API_KEY env var by default if not passed explicitly
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

def determine_prompt_nature(query: str) -> str:
    """
    Determines the nature of the user's query using Anthropic Claude model
    and function calling.
    """
    ALLOWED_NATURES = [
        "question_answering",
        "summarization",
        "comparison",
        "code_generation",
        "general_discussion"
    ]
    DEFAULT_NATURE = "general_discussion"
    CONFIDENCE_THRESHOLD = 0.7

    try:
        if not get_anthropic_api_key(): # Check if key exists
            logging.error("ANTHROPIC_API_KEY not found. Cannot determine prompt nature.")
            return DEFAULT_NATURE

        # Anthropic client uses ANTHROPIC_API_KEY env var by default if not passed explicitly
        client = Anthropic()

        classify_tool = {
            "name": "classify_prompt_nature",
            "description": "Classify the user's query into one of the predefined categories based on its primary intent.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "nature": {
                        "type": "string",
                        "description": f"The classified nature of the prompt. Must be one of: {', '.join(ALLOWED_NATURES)}",
                        "enum": ALLOWED_NATURES
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence score between 0.0 and 1.0 for the classification."
                    }
                },
                "required": ["nature", "confidence"]
            }
        }

        prompt_message = (
            f"Please classify the following user query into one of these categories: "
            f"{', '.join(ALLOWED_NATURES)}. Focus on the primary intent of the query.\n\n"
            f"User Query: \"{query}\"\n\n"
            "Use the 'classify_prompt_nature' tool to provide your classification."
        )

        response = client.messages.create(
            model="claude-3-sonnet-20240229", # Or a similar suitable model
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": prompt_message
            }],
            tools=[classify_tool],
            tool_choice={"type": "tool", "name": "classify_prompt_nature"} # Force tool usage
        )

        tool_calls = [content for content in response.content if content.type == "tool_use"]

        if tool_calls:
            tool_input = tool_calls[0].input
            nature = tool_input.get("nature")
            confidence = tool_input.get("confidence")

            logging.info(f"Prompt nature classification for query '{query}': Nature='{nature}', Confidence={confidence}")

            if nature in ALLOWED_NATURES and isinstance(confidence, (float, int)) and confidence >= CONFIDENCE_THRESHOLD:
                return nature
            else:
                logging.warning(
                    f"Low confidence or invalid nature for query '{query}'. "
                    f"Nature: {nature}, Confidence: {confidence}. Falling back to default."
                )
                return DEFAULT_NATURE
        else:
            logging.warning(f"No tool call found in response for query '{query}'. Falling back to default.")
            return DEFAULT_NATURE

    except Exception as e:
        logging.error(f"Error determining prompt nature for query '{query}': {e}", exc_info=True)
        return DEFAULT_NATURE

# API Key Getters

def get_openai_api_key() -> str | None:
    """Returns the OpenAI API key from environment variables."""
    return os.getenv("OPENAI_API_KEY")

def get_cohere_api_key() -> str | None:
    """Returns the Cohere API key from environment variables."""
    return os.getenv("COHERE_API_KEY")

def get_gemini_api_key() -> str | None:
    """Returns the Gemini API key from environment variables."""
    return os.getenv("GEMINI_API_KEY")

def get_anthropic_api_key() -> str | None:
    """Returns the Anthropic API key from environment variables."""
    return os.getenv("ANTHROPIC_API_KEY")

def get_mistral_api_key() -> str | None:
    """Returns the Mistral API key from environment variables."""
    return os.getenv("MISTRAL_API_KEY")

def get_voyage_api_key() -> str | None:
    """Returns the Voyage API key from environment variables."""
    return os.getenv("VOYAGE_API_KEY")

def get_langchain_api_key() -> str | None:
    """Returns the Langchain API key (for LangSmith) from environment variables."""
    return os.getenv("LANGCHAIN_API_KEY")

def get_jina_api_key() -> str | None:
    """Returns the Jina API key from environment variables."""
    return os.getenv("JINA_API_KEY")
