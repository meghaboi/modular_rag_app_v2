import os
import logging
import random
from anthropic import Anthropic
from prompts import get_provider

def is_greeting(query: str) -> tuple[bool, str]:
    """Detect if the query is a greeting using Anthropic's function calling and get the response."""
    try:
        client = Anthropic()
        
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

        greeting_provider = get_provider('greeting')
        prompt_message = greeting_provider.get_prompt('greeting_detection', query=query)

        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": prompt_message
            }],
            tools=[greeting_function]
        )

        tool_calls = [content for content in response.content if content.type == "tool_use"]
        if tool_calls:
            result = tool_calls[0].input
            is_greeting = result.get("is_greeting", False)
            confidence = result.get("confidence", 0.0)
            greeting_response = result.get("response", "")
            
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
        if not os.getenv("ANTHROPIC_API_KEY"):
            logging.error("ANTHROPIC_API_KEY not found. Cannot determine prompt nature.")
            return DEFAULT_NATURE

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

        query_provider = get_provider('query')
        prompt_message = query_provider.get_prompt(
            'nature_classification',
            allowed_natures=', '.join(ALLOWED_NATURES),
            query=query
        )

        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": prompt_message
            }],
            tools=[classify_tool],
            tool_choice={"type": "tool", "name": "classify_prompt_nature"}
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