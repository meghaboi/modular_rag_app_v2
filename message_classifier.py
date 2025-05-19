from typing import Dict, Any, Literal
import anthropic
import os
import logging
import json

class MessageClassifier:
    """Classifies user messages using Claude's function calling capabilities"""
    
    def __init__(self):
        """Initialize the message classifier"""
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError("Anthropic API key not found in environment variables")
            
        self._model = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        
        # Define the function schema for message classification
        self._function_schema = {
            "name": "classify_message",
            "description": "Classify the user's message into one of two categories",
            "input_schema": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["greeting", "other"],
                        "description": "The category of the message"
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Brief explanation of why this category was chosen"
                    },
                    "response": {
                        "type": "string",
                        "description": "Appropriate response message for greetings. Leave empty for other queries."
                    }
                },
                "required": ["category", "explanation", "response"]
            }
        }
        
    def classify(self, message: str) -> Dict[str, Any]:
        """Classify a user message using Claude's function calling
        
        Args:
            message: The user's message to classify
            
        Returns:
            Dict containing classification results with keys:
            - category: The message category (greeting/other)
            - explanation: Why this category was chosen
            - response: Response message for greetings (empty for other queries)
        """
        # Create the system prompt for classification
        system_prompt = """You are a helpful AI assistant that classifies user messages into two categories:
        1. greeting: General greetings, pleasantries, or small talk
        2. other: Any other type of message that should be handled by the RAG pipeline
        
        For greetings, provide an appropriate response message.
        For all other messages, leave the response empty as they will be handled by the RAG pipeline.
        
        Always maintain a friendly and helpful tone."""
        
        try:
            # Call Claude with function calling
            response = self._model.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1024,
                temperature=0,
                system=system_prompt,
                messages=[{"role": "user", "content": message}],
                tools=[self._function_schema]
            )
            
            # Log the first content block for debugging
            if response.content:
                logging.info(f"First response block: {response.content[0]}")
            
            # Find the tool use block in the response
            tool_block = next(
                (block for block in response.content if block.type == "tool_use"),
                None
            )
            
            if tool_block:
                logging.info(f"Tool block found: {tool_block}")
                return tool_block.input
            else:
                logging.warning("No tool use block found in response")
                return {
                    "category": "other",
                    "explanation": "Failed to classify message, treating as other by default",
                    "response": ""
                }
                
        except Exception as e:
            logging.error(f"Error in message classification: {e}")
            # Fallback in case of any errors
            return {
                "category": "other",
                "explanation": f"Error during classification: {str(e)}. Treating as other by default",
                "response": ""
            } 