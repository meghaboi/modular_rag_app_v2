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
            "description": "Classify the user's message into one of three categories",
            "input_schema": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["greeting", "relevant_query", "irrelevant_query"],
                        "description": "The category of the message"
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Brief explanation of why this category was chosen"
                    },
                    "response": {
                        "type": "string",
                        "description": "Appropriate response message for greetings or irrelevant queries. Leave empty for relevant queries."
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
            - category: The message category (greeting/relevant_query/irrelevant_query)
            - explanation: Why this category was chosen
            - response: Response message for greetings/irrelevant queries (empty for relevant)
        """
        # Create the system prompt for classification
        system_prompt = """You are a helpful AI assistant that classifies user messages into three categories:
        1. greeting: General greetings, pleasantries, or small talk
        2. relevant_query: Questions or requests related to studying, learning, or understanding academic content
        3. irrelevant_query: Questions or requests not related to academic content or learning
        
        For greetings and irrelevant queries, provide an appropriate response message.
        For relevant queries, leave the response empty as they will be handled by the RAG pipeline.
        
        Always maintain a friendly and helpful tone, even when marking something as irrelevant."""
        
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
                    "category": "relevant_query",
                    "explanation": "Failed to classify message, treating as relevant by default",
                    "response": ""
                }
                
        except Exception as e:
            logging.error(f"Error in message classification: {e}")
            # Fallback in case of any errors
            return {
                "category": "relevant_query",
                "explanation": f"Error during classification: {str(e)}. Treating as relevant by default",
                "response": ""
            } 