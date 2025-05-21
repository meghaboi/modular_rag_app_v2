from typing import Dict, Any
import openai
from subject_configs import SubjectConfig, get_subject_config

def get_subject_configuration(subject: str) -> Dict[str, Any]:
    """
    Get the optimal RAG configuration for a specific subject using OpenAI's function calling API.
    Falls back to predefined configurations if API call fails.
    """
    try:
        # Define the function schema for OpenAI
        functions = [{
            "name": "get_subject_config",
            "description": "Get the optimal RAG configuration for a specific subject",
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_size": {
                        "type": "integer",
                        "description": "Size of text chunks for processing"
                    },
                    "chunk_overlap": {
                        "type": "integer",
                        "description": "Overlap between chunks"
                    },
                    "similarity_threshold": {
                        "type": "number",
                        "description": "Threshold for similarity matching"
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens for response generation"
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Temperature for response generation"
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "System prompt for the LLM"
                    }
                },
                "required": ["chunk_size", "chunk_overlap", "similarity_threshold", 
                           "max_tokens", "temperature", "system_prompt"]
            }
        }]

        # Call OpenAI API to get optimal configuration
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"Determine the optimal RAG configuration for {subject} textbooks."},
                {"role": "user", "content": f"What is the optimal RAG configuration for processing {subject} textbooks?"}
            ],
            functions=functions,
            function_call={"name": "get_subject_config"}
        )

        # Extract configuration from response
        config = response.choices[0].message.function_call.arguments
        return eval(config)  # Convert string to dict

    except Exception as e:
        print(f"Error getting configuration from OpenAI: {e}")
        # Fall back to predefined configuration
        config = get_subject_config(subject)
        return {
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "similarity_threshold": config.similarity_threshold,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "system_prompt": config.system_prompt
        }

def update_rag_configuration(subject: str, rag_pipeline) -> None:
    """
    Update the RAG pipeline configuration based on the selected subject
    """
    config = get_subject_configuration(subject)
    
    # Update RAG pipeline parameters
    rag_pipeline.chunk_size = config["chunk_size"]
    rag_pipeline.chunk_overlap = config["chunk_overlap"]
    rag_pipeline.similarity_threshold = config["similarity_threshold"]
    rag_pipeline.max_tokens = config["max_tokens"]
    rag_pipeline.temperature = config["temperature"]
    rag_pipeline.system_prompt = config["system_prompt"] 