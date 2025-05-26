import json
import logging
import streamlit as st # For potential access to session_state if needed for LLM client

# Assuming llm_models.py and enums.py are accessible in the PYTHONPATH
# You might need to adjust imports based on the actual project structure
from llm_models import LLMFactory, LLMModelType, StreamingLLM
from enums import (
    EmbeddingModelType,
    VectorStoreType,
    RerankerModelType,
    ChunkingStrategyType
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_SMARTER_JEFF_CONFIG = {
    "embedding_model": EmbeddingModelType.MISTRAL.value,
    "vector_store": VectorStoreType.CHROMA.value,
    "reranker": RerankerModelType.COHERE_V3.value,
    "llm_model": LLMModelType.CLAUDE_37_SONNET.value, # Ensure this is a valid key in LLMFactory
    "chunking_strategy": ChunkingStrategyType.HIERARCHICAL.value,
    "hybrid_alpha": 0.5,
    "chunk_size": 500,
    "chunk_overlap": 100,
    "top_k": 5
}

def get_ai_suggested_config(query: str, current_config: dict) -> dict:
    '''
    Uses an LLM to suggest an optimal RAG pipeline configuration based on the user's query.

    Args:
        query (str): The user's query.
        current_config (dict): The current RAG configuration, to be used as a fallback.

    Returns:
        dict: The AI-suggested RAG configuration or the current_config if an error occurs.
    '''
    try:
        # Choose a cost-effective and fast LLM for configuration suggestion.
        # Example: Using Mistral Small, or another suitable model available in LLMFactory.
        # Ensure the chosen LLMModelType enum member is correctly mapped in LLMFactory.
        # For this example, let's assume CLAUDE_37_SONNET can be used,
        # but in a real scenario, a smaller/faster model would be preferable.
        config_llm: StreamingLLM = LLMFactory.create_llm(LLMModelType.CLAUDE_37_SONNET)

        # Construct the prompt
        prompt = f"""
Given the user's query: "{query}"

Suggest an optimal RAG pipeline configuration. Your response MUST be a valid JSON object.
The JSON object should have the following keys, with values chosen from the provided options:

- "embedding_model": Choose one from {EmbeddingModelType.list()}.
- "vector_store": Choose one from {VectorStoreType.list()}.
- "reranker": Choose one from {RerankerModelType.list()}.
- "llm_model": Choose one from {LLMModelType.list()}.
- "chunking_strategy": Choose one from {ChunkingStrategyType.list()}.
- "hybrid_alpha": A float between 0.0 and 1.0. This is only relevant if "vector_store" is "{VectorStoreType.HYBRID.value}". If not, you can set it to 0.5 or omit it.
- "chunk_size": An integer between 100 and 2000.
- "chunk_overlap": An integer between 0 and 500, and it must be less than "chunk_size".
- "top_k": An integer between 1 and 15.

Consider the nature of the query to select the best configuration.
For example:
- For highly technical or code-related queries, smaller 'chunk_size' (e.g., 200-400) and a precise 'embedding_model' might be better.
- For broad, conceptual questions, larger 'chunk_size' (e.g., 500-1000) and a 'chunking_strategy' like "{ChunkingStrategyType.HIERARCHICAL.value}" or "{ChunkingStrategyType.SEMANTIC.value}" could be beneficial.
- If the query implies comparing multiple documents or aspects, a higher 'top_k' (e.g., 5-7) might be useful.
- Choose '{RerankerModelType.NONE.value}' for 'reranker' if the query is simple or speed is paramount.

Current configuration is: {json.dumps(current_config, indent=2)}
If the query is too generic or you are unsure, you can lean towards the current configuration or common defaults.

Return ONLY the JSON object. Do not include any other text before or after the JSON.
Example of a valid JSON response:
{{
  "embedding_model": "{EmbeddingModelType.MISTRAL.value}",
  "vector_store": "{VectorStoreType.CHROMA.value}",
  "reranker": "{RerankerModelType.COHERE_V3.value}",
  "llm_model": "{LLMModelType.CLAUDE_37_SONNET.value}",
  "chunking_strategy": "{ChunkingStrategyType.HIERARCHICAL.value}",
  "hybrid_alpha": 0.4,
  "chunk_size": 450,
  "chunk_overlap": 75,
  "top_k": 4
}}
"""

        logger.info(f"Attempting to get AI suggested config for query: {query}")
        # Use the generate method (non-streaming) for a single JSON output.
        # The LLM used here should be capable of understanding instructions and generating JSON.
        # The 'evaluation_mode=True' might be useful if your LLM classes have special prompting for it,
        # but here we want the LLM to follow our specific JSON generation prompt.
        # We are not evaluating the LLM's response in a RAGAS sense, but its adherence to the format.
        raw_response = config_llm.generate(prompt, context=None, evaluation_mode=False) # evaluation_mode might need adjustment

        logger.info(f"Raw AI response for config: {raw_response}")

        # Attempt to parse the JSON response
        # The LLM might sometimes include markdown ```json ... ``` or other text.
        cleaned_response = raw_response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()
        
        suggested_config = json.loads(cleaned_response)

        # Basic validation (more can be added)
        if not isinstance(suggested_config, dict):
            raise ValueError("AI response was not a JSON object.")

        # Ensure all essential keys are present and have valid enum values
        for key, enum_type in [
            ("embedding_model", EmbeddingModelType),
            ("vector_store", VectorStoreType),
            ("reranker", RerankerModelType),
            ("llm_model", LLMModelType),
            ("chunking_strategy", ChunkingStrategyType)
        ]:
            if key not in suggested_config:
                logger.warning(f"Key '{key}' missing in AI suggestion. Using default.")
                suggested_config[key] = DEFAULT_SMARTER_JEFF_CONFIG[key]
            else:
                try:
                    enum_type.from_string(suggested_config[key]) # Validate if value is in enum
                except ValueError:
                    logger.warning(f"Invalid value '{suggested_config[key]}' for '{key}'. Using default.")
                    suggested_config[key] = DEFAULT_SMARTER_JEFF_CONFIG[key]
        
        # Validate numerical ranges
        suggested_config["chunk_size"] = max(100, min(2000, int(suggested_config.get("chunk_size", DEFAULT_SMARTER_JEFF_CONFIG["chunk_size"]))))
        suggested_config["chunk_overlap"] = max(0, min(suggested_config["chunk_size"] -1 , int(suggested_config.get("chunk_overlap", DEFAULT_SMARTER_JEFF_CONFIG["chunk_overlap"]))))
        suggested_config["top_k"] = max(1, min(15, int(suggested_config.get("top_k", DEFAULT_SMARTER_JEFF_CONFIG["top_k"]))))
        suggested_config["hybrid_alpha"] = max(0.0, min(1.0, float(suggested_config.get("hybrid_alpha", DEFAULT_SMARTER_JEFF_CONFIG["hybrid_alpha"]))))


        logger.info(f"Successfully parsed AI suggested config: {suggested_config}")
        return suggested_config

    except Exception as e:
        logger.error(f"Error in get_ai_suggested_config: {e}", exc_info=True)
        logger.warning("Falling back to current_config due to error.")
        # Fallback to a mix of default and current_config to ensure all keys are present
        fallback_config = DEFAULT_SMARTER_JEFF_CONFIG.copy()
        fallback_config.update(current_config) # Override defaults with any relevant current settings
        return fallback_config

if __name__ == '__main__':
    # Example usage for testing (requires appropriate environment setup for LLMFactory)
    # This part will not be executed by the worker but is useful for local testing.
    # Ensure API keys are set in your environment if you run this directly.
    print("Testing AI Config Suggester...")
    test_query = "Explain quantum entanglement in simple terms."
    test_current_config = {
        "embedding_model": "OpenAI",
        "vector_store": "FAISS",
        "reranker": "None",
        "llm_model": "OpenAI GPT-3.5",
        "chunking_strategy": "Paragraph-based",
        "hybrid_alpha": 0.5,
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "top_k": 3
    }
    
    # Mock LLMFactory and LLM for local testing without real API calls if needed:
    # class MockLLM(StreamingLLM):
    #     def generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False) -> str:
    #         print("---- MockLLM generating for prompt ----")
    #         # print(prompt) # Uncomment to see the prompt
    #         print("---- End of MockLLM prompt ----")
    #         # Return a valid JSON string
    #         return json.dumps({
    #             "embedding_model": "Mistral", "vector_store": "Chroma", "reranker": "Cohere-V3",
    #             "llm_model": "Claude-3.7-Sonnet", "chunking_strategy": "Hierarchical",
    #             "hybrid_alpha": 0.6, "chunk_size": 600, "chunk_overlap": 150, "top_k": 6
    #         })
    #     def stream_generate(self, prompt: str, context: Optional[str] = None, evaluation_mode: bool = False) -> Iterator[str]:
    #         yield self.generate(prompt, context, evaluation_mode)
    #     def get_model_name(self) -> str: return "mock-llm"
    #
    # # Replace LLMFactory temporarily for the test
    # original_llm_factory_create = LLMFactory.create_llm
    # LLMFactory.create_llm = lambda model_type: MockLLM()

    try:
        suggested = get_ai_suggested_config(test_query, test_current_config)
        print("Suggested Configuration:")
        print(json.dumps(suggested, indent=2))
    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        # LLMFactory.create_llm = original_llm_factory_create # Restore original factory
        pass
