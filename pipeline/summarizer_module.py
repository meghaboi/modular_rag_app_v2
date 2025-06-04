import logging
from typing import List, Optional
from models.llm_models import StreamingLLM # Assuming llm_models.py is in the same directory or accessible
from pipeline.rag_pipeline import RAGPipeline # Assuming rag_pipeline.py is accessible

# Configure logging for this module
logger = logging.getLogger(__name__)

def extract_main_points(file_path: str, llm: StreamingLLM) -> List[str]:
    """
    Reads a text file, sends its content to an LLM to extract main points,
    and returns them as a list of strings.

    Args:
        file_path: The path to the text file.
        llm: An instance of a StreamingLLM to use for point extraction.

    Returns:
        A list of extracted main points, or an empty list if an error occurs
        or no points are found.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return []

    if not file_content.strip():
        logger.warning(f"File {file_path} is empty or contains only whitespace.")
        return []

    # Construct user prompt for extracting main points
    # Instructing the LLM to return a numbered list for easier parsing.
    user_prompt_for_extraction = (
        "Please extract the key topics or main points from the following text. "
        "Present them as a numbered list (e.g., 1. Point one, 2. Point two, ...).\n\n"
        "Text:\n"
        f"{file_content}"
    )

    # System prompt for neutral extraction (bypassing default persona if any)
    # An empty string or a very basic prompt can be used.
    extraction_system_prompt = "You are a helpful assistant tasked with extracting key information."

    try:
        # Call the LLM's generate method with the custom system prompt for extraction
        # evaluation_mode might also be considered if it more directly disables default system prompts.
        # Using system_prompt_override as implemented in Step 1.
        response_text, _ = llm.generate(
            prompt=user_prompt_for_extraction,
            context=None, # No additional context needed for this task
            system_prompt_override=extraction_system_prompt
        )

        if not response_text:
            logger.warning("LLM returned no response for main point extraction.")
            return []

        # Parse the response to extract numbered list items
        points = []
        # Regex to find lines starting with a number, a dot, and optional space
        # This is a basic parser; more robust parsing might be needed depending on LLM output variance.
        import re
        for line in response_text.splitlines():
            match = re.match(r"^\s*\d+\.\s*(.+)", line)
            if match:
                points.append(match.group(1).strip())

        if not points:
            logger.info("No numbered list points found in LLM response for extraction. Returning raw response lines if any.")
            # Fallback: if no numbered list, maybe the LLM just listed them.
            # This is a simple fallback; might need refinement.
            raw_lines = [line.strip() for line in response_text.splitlines() if line.strip()]
            if raw_lines:
                return raw_lines
            return []

        logger.info(f"Extracted {len(points)} main points from {file_path}.")
        return points

    except Exception as e:
        logger.error(f"Error during LLM call for main point extraction: {e}", exc_info=True)
        return []

def generate_summary_for_point(
    main_point_query: str,
    rag_pipeline: RAGPipeline,
    summarization_system_prompt: str
) -> str:
    """
    Generates a summary for a given main point using the RAG pipeline
    with a specific system prompt for summarization.

    Args:
        main_point_query: The main point/topic to summarize (acts as the query).
        rag_pipeline: An instance of the RAGPipeline.
        summarization_system_prompt: The system prompt to guide the LLM for summarization.

    Returns:
        The generated summary string, or an error message string if summarization fails.
    """
    if not rag_pipeline:
        logger.error("RAG pipeline instance is not provided or is None.")
        return "Error: RAG pipeline is not available."
    if not rag_pipeline.llm: # Check if the pipeline has an LLM configured
        logger.error("RAG pipeline's LLM is not configured.")
        return "Error: RAG pipeline's LLM is not configured."

    try:
        logger.info(f"Generating summary for point: '{main_point_query}'")
        # The RAGPipeline's run method was updated in Step 2 to accept system_prompt_override
        summary_text, _, _ = rag_pipeline.run(
            query=main_point_query,
            system_prompt_override=summarization_system_prompt
        )

        if summary_text:
            logger.info("Summary generated successfully.")
            return summary_text
        else:
            logger.warning("Summarization returned empty text.")
            return "Summarization resulted in an empty response."

    except Exception as e:
        logger.error(f"Error during RAG pipeline execution for summarization: {e}", exc_info=True)
        return f"Error generating summary: {str(e)}"

if __name__ == '__main__':
    # This section can be used for basic testing of the module if run directly.
    # For actual testing, you'd need to mock LLM and RAGPipeline or have a test setup.
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

    logger.info("summarizer_module.py executed directly (for testing/dev purposes).")
    # Example (requires mock objects or a running environment):
    # class MockLLM(StreamingLLM):
    #     def generate(self, prompt: str, context: Optional[str] = None, system_prompt_override: Optional[str] = None, evaluation_mode: bool = False) -> tuple[str, Optional[dict[str, int]]]:
    #         logger.info(f"MockLLM generate called with system_prompt_override: {system_prompt_override}")
    #         if "extract" in prompt:
    #             return "1. First point\n2. Second key topic\n3. Another item", {"total_tokens": 10}
    #         return "Mocked response", {"total_tokens": 5}
    #     def stream_generate(self, prompt: str, context: Optional[str] = None, system_prompt_override: Optional[str] = None, evaluation_mode: bool = False) -> list[str]:
    #         return ["mock stream part"]
    #     def get_model_name(self) -> str:
    #         return "mock_llm"

    # test_llm = MockLLM()
    # Create a dummy file for testing extract_main_points
    # dummy_file_path = "dummy_test_file.txt"
    # with open(dummy_file_path, "w") as f:
    #     f.write("This is a test document.\nIt has several lines.\nTopic A is important.\nAlso consider Topic B.")

    # points = extract_main_points(dummy_file_path, test_llm)
    # logger.info(f"Extracted points: {points}")

    # class MockRAGPipeline:
    #     def __init__(self, llm):
    #         self.llm = llm # RAGPipeline needs an LLM
    #     def run(self, query: str, system_prompt_override: Optional[str] = None):
    #         logger.info(f"MockRAGPipeline run called for query '{query}' with system_prompt_override: {system_prompt_override}")
    #         return f"Summary for {query}", ["context1"], {"total_tokens": 20}

    # test_rag_pipeline = MockRAGPipeline(test_llm)
    # summary = generate_summary_for_point("First point", test_rag_pipeline, "Summarize this topic.")
    # logger.info(f"Generated summary: {summary}")

    # import os
    # if os.path.exists(dummy_file_path):
    #    os.remove(dummy_file_path)
