import logging
from typing import List, Optional
from sympy import re
from models.llm_models import StreamingLLM
from pipeline.rag_pipeline import RAGPipeline

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

    user_prompt_for_extraction = (
        "Please extract the key topics or main points from the following text. "
        "Present them as a numbered list (e.g., 1. Point one, 2. Point two, ...).\n\n"
        "Text:\n"
        f"{file_content}"
    )

    extraction_system_prompt = "You are a helpful assistant tasked with extracting key information."

    try:
        response_text, _ = llm.generate(
            prompt=user_prompt_for_extraction,
            context=None,
            system_prompt_override=extraction_system_prompt
        )

        if not response_text:
            logger.warning("LLM returned no response for main point extraction.")
            return []

        points = []
        for line in response_text.splitlines():
            match = re.match(r"^\s*\d+\.\s*(.+)", line)
            if match:
                points.append(match.group(1).strip())

        if not points:
            logger.info("No numbered list points found in LLM response for extraction. Returning raw response lines if any.")
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
    if not rag_pipeline.llm:
        logger.error("RAG pipeline's LLM is not configured.")
        return "Error: RAG pipeline's LLM is not configured."

    try:
        logger.info(f"Generating summary for point: '{main_point_query}'")
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
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

    logger.info("summarizer_module.py executed directly (for testing/dev purposes).")
