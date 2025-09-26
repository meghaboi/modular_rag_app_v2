import logging
import re
from typing import List

from models.llm_models import StreamingLLM
from pipeline.rag_pipeline import RAGPipeline
from prompts import get_provider

logger = logging.getLogger(__name__)

# Constants
NUMBERED_LIST_PATTERN = r"^\s*\d+\.\s*(.+)"
DEFAULT_ENCODING = 'utf-8'
EXTRACTION_SYSTEM_PROMPT = "You are a helpful assistant tasked with extracting key information."
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


class FileReadError(Exception):
    """Raised when file reading operations fail."""
    pass

class LLMProcessingError(Exception):
    """Raised when LLM processing operations fail."""
    pass

class PointExtractor:
    """Handles extraction of main points from text using LLM."""

    def __init__(self, llm: StreamingLLM):
        self._llm = llm
        self._summarizer_provider = get_provider('summarizer')

    def extract_from_response(self, response_text: str) -> List[str]:
        """Extract numbered points from LLM response text."""
        if not response_text:
            return []

        points = self._extract_numbered_points(response_text)
        if points:
            return points

        return self._extract_raw_lines(response_text)

    def _extract_numbered_points(self, response_text: str) -> List[str]:
        """Extract points from numbered list format."""
        points = []
        for line in response_text.splitlines():
            match = re.match(NUMBERED_LIST_PATTERN, line)
            if match:
                points.append(match.group(1).strip())
        return points

    def _extract_raw_lines(self, response_text: str) -> List[str]:
        """Extract non-empty lines as fallback."""
        return [line.strip() for line in response_text.splitlines() if line.strip()]

class FileReader:
    """Handles file reading operations with proper error handling."""

    @staticmethod
    def read_text_file(file_path: str) -> str:
        """Read text file content with error handling."""
        try:
            with open(file_path, 'r', encoding=DEFAULT_ENCODING) as file:
                content = file.read()

            if not content.strip():
                raise FileReadError(f"File {file_path} is empty or contains only whitespace")

            return content

        except FileNotFoundError:
            raise FileReadError(f"File not found: {file_path}")
        except Exception as e:
            raise FileReadError(f"Error reading file {file_path}: {e}")

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
        file_content = FileReader.read_text_file(file_path)
        extractor = PointExtractor(llm)

        user_prompt = extractor._summarizer_provider.get_prompt('main_points', text=file_content)

        response_text = _generate_llm_response(llm, user_prompt)
        points = extractor.extract_from_response(response_text)

        logger.info(f"Extracted {len(points)} main points from {file_path}")
        return points

    except (FileReadError, LLMProcessingError) as e:
        logger.error(str(e))
        return []

def _generate_llm_response(llm: StreamingLLM, user_prompt: str) -> str:
    """Generate response from LLM with error handling."""
    try:
        response_text, _ = llm.generate(
            prompt=user_prompt,
            context=None,
            system_prompt_override=EXTRACTION_SYSTEM_PROMPT
        )

        if not response_text:
            raise LLMProcessingError("LLM returned no response for main point extraction")

        return response_text

    except Exception as e:
        raise LLMProcessingError(f"Error during LLM call for main point extraction: {e}")

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
    try:
        _validate_rag_pipeline(rag_pipeline)

        logger.info(f"Generating summary for point: '{main_point_query}'")

        summary_text, _, _ = rag_pipeline.run(
            query=main_point_query,
            system_prompt_override=summarization_system_prompt
        )

        return _process_summary_result(summary_text)

    except Exception as e:
        error_message = f"Error generating summary: {str(e)}"
        logger.error(f"Error during RAG pipeline execution for summarization: {e}", exc_info=True)
        return error_message

def _validate_rag_pipeline(rag_pipeline: RAGPipeline) -> None:
    """Validate RAG pipeline configuration."""
    if not rag_pipeline:
        raise ValueError("RAG pipeline instance is not provided or is None")
    if not rag_pipeline.llm:
        raise ValueError("RAG pipeline's LLM is not configured")

def _process_summary_result(summary_text: str) -> str:
    """Process and validate summary result."""
    if summary_text:
        logger.info("Summary generated successfully")
        return summary_text
    else:
        logger.warning("Summarization returned empty text")
        return "Summarization resulted in an empty response."