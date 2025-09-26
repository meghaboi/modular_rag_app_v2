import unittest
from unittest.mock import patch, MagicMock, call
import os
import logging

# Assuming utils.py is in the parent directory or accessible via PYTHONPATH
from utils.analysis.analysis_utils import determine_prompt_nature

# Disable logging for tests unless specifically testing logging
logging.disable(logging.CRITICAL)

class TestDeterminePromptNature(unittest.TestCase):

    @patch('utils.os.getenv')
    @patch('utils.Anthropic')
    def test_valid_query_high_confidence(self, MockAnthropic, mock_getenv):
        mock_getenv.return_value = "fake_api_key"
        mock_client = MockAnthropic.return_value
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(type="tool_use", name="classify_prompt_nature", input={
                "nature": "question_answering",
                "confidence": 0.9
            })
        ]
        mock_client.messages.create.return_value = mock_response

        result = determine_prompt_nature("What is the capital of France?")
        self.assertEqual(result, "question_answering")
        mock_client.messages.create.assert_called_once()

    @patch('utils.os.getenv')
    @patch('utils.Anthropic')
    def test_valid_query_low_confidence(self, MockAnthropic, mock_getenv):
        mock_getenv.return_value = "fake_api_key"
        mock_client = MockAnthropic.return_value
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(type="tool_use", name="classify_prompt_nature", input={
                "nature": "question_answering",
                "confidence": 0.5
            })
        ]
        mock_client.messages.create.return_value = mock_response

        result = determine_prompt_nature("Tell me about photosynthesis.")
        self.assertEqual(result, "general_discussion")

    @patch('utils.os.getenv')
    @patch('utils.Anthropic')
    @patch('utils.logging.error')
    def test_anthropic_api_error(self, mock_log_error, MockAnthropic, mock_getenv):
        mock_getenv.return_value = "fake_api_key"
        mock_client = MockAnthropic.return_value
        mock_client.messages.create.side_effect = Exception("API Failure")

        result = determine_prompt_nature("This will cause an error.")
        self.assertEqual(result, "general_discussion")
        mock_log_error.assert_called_with(
            "Error determining prompt nature for query 'This will cause an error.': API Failure",
            exc_info=True
        )

    @patch('utils.os.getenv')
    @patch('utils.Anthropic')
    def test_unexpected_nature_string(self, MockAnthropic, mock_getenv):
        mock_getenv.return_value = "fake_api_key"
        mock_client = MockAnthropic.return_value
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(type="tool_use", name="classify_prompt_nature", input={
                "nature": "unexpected_value", # Not in ALLOWED_NATURES
                "confidence": 0.9
            })
        ]
        mock_client.messages.create.return_value = mock_response

        result = determine_prompt_nature("A query leading to unexpected nature.")
        self.assertEqual(result, "general_discussion")

    @patch('utils.os.getenv')
    @patch('utils.Anthropic')
    @patch('utils.logging.warning')
    def test_no_tool_call_in_response(self, mock_log_warning, MockAnthropic, mock_getenv):
        mock_getenv.return_value = "fake_api_key"
        mock_client = MockAnthropic.return_value
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Some text")] # No tool_use
        mock_client.messages.create.return_value = mock_response

        query = "Query with no tool call."
        result = determine_prompt_nature(query)
        self.assertEqual(result, "general_discussion")
        mock_log_warning.assert_called_with(
            f"No tool call found in response for query '{query}'. Falling back to default."
        )

    @patch('utils.os.getenv')
    @patch('utils.logging.error')
    def test_missing_api_key(self, mock_log_error, mock_getenv):
        mock_getenv.return_value = None # Simulate missing API key

        result = determine_prompt_nature("Query when API key is missing.")
        self.assertEqual(result, "general_discussion")
        mock_log_error.assert_called_with(
            "ANTHROPIC_API_KEY not found. Cannot determine prompt nature."
        )

if __name__ == '__main__':
    unittest.main()
