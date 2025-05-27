import unittest
from unittest.mock import patch, MagicMock
import json 

from subject_handler import get_subject_configuration
from subject_configs import get_subject_config, SubjectConfig # SUBJECT_CONFIGS is not directly used by tests but good for reference

# Define a custom OpenAIError for mocking API errors
class OpenAIError(Exception): # Can be any exception type for the mock
    pass

class TestSubjectHandler(unittest.TestCase):

    @patch('subject_handler.openai.ChatCompletion.create')
    def test_llm_success_valid_config(self, mock_chat_completion_create):
        print("Running test_llm_success_valid_config")
        expected_config_dict = {
            "chunk_size": 100, 
            "chunk_overlap": 10, 
            "similarity_threshold": 0.8, 
            "max_tokens": 500, 
            "temperature": 0.5, 
            "system_prompt": "Test prompt"
        }
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.function_call = MagicMock()
        mock_response.choices[0].message.function_call.arguments = json.dumps(expected_config_dict)
        
        mock_chat_completion_create.return_value = mock_response

        subject = "test_subject"
        query = "test_query"
        result = get_subject_configuration(subject, query)

        mock_chat_completion_create.assert_called_once()
        args, kwargs = mock_chat_completion_create.call_args
        
        self.assertEqual(kwargs['model'], "gpt-4")
        self.assertEqual(len(kwargs['functions']), 1)
        self.assertEqual(kwargs['functions'][0]['name'], "get_subject_config")
        self.assertEqual(kwargs['functions'][0]['parameters']['type'], "object") # Basic check of schema
        self.assertEqual(kwargs['function_call'], {"name": "get_subject_config"})
        
        messages = kwargs['messages']
        self.assertEqual(messages[0]['role'], "system")
        self.assertIn(subject, messages[0]['content'])
        self.assertIn(query, messages[0]['content'])
        self.assertEqual(messages[1]['role'], "user")
        self.assertIn(subject, messages[1]['content'])
        self.assertIn(query, messages[1]['content'])

        self.assertEqual(result, expected_config_dict)
        print("Finished test_llm_success_valid_config")

    @patch('subject_handler.openai.ChatCompletion.create')
    def test_llm_failure_api_error(self, mock_chat_completion_create):
        print("Running test_llm_failure_api_error")
        # Make the mocked create call raise our custom OpenAIError (or any Exception)
        mock_chat_completion_create.side_effect = OpenAIError("Simulated API Error")

        subject = "mathematics"
        query = "test_query_for_math_fallback"
        result = get_subject_configuration(subject, query)

        mock_chat_completion_create.assert_called_once() # Verify it was called
        
        # Based on the corrected fallback logic in subject_handler.py
        expected_fallback_config_obj = get_subject_config(subject)
        expected_dict = {
            "chunk_size": expected_fallback_config_obj.chunk_size,
            "chunk_overlap": expected_fallback_config_obj.chunk_overlap,
            "similarity_threshold": 0.7,
            "max_tokens": 1000,
            "temperature": 0.7,
            "system_prompt": "You are a helpful assistant."
        }
        self.assertEqual(result, expected_dict)
        print("Finished test_llm_failure_api_error")

    @patch('subject_handler.eval') # Mock the eval function within subject_handler
    @patch('subject_handler.openai.ChatCompletion.create')
    def test_llm_success_eval_failure(self, mock_chat_completion_create, mock_eval):
        print("Running test_llm_success_eval_failure (formerly malformed_json)")
        
        # Configure openai.ChatCompletion.create to return a response
        # The content of 'arguments' doesn't matter as much now since eval is mocked
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.function_call = MagicMock()
        # It could still be a valid JSON string, eval is what we're testing the failure of
        mock_response.choices[0].message.function_call.arguments = json.dumps({"some_key": "some_value"})
        mock_chat_completion_create.return_value = mock_response

        # Make the mocked eval raise an exception
        mock_eval.side_effect = SyntaxError("Simulated eval failure")

        subject = "general" 
        query = "test_query_for_general_fallback_on_eval_error"
        result = get_subject_configuration(subject, query)

        mock_chat_completion_create.assert_called_once()
        mock_eval.assert_called_once_with(json.dumps({"some_key": "some_value"})) # Assert eval was called with the arguments

        expected_fallback_config_obj = get_subject_config(subject) 
        expected_dict = {
            "chunk_size": expected_fallback_config_obj.chunk_size,
            "chunk_overlap": expected_fallback_config_obj.chunk_overlap,
            "similarity_threshold": 0.7,
            "max_tokens": 1000,
            "temperature": 0.7,
            "system_prompt": "You are a helpful assistant."
        }
        self.assertEqual(result, expected_dict)
        print("Finished test_llm_success_eval_failure")

    @patch('subject_handler.openai.ChatCompletion.create')
    def test_fallback_for_unknown_subject(self, mock_chat_completion_create):
        print("Running test_fallback_for_unknown_subject")
        mock_chat_completion_create.side_effect = OpenAIError("Simulated API Error for unknown subject")

        subject = "unknown_subject_for_fallback"
        query = "test_query"
        result = get_subject_configuration(subject, query)

        mock_chat_completion_create.assert_called_once()

        # get_subject_config defaults to "general" for unknown subjects
        expected_fallback_config_obj = get_subject_config("general") 
        expected_dict = {
            "chunk_size": expected_fallback_config_obj.chunk_size,
            "chunk_overlap": expected_fallback_config_obj.chunk_overlap,
            "similarity_threshold": 0.7,
            "max_tokens": 1000,
            "temperature": 0.7,
            "system_prompt": "You are a helpful assistant."
        }
        self.assertEqual(result, expected_dict)
        print("Finished test_fallback_for_unknown_subject")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    print("Test definitions processed by test_subject_handler.py")
