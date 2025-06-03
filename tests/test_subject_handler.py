import unittest
from unittest.mock import patch, MagicMock, call  # Added call
import json
import logging  # Ensure logging is imported at the top

from utils.subject_handler import (
    get_subject_configuration,
    get_config_by_prompt_nature,
    update_rag_configuration,
)  # Corrected path, consolidated
from utils.subject_configs import (
    get_subject_config,
    SubjectConfig,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    DEFAULT_HYBRID_ALPHA,
)  # Corrected path, consolidated
from utils.enums import (
    EmbeddingModelType,
    VectorStoreType,
    RerankerModelType,
    LLMModelType,
    ChunkingStrategyType,
)  # Corrected path, consolidated

# Disable logging for most tests unless specifically testing logging behavior
logging.disable(logging.CRITICAL)


# Define a custom OpenAIError for mocking API errors
class OpenAIError(Exception):  # Can be any exception type for the mock
    pass


class TestSubjectHandler(unittest.TestCase):

    @patch("utils.subject_handler.openai.ChatCompletion.create")  # Corrected patch path
    def test_llm_success_valid_config(self, mock_chat_completion_create):
        print("Running test_llm_success_valid_config")
        expected_config_dict = {
            "chunk_size": 100,
            "chunk_overlap": 10,
            "similarity_threshold": 0.8,
            "max_tokens": 500,
            "temperature": 0.5,
            "system_prompt": "Test prompt",
        }
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.function_call = MagicMock()
        mock_response.choices[0].message.function_call.arguments = json.dumps(
            expected_config_dict
        )

        mock_chat_completion_create.return_value = mock_response

        subject = "test_subject"
        query = "test_query"
        result = get_subject_configuration(subject, query)

        mock_chat_completion_create.assert_called_once()
        args, kwargs = mock_chat_completion_create.call_args

        self.assertEqual(kwargs["model"], "gpt-4")
        self.assertEqual(len(kwargs["functions"]), 1)
        self.assertEqual(kwargs["functions"][0]["name"], "get_subject_config")
        self.assertEqual(
            kwargs["functions"][0]["parameters"]["type"], "object"
        )  # Basic check of schema
        self.assertEqual(kwargs["function_call"], {"name": "get_subject_config"})

        messages = kwargs["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn(subject, messages[0]["content"])
        self.assertIn(query, messages[0]["content"])
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn(subject, messages[1]["content"])
        self.assertIn(query, messages[1]["content"])

        self.assertEqual(result, expected_config_dict)
        print("Finished test_llm_success_valid_config")

    @patch("utils.subject_handler.openai.ChatCompletion.create")  # Corrected patch path
    def test_llm_failure_api_error(self, mock_chat_completion_create):
        print("Running test_llm_failure_api_error")
        # Make the mocked create call raise our custom OpenAIError (or any Exception)
        mock_chat_completion_create.side_effect = OpenAIError("Simulated API Error")

        subject = "mathematics"
        query = "test_query_for_math_fallback"
        result = get_subject_configuration(subject, query)

        mock_chat_completion_create.assert_called_once()  # Verify it was called

        # Based on the corrected fallback logic in subject_handler.py
        expected_fallback_config_obj = get_subject_config(subject)
        expected_dict = {
            "chunk_size": expected_fallback_config_obj.chunk_size,
            "chunk_overlap": expected_fallback_config_obj.chunk_overlap,
            "similarity_threshold": 0.7,
            "max_tokens": 1000,
            "temperature": 0.7,
            "system_prompt": "You are a helpful assistant.",
        }
        self.assertEqual(result, expected_dict)
        print("Finished test_llm_failure_api_error")

    @patch("utils.subject_handler.eval")  # Corrected patch path
    @patch("utils.subject_handler.openai.ChatCompletion.create")  # Corrected patch path
    def test_llm_success_eval_failure(self, mock_chat_completion_create, mock_eval):
        print("Running test_llm_success_eval_failure (formerly malformed_json)")

        # Configure openai.ChatCompletion.create to return a response
        # The content of 'arguments' doesn't matter as much now since eval is mocked
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.function_call = MagicMock()
        # It could still be a valid JSON string, eval is what we're testing the failure of
        mock_response.choices[0].message.function_call.arguments = json.dumps(
            {"some_key": "some_value"}
        )
        mock_chat_completion_create.return_value = mock_response

        # Make the mocked eval raise an exception
        mock_eval.side_effect = SyntaxError("Simulated eval failure")

        subject = "general"
        query = "test_query_for_general_fallback_on_eval_error"
        result = get_subject_configuration(subject, query)

        mock_chat_completion_create.assert_called_once()
        mock_eval.assert_called_once_with(
            json.dumps({"some_key": "some_value"})
        )  # Assert eval was called with the arguments

        expected_fallback_config_obj = get_subject_config(subject)
        expected_dict = {
            "chunk_size": expected_fallback_config_obj.chunk_size,
            "chunk_overlap": expected_fallback_config_obj.chunk_overlap,
            "similarity_threshold": 0.7,
            "max_tokens": 1000,
            "temperature": 0.7,
            "system_prompt": "You are a helpful assistant.",
        }
        self.assertEqual(result, expected_dict)
        print("Finished test_llm_success_eval_failure")

    @patch("utils.subject_handler.openai.ChatCompletion.create")  # Corrected patch path
    def test_fallback_for_unknown_subject(self, mock_chat_completion_create):
        print("Running test_fallback_for_unknown_subject")
        mock_chat_completion_create.side_effect = OpenAIError(
            "Simulated API Error for unknown subject"
        )

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
            "system_prompt": "You are a helpful assistant.",
        }
        self.assertEqual(result, expected_dict)
        print("Finished test_fallback_for_unknown_subject")


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
    print("Test definitions processed by test_subject_handler.py")


# It's good practice to keep related tests in the same file if they test the same module.
# We can add new classes for the new functions or extend the existing one if preferred.
# For clarity, let's add new classes.

# All necessary imports moved to the top.
# Ensure no re-imports of logging, subject_handler functions, subject_configs, or enums below.


class TestGetConfigByPromptNature(unittest.TestCase):

    @patch("utils.subject_handler.determine_prompt_nature")  # Corrected patch path
    @patch("utils.subject_handler.get_subject_config")  # Corrected patch path
    @patch("utils.subject_handler.logging.info")  # Corrected patch path
    def test_specific_nature_returned(
        self, mock_log_info, mock_get_subject_config, mock_determine_prompt_nature
    ):
        mock_determine_prompt_nature.return_value = "question_answering"
        expected_config = SubjectConfig(  # SubjectConfig is imported at the top
            chunk_size=200, chunk_overlap=20, top_k=2, hybrid_alpha=0.2
        )
        mock_get_subject_config.return_value = expected_config

        query = "What is the capital of Testland?"
        result_config = get_config_by_prompt_nature(
            query
        )  # get_config_by_prompt_nature imported at top

        mock_determine_prompt_nature.assert_called_once_with(query)
        mock_get_subject_config.assert_called_once_with("question_answering")
        self.assertEqual(result_config, expected_config)
        self.assertIn(
            call(  # call is now imported
                f"Determined prompt nature: question_answering for query: '{query[:50]}...'"
            ),
            mock_log_info.call_args_list,
        )
        self.assertIn(
            call(  # call is now imported
                f"Using configuration for 'question_answering': ChunkSize={expected_config.chunk_size}, Overlap={expected_config.chunk_overlap}, TopK={expected_config.top_k}, Alpha={expected_config.hybrid_alpha}"
            ),
            mock_log_info.call_args_list,
        )

    @patch("utils.subject_handler.determine_prompt_nature")  # Corrected patch path
    @patch("utils.subject_handler.get_subject_config")  # Corrected patch path
    def test_general_discussion_nature(
        self, mock_get_subject_config, mock_determine_prompt_nature
    ):
        mock_determine_prompt_nature.return_value = "general_discussion"
        general_config = SubjectConfig(
            chunk_size=500, chunk_overlap=50, top_k=3, hybrid_alpha=0.5
        )  # Example
        mock_get_subject_config.return_value = general_config

        query = "Just chatting about stuff."
        result_config = get_config_by_prompt_nature(query)

        mock_determine_prompt_nature.assert_called_once_with(query)
        mock_get_subject_config.assert_called_once_with("general_discussion")
        self.assertEqual(result_config, general_config)


class TestUpdateRagConfiguration(unittest.TestCase):

    def setUp(self):
        # Basic session state mock for all tests in this class
        self.mock_session_state = {  # Uses DEFAULT values from utils.subject_configs
            "chunk_size": DEFAULT_CHUNK_SIZE,
            "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
            "top_k": DEFAULT_TOP_K,
            "hybrid_alpha": DEFAULT_HYBRID_ALPHA,
            "embedding_model": "OPENAI",
            "vector_store": "FAISS",
            "reranker": "COHERE_V2",
            "llm_model": "CLAUDE_37_SONNET",
            "chunking_strategy": "HIERARCHICAL",
            "file_path": "/fake/path/to/file.txt",
            "pipeline": MagicMock(),  # Mock the pipeline object itself
        }

    @patch("utils.subject_handler.st")  # Corrected patch path
    @patch("utils.subject_handler.get_config_by_prompt_nature")  # Corrected patch path
    @patch("utils.subject_handler.initialize_pipeline")  # Corrected patch path
    @patch("utils.subject_handler.logging")  # Corrected patch path
    def test_config_matches_current_returns_none(
        self,
        mock_logging,
        mock_initialize_pipeline,
        mock_get_config_by_prompt_nature,
        mock_st,
    ):
        mock_st.session_state = self.mock_session_state

        matching_config = (
            SubjectConfig(  # Uses DEFAULT values from utils.subject_configs
                chunk_size=DEFAULT_CHUNK_SIZE,
                chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                top_k=DEFAULT_TOP_K,
                hybrid_alpha=DEFAULT_HYBRID_ALPHA,
            )
        )
        mock_get_config_by_prompt_nature.return_value = matching_config

        query = "A standard query"
        result = update_rag_configuration(
            query=query, pipeline=self.mock_session_state["pipeline"]
        )

        self.assertIsNone(result)
        mock_initialize_pipeline.assert_not_called()
        mock_get_config_by_prompt_nature.assert_called_once_with(query)
        mock_logging.info.assert_any_call(
            f"Current RAG parameters already match query nature-derived settings. No pipeline re-initialization needed for query: '{query[:50]}...'"
        )

    @patch("utils.subject_handler.st")  # Corrected patch path
    @patch("utils.subject_handler.get_config_by_prompt_nature")  # Corrected patch path
    @patch("utils.subject_handler.initialize_pipeline")  # Corrected patch path
    @patch("utils.subject_handler.logging")  # Corrected patch path
    def test_config_differs_updates_session_and_pipeline(
        self,
        mock_logging,
        mock_initialize_pipeline,
        mock_get_config_by_prompt_nature,
        mock_st,
    ):
        mock_st.session_state = (
            self.mock_session_state.copy()
        )  # Use a copy to allow modification

        new_config = SubjectConfig(  # Uses DEFAULT values from utils.subject_configs for comparison
            chunk_size=1000,  # Different from DEFAULT_CHUNK_SIZE (from utils.subject_configs)
            chunk_overlap=200,  # Different from DEFAULT_CHUNK_OVERLAP (from utils.subject_configs)
            top_k=10,  # Different from DEFAULT_TOP_K (from utils.subject_configs)
            hybrid_alpha=0.8,  # Different from DEFAULT_HYBRID_ALPHA (from utils.subject_configs)
        )
        mock_get_config_by_prompt_nature.return_value = new_config
        mock_pipeline_instance = MagicMock()
        mock_initialize_pipeline.return_value = mock_pipeline_instance

        query = "A query requiring new config"
        result = update_rag_configuration(
            query=query, pipeline=self.mock_session_state["pipeline"]
        )

        self.assertTrue(result)
        mock_get_config_by_prompt_nature.assert_called_once_with(query)

        # Check session state updates
        self.assertEqual(mock_st.session_state["chunk_size"], new_config.chunk_size)
        self.assertEqual(
            mock_st.session_state["chunk_overlap"], new_config.chunk_overlap
        )
        self.assertEqual(mock_st.session_state["top_k"], new_config.top_k)
        self.assertEqual(mock_st.session_state["hybrid_alpha"], new_config.hybrid_alpha)

        # Check initialize_pipeline call
        mock_initialize_pipeline.assert_called_once_with(
            file_path=self.mock_session_state["file_path"],
            embedding_model_enum=EmbeddingModelType.OPENAI,
            vector_store_enum=VectorStoreType.FAISS,
            reranker_enum=RerankerModelType.COHERE_V2,
            llm_enum=LLMModelType.CLAUDE_37_SONNET,
            chunking_strategy_enum=ChunkingStrategyType.HIERARCHICAL,
            hybrid_alpha=new_config.hybrid_alpha,
            chunk_size=new_config.chunk_size,
            chunk_overlap=new_config.chunk_overlap,
            top_k=new_config.top_k,
        )
        self.assertEqual(mock_st.session_state["pipeline"], mock_pipeline_instance)
        mock_logging.info.assert_any_call(
            f"Updating RAG parameters based on query nature. New config: ChunkSize={new_config.chunk_size}, Overlap={new_config.chunk_overlap}, TopK={new_config.top_k}, Alpha={new_config.hybrid_alpha} for query: '{query[:50]}...'"
        )

    @patch("utils.subject_handler.st")  # Corrected patch path
    @patch("utils.subject_handler.get_config_by_prompt_nature")  # Corrected patch path
    @patch("utils.subject_handler.initialize_pipeline")  # Corrected patch path
    @patch(
        "utils.subject_handler.get_subject_config"
    )  # Corrected patch path (for fallback)
    @patch("utils.subject_handler.logging")  # Corrected patch path
    def test_get_config_by_prompt_nature_returns_none_no_subject_fallback(
        self,
        mock_logging,
        mock_get_subject_config_fallback,
        mock_initialize_pipeline,
        mock_get_config_by_prompt_nature,
        mock_st,
    ):
        mock_st.session_state = self.mock_session_state.copy()
        mock_get_config_by_prompt_nature.return_value = None  # Simulate failure

        query = "Query where nature determination fails"
        # Ensure no subject is passed for this test case, or subject is None
        result = update_rag_configuration(
            query=query, pipeline=self.mock_session_state["pipeline"], subject=None
        )

        self.assertIsNone(result)  # As per example logic in update_rag_configuration
        mock_initialize_pipeline.assert_not_called()
        mock_get_subject_config_fallback.assert_not_called()  # Fallback should not be called
        mock_logging.warning.assert_any_call(
            f"Could not determine configuration for query: '{query[:50]}...'. No update will be performed based on nature."
        )
        mock_logging.warning.assert_any_call(
            "No subject provided for fallback. No RAG configuration update."
        )

    @patch("utils.subject_handler.st")  # Corrected patch path
    @patch("utils.subject_handler.get_config_by_prompt_nature")  # Corrected patch path
    @patch("utils.subject_handler.initialize_pipeline")  # Corrected patch path
    @patch("utils.subject_handler.logging")  # Corrected patch path
    def test_initialize_pipeline_fails_returns_false(
        self,
        mock_logging,
        mock_initialize_pipeline,
        mock_get_config_by_prompt_nature,
        mock_st,
    ):
        mock_st.session_state = self.mock_session_state.copy()

        new_config = SubjectConfig(
            chunk_size=100, chunk_overlap=10, top_k=1, hybrid_alpha=0.1
        )  # Different config
        mock_get_config_by_prompt_nature.return_value = new_config
        mock_initialize_pipeline.return_value = (
            None  # Simulate pipeline initialization failure
        )

        query = "Query leading to pipeline init failure"
        result = update_rag_configuration(
            query=query, pipeline=self.mock_session_state["pipeline"]
        )

        self.assertFalse(result)
        mock_initialize_pipeline.assert_called_once()  # It should be called
        mock_logging.error.assert_any_call(
            f"Failed to reinitialize RAG pipeline based on query nature for query: '{query[:50]}...'. initialize_pipeline returned None."
        )

    @patch("utils.subject_handler.st")  # Corrected patch path
    @patch("utils.subject_handler.get_config_by_prompt_nature")  # Corrected patch path
    @patch(
        "utils.subject_handler.get_subject_config"
    )  # Corrected patch path (for fallback)
    @patch("utils.subject_handler.initialize_pipeline")  # Corrected patch path
    @patch("utils.subject_handler.logging")  # Corrected patch path
    def test_prompt_nature_fails_subject_fallback_success(
        self,
        mock_logging,
        mock_initialize_pipeline,
        mock_get_subject_config_fallback,
        mock_get_config_by_prompt_nature,
        mock_st,
    ):
        mock_st.session_state = self.mock_session_state.copy()
        mock_get_config_by_prompt_nature.return_value = (
            None  # Simulate failure to get nature-based config
        )

        subject_for_fallback = "mathematics"
        # Ensure this fallback config is different from current session state to trigger update
        fallback_subject_config = SubjectConfig(
            chunk_size=333, chunk_overlap=33, top_k=3, hybrid_alpha=0.33
        )
        mock_get_subject_config_fallback.return_value = fallback_subject_config

        mock_pipeline_instance = MagicMock()
        mock_initialize_pipeline.return_value = mock_pipeline_instance

        query = "Query where nature fails, but subject saves the day"
        result = update_rag_configuration(
            query=query,
            pipeline=self.mock_session_state["pipeline"],
            subject=subject_for_fallback,
        )

        self.assertTrue(result)
        mock_get_config_by_prompt_nature.assert_called_once_with(query)
        mock_get_subject_config_fallback.assert_called_once_with(subject_for_fallback)

        # Check session state updates based on fallback_subject_config
        self.assertEqual(
            mock_st.session_state["chunk_size"], fallback_subject_config.chunk_size
        )
        self.assertEqual(
            mock_st.session_state["chunk_overlap"],
            fallback_subject_config.chunk_overlap,
        )

        mock_initialize_pipeline.assert_called_once()
        self.assertEqual(mock_st.session_state["pipeline"], mock_pipeline_instance)
        mock_logging.info.assert_any_call(
            f"Falling back to subject-based configuration for: {subject_for_fallback}"
        )
        mock_logging.info.assert_any_call(
            f"Updating RAG parameters based on query nature. New config: ChunkSize={fallback_subject_config.chunk_size}, Overlap={fallback_subject_config.chunk_overlap}, TopK={fallback_subject_config.top_k}, Alpha={fallback_subject_config.hybrid_alpha} for query: '{query[:50]}...'"
        )

    @patch("utils.subject_handler.st")  # Corrected patch path
    @patch("utils.subject_handler.get_config_by_prompt_nature")  # Corrected patch path
    @patch("utils.subject_handler.initialize_pipeline")  # Corrected patch path
    @patch("utils.subject_handler.logging")  # Corrected patch path
    def test_file_path_missing_in_session_state(
        self,
        mock_logging,
        mock_initialize_pipeline,
        mock_get_config_by_prompt_nature,
        mock_st,
    ):
        mock_st.session_state = self.mock_session_state.copy()
        del mock_st.session_state["file_path"]  # Remove file_path

        new_config = SubjectConfig(
            chunk_size=100, chunk_overlap=10, top_k=1, hybrid_alpha=0.1
        )
        mock_get_config_by_prompt_nature.return_value = new_config

        query = "A query that triggers update"
        result = update_rag_configuration(
            query=query, pipeline=self.mock_session_state["pipeline"]
        )

        self.assertFalse(
            result
        )  # Should fail if file_path is required for re-initialization
        mock_initialize_pipeline.assert_not_called()  # Crucially, should not attempt to init without path
        mock_logging.error.assert_any_call(
            "Cannot reinitialize pipeline: File path is missing in session state."
        )

    @patch("utils.subject_handler.st")  # Corrected patch path
    @patch("utils.subject_handler.get_config_by_prompt_nature")  # Corrected patch path
    @patch("utils.subject_handler.initialize_pipeline")  # Corrected patch path
    @patch("utils.subject_handler.logging")  # Corrected patch path
    def test_enum_conversion_failure(
        self,
        mock_logging,
        mock_initialize_pipeline,
        mock_get_config_by_prompt_nature,
        mock_st,
    ):
        mock_st.session_state = self.mock_session_state.copy()
        mock_st.session_state["embedding_model"] = (
            "INVALID_MODEL_NAME"  # Cause EmbeddingModelType.from_string to fail
        )

        new_config = SubjectConfig(
            chunk_size=100, chunk_overlap=10, top_k=1, hybrid_alpha=0.1
        )
        mock_get_config_by_prompt_nature.return_value = new_config

        query = "Query triggering update with bad enum name"
        result = update_rag_configuration(
            query=query, pipeline=self.mock_session_state["pipeline"]
        )

        self.assertFalse(result)
        mock_initialize_pipeline.assert_not_called()
        # The error log will come from EmbeddingModelType.from_string, then caught in update_rag_configuration
        mock_logging.error.assert_any_call(
            unittest.mock.ANY
        )  # Check that some error was logged. Specific message is tricky.
        # A more specific check could be:
        # self.assertTrue(any("Failed to convert model string to enum" in call_args[0][0] for call_args in mock_logging.error.call_args_list))


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
    # This will run all Test classes in this file
