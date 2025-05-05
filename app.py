import streamlit as st
import os
import tempfile
import io
import re
import time
import datetime
import base64
import logging
import itertools
from typing import List, Dict, Any, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import httpx # Import httpx for OpenAI client timeout
# Note: gTTS is not strictly needed anymore if only using OpenAI TTS, but kept for reference
# from gtts import gTTS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Enums and Custom Modules ---
# (Assuming these files: enums.py, embedding_models.py, rerankers.py, vector_stores.py,
# llm_models.py, evaluator.py, rag_pipeline.py exist in the same directory)
try:
    from enums import (
        EmbeddingModelType, RerankerModelType, LLMModelType, VectorStoreType,
        ChunkingStrategyType, EvaluationBackendType, EvaluationMetricType
    )
    from embedding_models import EmbeddingModelFactory
    from rerankers import RerankerFactory
    from vector_stores import VectorStoreFactory
    from llm_models import LLMFactory
    from evaluator import EvaluatorFactory
    from rag_pipeline import RAGPipeline, ChunkingStrategyFactory
except ImportError as e:
    st.error(f"Failed to import custom modules: {e}. Make sure all required .py files are present.")
    logging.error(f"ImportError: {e}", exc_info=True)
    st.stop()


# Load environment variables from .env file
load_dotenv()

st.set_page_config(page_title="Ask JEFF - Study Buddy", layout="wide")

# --- Default RAG Configuration ---
DEFAULT_EMBEDDING_MODEL = EmbeddingModelType.MISTRAL
DEFAULT_RERANKER_MODEL = RerankerModelType.COHERE_V3
DEFAULT_LLM_MODEL = LLMModelType.CLAUDE_37_SONNET
DEFAULT_VECTOR_STORE = VectorStoreType.CHROMA
DEFAULT_CHUNKING_STRATEGY = ChunkingStrategyType.HIERARCHICAL
DEFAULT_CHUNK_SIZE = 2095
DEFAULT_CHUNK_OVERLAP = 195
DEFAULT_TOP_K = 4
DEFAULT_HYBRID_ALPHA = 0.5 # Default even if not always used

IsInEvaluationMode = False
# --- Initialize Session State ---
# Make sure Optional is here
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'file_path' not in st.session_state:
    st.session_state.file_path = None
if 'last_uploaded_filename' not in st.session_state:
    st.session_state.last_uploaded_filename = None
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'mode' not in st.session_state:
    st.session_state.mode = "chat"  # Default to chat mode
if 'permutation_results' not in st.session_state:
    st.session_state.permutation_results = None
if 'permutation_df' not in st.session_state:
    st.session_state.permutation_df = None
if 'show_contexts' not in st.session_state:
    st.session_state.show_contexts = False
if 'api_key_status' not in st.session_state:
    st.session_state.api_key_status = {}
if 'auto_init_attempted' not in st.session_state:
     st.session_state.auto_init_attempted = False

# Initialize RAG config defaults in session state if they don't exist
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = DEFAULT_EMBEDDING_MODEL.value
if 'reranker' not in st.session_state:
    st.session_state.reranker = DEFAULT_RERANKER_MODEL.value
if 'llm_model' not in st.session_state:
    st.session_state.llm_model = DEFAULT_LLM_MODEL.value
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = DEFAULT_VECTOR_STORE.value
if 'chunking_strategy' not in st.session_state:
    st.session_state.chunking_strategy = DEFAULT_CHUNKING_STRATEGY.value
if 'chunk_size' not in st.session_state:
    st.session_state.chunk_size = DEFAULT_CHUNK_SIZE
if 'chunk_overlap' not in st.session_state:
    st.session_state.chunk_overlap = DEFAULT_CHUNK_OVERLAP
if 'top_k' not in st.session_state:
    st.session_state.top_k = DEFAULT_TOP_K
if 'hybrid_alpha' not in st.session_state:
    st.session_state.hybrid_alpha = DEFAULT_HYBRID_ALPHA


# --- Helper Functions ---

def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary location and return the path"""
    try:
        file_suffix = os.path.splitext(uploaded_file.name)[1] if '.' in uploaded_file.name else '.txt'
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp:
            temp.write(uploaded_file.getvalue())
            logging.info(f"Saved uploaded file '{uploaded_file.name}' to temporary path: {temp.name}")
            return temp.name
    except Exception as e:
        logging.error(f"Error saving uploaded file: {e}")
        st.error(f"Error saving file: {e}")
        return None

def get_csv_download_link(df, filename="permutation_results.csv"):
    """Generate a download link for a pandas dataframe as CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Results as CSV</a>'
    return href

def check_api_keys(embedding_model_enum, vector_store_enum, reranker_enum, llm_enum):
    """Check if required API keys are available in environment"""
    api_keys_status = {}
    missing_keys_list = []

    # Determine required keys based on selections
    openai_needed = (embedding_model_enum == EmbeddingModelType.OPENAI or
                     llm_enum in [LLMModelType.OPENAI_GPT35, LLMModelType.OPENAI_GPT4] or
                     True) # OpenAI TTS always needs it
    cohere_needed = (embedding_model_enum == EmbeddingModelType.COHERE or
                     reranker_enum in [RerankerModelType.COHERE_V2, RerankerModelType.COHERE_V3, RerankerModelType.COHERE_MULTILINGUAL])
    gemini_needed = (embedding_model_enum == EmbeddingModelType.GEMINI or
                     llm_enum == LLMModelType.GEMINI)
    anthropic_needed = (llm_enum in [LLMModelType.CLAUDE_3_OPUS, LLMModelType.CLAUDE_37_SONNET])
    mistral_needed = (embedding_model_enum == EmbeddingModelType.MISTRAL or
                      llm_enum in [LLMModelType.MISTRAL_LARGE, LLMModelType.MISTRAL_MEDIUM, LLMModelType.MISTRAL_SMALL])
    voyage_needed = (embedding_model_enum == EmbeddingModelType.VOYAGE or
                     reranker_enum in [RerankerModelType.VOYAGE, RerankerModelType.VOYAGE_2])

    # Check and record status
    if openai_needed:
        key_name = "OpenAI API Key"
        is_available = bool(os.getenv("OPENAI_API_KEY"))
        api_keys_status[key_name] = "Available" if is_available else "Missing"
        if not is_available: missing_keys_list.append(key_name)

    if cohere_needed:
        key_name = "Cohere API Key"
        is_available = bool(os.getenv("COHERE_API_KEY"))
        api_keys_status[key_name] = "Available" if is_available else "Missing"
        if not is_available: missing_keys_list.append(key_name)

    if gemini_needed:
        key_name = "Gemini API Key"
        is_available = bool(os.getenv("GOOGLE_API_KEY")) # Often GOOGLE_API_KEY for Gemini
        api_keys_status[key_name] = "Available" if is_available else "Missing"
        if not is_available: missing_keys_list.append(key_name)

    if anthropic_needed:
        key_name = "Anthropic API Key"
        is_available = bool(os.getenv("ANTHROPIC_API_KEY"))
        api_keys_status[key_name] = "Available" if is_available else "Missing"
        if not is_available: missing_keys_list.append(key_name)

    if mistral_needed:
        key_name = "Mistral API Key"
        is_available = bool(os.getenv("MISTRAL_API_KEY"))
        api_keys_status[key_name] = "Available" if is_available else "Missing"
        if not is_available: missing_keys_list.append(key_name)

    if voyage_needed:
        key_name = "Voyage AI API Key"
        is_available = bool(os.getenv("VOYAGE_API_KEY"))
        api_keys_status[key_name] = "Available" if is_available else "Missing"
        if not is_available: missing_keys_list.append(key_name)

    st.session_state.api_key_status = api_keys_status
    return missing_keys_list

def initialize_pipeline(file_path, embedding_model_enum, vector_store_enum, reranker_enum, llm_enum,
                        chunking_strategy_enum, hybrid_alpha, chunk_size, chunk_overlap, top_k):
    """Initialize RAG pipeline with selected configuration"""
    logging.info(f"Attempting to initialize RAG pipeline with config:")
    logging.info(f"  Embedding: {embedding_model_enum.value}, Vector Store: {vector_store_enum.value}, Reranker: {reranker_enum.value}, LLM: {llm_enum.value}")
    logging.info(f"  Chunking: {chunking_strategy_enum.value}, Size: {chunk_size}, Overlap: {chunk_overlap}, Top K: {top_k}, Hybrid Alpha: {hybrid_alpha}")

    if not file_path or not os.path.exists(file_path):
        st.error("Cannot initialize pipeline: Document file path is invalid or missing.")
        logging.error("Pipeline initialization failed: Invalid file path.")
        return None

    try:
        # Initialize components
        embedding_model_instance = EmbeddingModelFactory.create_model(embedding_model_enum)

        if vector_store_enum == VectorStoreType.HYBRID:
            vector_store_instance = VectorStoreFactory.create_store(vector_store_enum, alpha=hybrid_alpha)
        else:
            vector_store_instance = VectorStoreFactory.create_store(vector_store_enum)

        reranker_instance = None
        if reranker_enum != RerankerModelType.NONE:
            reranker_instance = RerankerFactory.create_reranker(reranker_enum)

        llm_instance = LLMFactory.create_llm(llm_enum) # Assuming LLMFactory handles persona if needed, or RAGPipeline does
        chunking_strategy_instance = ChunkingStrategyFactory.get_strategy(chunking_strategy_enum.value)

        IsInEvaluationMode = False
        if st.session_state.mode == "evaluation":
            IsInEvaluationMode = True

        # Create RAG pipeline
        pipeline = RAGPipeline(
            embedding_model=embedding_model_instance,
            vector_store=vector_store_instance,
            reranker=reranker_instance,
            llm=llm_instance,
            top_k=top_k,
            chunking_strategy=chunking_strategy_instance,
            evaluation_mode=IsInEvaluationMode 
        )

        # Index documents
        logging.info(f"Indexing documents from: {file_path}")
        index_start_time = time.time()
        try:
             pipeline.index_documents(file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        except Exception as index_e:
             logging.error(f"Error during document indexing: {index_e}", exc_info=True)
             st.error(f"Failed to index document: {index_e}")
             return None
        index_end_time = time.time()
        logging.info(f"Document indexing completed in {index_end_time - index_start_time:.2f} seconds.")

        st.session_state.pipeline = pipeline
        return pipeline

    except Exception as e:
        logging.error(f"Error initializing RAG pipeline: {e}", exc_info=True)
        st.error(f"Failed to initialize RAG pipeline: {e}")
        st.session_state.pipeline = None
        return None

def run_pipeline_with_config(
    file_path: str,
    user_query: str,
    ground_truth: str,
    embedding_model_enum: EmbeddingModelType,
    vector_store_enum: VectorStoreType,
    reranker_enum: RerankerModelType,
    llm_enum: LLMModelType,
    chunking_strategy_enum: ChunkingStrategyType,
    hybrid_alpha: float = 0.5,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    top_k: int = 3
) -> Dict[str, Any]:
    """Run a single pipeline configuration and return results"""
    config_str = f"{embedding_model_enum.value}, {vector_store_enum.value}, {reranker_enum.value}, {llm_enum.value}, {chunking_strategy_enum.value}"
    logging.info(f"Running pipeline with config: {config_str}")
    start_run_time = time.time()
    try:
        # Initialize components
        embedding_model_instance = EmbeddingModelFactory.create_model(embedding_model_enum)
        vector_store_instance = VectorStoreFactory.create_store(vector_store_enum, alpha=hybrid_alpha) if vector_store_enum == VectorStoreType.HYBRID else VectorStoreFactory.create_store(vector_store_enum)
        reranker_instance = RerankerFactory.create_reranker(reranker_enum) if reranker_enum != RerankerModelType.NONE else None
        llm_instance = LLMFactory.create_llm(llm_enum)
        chunking_strategy_instance = ChunkingStrategyFactory.get_strategy(chunking_strategy_enum.value)

        pipeline = RAGPipeline(
            embedding_model=embedding_model_instance,
            vector_store=vector_store_instance,
            reranker=reranker_instance,
            llm=llm_instance,
            top_k=top_k,
            chunking_strategy=chunking_strategy_instance,
            evaluation_mode=IsInEvaluationMode 
        )

        # Indexing (re-index per config for isolation in eval)
        pipeline.index_documents(file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Process query
        start_query_time = time.time()
        response, contexts = pipeline.process_query(user_query)
        query_elapsed_time = time.time() - start_query_time
        logging.info(f"Query processed in {query_elapsed_time:.2f}s. Response length: {len(response)}")

        # Run evaluation
        evaluation_results = {}
        avg_score = 0
        if ground_truth:
            try:
                evaluator = EvaluatorFactory.create_evaluator(
                    EvaluationBackendType.RAGAS,
                    EvaluationMetricType.get_metrics_for_backend(EvaluationBackendType.RAGAS)
                )
                evaluation_results = evaluator.evaluate(
                    query=user_query, response=response, contexts=contexts, ground_truth=ground_truth
                )
                if evaluation_results and isinstance(evaluation_results, dict):
                    valid_scores = [v for v in evaluation_results.values() if isinstance(v, (int, float))]
                    if valid_scores:
                        avg_score = sum(valid_scores) / len(valid_scores)
                logging.info(f"Evaluation scores: {evaluation_results}")
            except Exception as eval_e:
                logging.error(f"Error during evaluation for config {config_str}: {eval_e}", exc_info=True)
                st.warning(f"Evaluation failed for this configuration: {eval_e}")
                evaluation_results = {"error": str(eval_e)}
        else:
             logging.warning("No ground truth provided, skipping RAGAS evaluation.")

        total_elapsed_time = time.time() - start_run_time
        logging.info(f"Total run time for config {config_str}: {total_elapsed_time:.2f}s")

        return {
            "embedding_model": embedding_model_enum.value, "vector_store": vector_store_enum.value,
            "reranker": reranker_enum.value, "llm_model": llm_enum.value,
            "chunking_strategy": chunking_strategy_enum.value, "response": response,
            "evaluation_scores": evaluation_results, "avg_score": avg_score,
            "elapsed_time": total_elapsed_time, "contexts": contexts
        }
    except Exception as e:
        total_elapsed_time = time.time() - start_run_time
        logging.error(f"Error running pipeline config {config_str}: {e}", exc_info=True)
        st.error(f"Error with configuration - {config_str}: {str(e)}")
        return {
            "embedding_model": embedding_model_enum.value, "vector_store": vector_store_enum.value,
            "reranker": reranker_enum.value, "llm_model": llm_enum.value,
            "chunking_strategy": chunking_strategy_enum.value, "response": f"ERROR: {str(e)}",
            "evaluation_scores": {"error": str(e)}, "avg_score": 0,
            "elapsed_time": total_elapsed_time, "contexts": []
        }

def run_all_permutations(
    file_path: str,
    user_query: str,
    ground_truth: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    hybrid_alpha: float,
    chunking_strategy_enum: ChunkingStrategyType
):
    """Run all permutations of models and return results as a dataframe"""
    logging.info("Starting 'Run All Permutations'")
    # Define the models to include in permutations
    embedding_models = [
        EmbeddingModelType.OPENAI, EmbeddingModelType.GEMINI, EmbeddingModelType.MISTRAL
    ]
    vector_stores = [
        VectorStoreType.FAISS, VectorStoreType.CHROMA
    ]
    rerankers = list(RerankerModelType) # Includes NONE
    llm_models = [
        LLMModelType.CLAUDE_37_SONNET, LLMModelType.GEMINI
    ]

    all_permutations = list(itertools.product(embedding_models, vector_stores, rerankers, llm_models))
    num_permutations = len(all_permutations)
    logging.info(f"Total permutations to run: {num_permutations}")

    progress_bar = st.progress(0, text="Starting permutations...")
    results = []
    start_permutations_time = time.time()

    for i, (embedding_model, vector_store, reranker, llm_model) in enumerate(all_permutations):
        current_config_str = f"{embedding_model.value}, {vector_store.value}, {reranker.value}, {llm_model.value}"
        logging.info(f"Running permutation {i+1}/{num_permutations}: {current_config_str}")
        progress_text = f"Running permutation {i+1}/{num_permutations}: {current_config_str}"
        try:
            progress_bar.progress((i + 1) / num_permutations, text=progress_text)
        except Exception as pb_e:
            logging.warning(f"Could not update progress bar: {pb_e}")

        missing_keys = check_api_keys(embedding_model, vector_store, reranker, llm_model)
        if missing_keys:
            st.warning(f"Skipping permutation {current_config_str} due to missing keys: {', '.join(missing_keys)}")
            result = {
                "embedding_model": embedding_model.value, "vector_store": vector_store.value,
                "reranker": reranker.value, "llm_model": llm_model.value,
                "chunking_strategy": chunking_strategy_enum.value, "response": "SKIPPED - Missing API Keys",
                "evaluation_scores": {}, "avg_score": 0, "elapsed_time": 0, "contexts": []
            }
        else:
             result = run_pipeline_with_config(
                 file_path=file_path, user_query=user_query, ground_truth=ground_truth,
                 embedding_model_enum=embedding_model, vector_store_enum=vector_store,
                 reranker_enum=reranker, llm_enum=llm_model,
                 chunking_strategy_enum=chunking_strategy_enum, hybrid_alpha=hybrid_alpha,
                 chunk_size=chunk_size, chunk_overlap=chunk_overlap, top_k=top_k
             )

        if "evaluation_scores" in result and isinstance(result["evaluation_scores"], dict):
             for metric, score in result["evaluation_scores"].items():
                 if isinstance(score, (int, float)):
                     result[f"metric_{metric}"] = score

        results.append(result)

    end_permutations_time = time.time()
    total_time = end_permutations_time - start_permutations_time
    logging.info(f"All {num_permutations} permutations completed in {total_time:.2f} seconds.")
    try:
        progress_bar.progress(1.0, text="Permutations complete!")
        time.sleep(1)
        progress_bar.empty()
    except Exception as pb_e:
        logging.warning(f"Could not update/empty progress bar: {pb_e}")

    # Convert to dataframe
    results_df = pd.DataFrame(results)

    # Define columns for CSV export
    base_columns = ["embedding_model", "vector_store", "reranker", "llm_model", "chunking_strategy", "avg_score", "elapsed_time"]
    metric_columns = sorted([col for col in results_df.columns if col.startswith("metric_")])
    csv_columns = base_columns + metric_columns + ["response"]

    # Ensure all expected columns exist
    for col in csv_columns:
        if col not in results_df.columns:
            results_df[col] = pd.NA

    # Prepare numeric columns
    results_df['avg_score'] = pd.to_numeric(results_df['avg_score'], errors='coerce')
    results_df['elapsed_time'] = pd.to_numeric(results_df['elapsed_time'], errors='coerce')
    for col in metric_columns:
        results_df[col] = pd.to_numeric(results_df[col], errors='coerce')

    display_df = results_df[csv_columns].copy()
    return display_df, results


# --- TTS Helper Function ---
@st.cache_data(show_spinner=False) # Cache TTS results
def text_to_speech(text: str) -> Optional[bytes]:
    """Generates speech from text using OpenAI TTS and returns audio bytes."""
    if not text or not isinstance(text, str):
        logging.warning("TTS skipped: Input text is empty or not a string.")
        return None

    # Clean the text
    cleaned_text = re.sub(r'[#*]', '', text) # Remove markdown emphasis
    cleaned_text = re.sub(r'http[s]?://\S+', '', cleaned_text) # Remove URLs
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip() # Normalize whitespace

    if not cleaned_text:
        logging.warning("TTS skipped: Text is empty after cleaning.")
        return None

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        logging.error("OpenAI API Key not found. Cannot generate audio.")
        return None

    try:
        # Use httpx timeout for potentially longer requests
        client = OpenAI(timeout=httpx.Timeout(45.0, connect=10.0)) # Increased timeout
        selected_voice = "fable"
        selected_model = "tts-1"

        logging.info(f"Requesting OpenAI TTS: voice='{selected_voice}', model='{selected_model}', text length (cleaned): {len(cleaned_text)}")

        response = client.audio.speech.create(
            model=selected_model,
            voice=selected_voice,
            input=cleaned_text,
            response_format="mp3"
        )

        audio_bytes = response.read()
        logging.info(f"OpenAI TTS audio generated successfully ({len(audio_bytes)} bytes).")
        return audio_bytes

    except ImportError:
        logging.error("OpenAI library not installed. Cannot generate audio.")
        st.warning("Audio generation requires `openai`. Please install it (`pip install openai`).", icon="⚠️")
        return None
    except Exception as e:
        logging.error(f"Error generating OpenAI TTS audio: {e}", exc_info=True)
        # Avoid flooding UI, maybe a single notification elsewhere if persistent
        # st.warning(f"Couldn't generate audio: {e}", icon="🔇")
        return None


# --- UI Display Functions ---

def display_chat_interface():
    st.header("💬 Chat with JEFF")
    st.markdown("Hey! Got questions about your textbook? Lay 'em on me. I'll break it down for ya.")

    # Initialize welcome message if chat is empty
    if not st.session_state.messages:
         welcome_msg = "Alright, let's get this study session started! What's on your mind?"
         welcome_audio_bytes = text_to_speech(welcome_msg) # Generate audio
         st.session_state.messages.append({
             "role": "assistant",
             "content": welcome_msg,
             "audio": welcome_audio_bytes, # Store audio bytes
             "contexts": [],
             "elapsed_time": None
         })

    # Display message history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                # Safely get message data
                response_text = message.get("content")
                audio_data = message.get("audio") # bytes or None
                contexts = message.get("contexts", [])
                elapsed_time = message.get("elapsed_time")

                # Use tabs if there's content
                if response_text:
                    tab_labels = ["📖 Read Response", "🔊 Hear Response"]
                    try:
                        tab_text, tab_audio = st.tabs(tab_labels)
                    except Exception as e:
                        # Fallback if tabs fail (e.g., during complex reruns)
                        logging.error(f"Error creating tabs: {e}")
                        st.write(response_text) # Just display text
                        if audio_data: st.audio(audio_data, format="audio/mp3")
                        tab_text = None # Prevent errors in 'with' blocks below

                    if tab_text: # Check if tabs were created successfully
                        with tab_text:
                            st.write(response_text)

                        with tab_audio:
                            if audio_data:
                                st.audio(audio_data, format="audio/mp3")
                            else:
                                st.info("Audio playback is not available for this message.")

                    # Display metadata *below* the tabs
                    if elapsed_time is not None:
                         st.write(f"_(JEFF cooked that up in {elapsed_time:.2f} seconds)_")

                    if st.session_state.show_contexts and contexts:
                         with st.expander("🧠 Check out the textbook bits I used:"):
                            for i, context in enumerate(contexts):
                                st.markdown(f"**Snippet {i+1}:**")
                                st.text(context)

                else: # Fallback if assistant message has no content
                    st.write("*Assistant message content missing.*")

            else: # User message
                st.write(message["content"])

    # User input
    user_query = st.chat_input("Type your question here...")

    if user_query:
        logging.info(f"User query received: {user_query}")
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        # Check pipeline initialization
        if st.session_state.pipeline is None:
            logging.warning("Chat query received, but pipeline not initialized.")
            warning_msg = "Whoa there! Looks like we haven't loaded your textbook into my brain yet. Upload it and hit 'Initialize' in the sidebar first!"
            warning_audio = text_to_speech(warning_msg)
            with st.chat_message("assistant"):
                 tab_labels_warn = ["📖 Read Message", "🔊 Hear Message"]
                 tab_warn_text, tab_warn_audio = st.tabs(tab_labels_warn)
                 with tab_warn_text: st.warning(warning_msg, icon="✋")
                 with tab_warn_audio:
                     if warning_audio: st.audio(warning_audio, format="audio/mp3")
                     else: st.info("Audio playback not available.")

            st.session_state.messages.append({
                "role": "assistant", "content": warning_msg, "audio": warning_audio,
                "contexts": [], "elapsed_time": None
            })
            st.stop()

        # Process query
        with st.spinner("JEFF's thinking... 🤔"):
            try:
                start_time = time.time()
                logging.info("Processing query with RAG pipeline...")
                response, contexts = st.session_state.pipeline.process_query(user_query)
                elapsed_time = time.time() - start_time
                logging.info(f"Query processed successfully in {elapsed_time:.2f}s.")

                # Generate TTS Audio
                logging.info("Generating TTS audio for the response...")
                tts_start_time = time.time()
                audio_bytes = text_to_speech(response) # bytes or None
                tts_elapsed_time = time.time() - tts_start_time
                log_msg = f"TTS generation {'succeeded' if audio_bytes else 'failed/skipped'} in {tts_elapsed_time:.2f}s."
                if audio_bytes: logging.info(log_msg)
                else: logging.warning(log_msg)


                # Display assistant response using tabs
                with st.chat_message("assistant"):
                    tab_labels_resp = ["📖 Read Response", "🔊 Hear Response"]
                    tab_resp_text, tab_resp_audio = st.tabs(tab_labels_resp)

                    with tab_resp_text: st.write(response)
                    with tab_resp_audio:
                        if audio_bytes: st.audio(audio_bytes, format="audio/mp3")
                        else: st.info("Audio playback is not available for this message.")

                    # Display metadata below tabs
                    st.write(f"_(JEFF cooked that up in {elapsed_time:.2f} seconds)_")
                    if st.session_state.show_contexts and contexts:
                         with st.expander("🧠 Check out the textbook bits I used:"):
                            for i, context in enumerate(contexts):
                                st.markdown(f"**Snippet {i+1}:**")
                                st.text(context)

                # Add assistant message to history
                st.session_state.messages.append({
                    "role": "assistant", "content": response, "contexts": contexts,
                    "elapsed_time": elapsed_time, "audio": audio_bytes
                })
                st.rerun() # Clear input box and update history display

            except Exception as e:
                logging.error(f"Error processing query or generating audio: {e}", exc_info=True)
                error_msg = f"Oof, hit a snag trying to answer that. Maybe try rephrasing? Error: {str(e)}"
                error_audio = text_to_speech(error_msg)

                # Display error using tabs
                with st.chat_message("assistant"):
                    tab_labels_err = ["📖 Read Error", "🔊 Hear Error"]
                    tab_err_text, tab_err_audio = st.tabs(tab_labels_err)
                    with tab_err_text: st.error(error_msg, icon="🔥")
                    with tab_err_audio:
                        if error_audio: st.audio(error_audio, format="audio/mp3")
                        else: st.info("Audio playback not available.")

                # Add error message to history
                st.session_state.messages.append({
                    "role": "assistant", "content": error_msg, "audio": error_audio,
                    "contexts": [], "elapsed_time": None
                })
                # No rerun needed after error usually


def display_evaluation_interface():

    st.header("🧪 RAG Evaluation Mode")
    st.markdown("Let's test out different setups. Give me a question and the perfect answer (ground truth) to see how well various RAG configurations perform.")

    # Status checks
    if st.session_state.pipeline is None and st.session_state.file_path:
        st.warning("💡 Pipeline isn't active. Hit 'Initialize JEFF' in the sidebar.")
    elif not st.session_state.file_path:
         st.warning("💡 Upload a document first using the sidebar!")

    col1, col2 = st.columns([1, 1]) # Input | Results

    with col1:
        st.subheader("Evaluation Inputs")
        user_query = st.text_area("Enter your question:", height=100, key="eval_query")
        ground_truth = st.text_area("Enter the ideal 'ground truth' answer:", height=100, key="eval_ground_truth")
        st.info("Providing ground truth enables detailed RAGAS evaluation scores.")

        # Buttons
        disable_eval_buttons = st.session_state.pipeline is None or not st.session_state.file_path
        process_button = st.button("Evaluate Current Config", disabled=disable_eval_buttons)
        permutation_button = st.button("Run All Permutations Test", disabled=disable_eval_buttons)

    # --- Process Single Config ---
    if process_button and user_query:
        if not st.session_state.file_path: st.warning("Please upload a document first.")
        elif st.session_state.pipeline is None: st.warning("Pipeline not initialized. Please initialize.")
        else:
            try: # Read current config from state
                 embedding_model_enum = EmbeddingModelType.from_string(st.session_state.embedding_model)
                 vector_store_enum = VectorStoreType.from_string(st.session_state.vector_store)
                 reranker_enum = RerankerModelType.from_string(st.session_state.reranker)
                 llm_enum = LLMModelType.from_string(st.session_state.llm_model)
                 chunking_strategy_enum = ChunkingStrategyType.from_string(st.session_state.chunking_strategy)
            except ValueError as e:
                 st.error(f"Error reading current configuration: {e}"); st.stop()

            missing_keys = check_api_keys(embedding_model_enum, vector_store_enum, reranker_enum, llm_enum)
            if missing_keys: st.error(f"Missing keys for current config: {', '.join(missing_keys)}")
            else:
                with st.spinner("Evaluating current configuration..."):
                    logging.info("Evaluating single query using the existing pipeline.")
                    start_eval_time = time.time()
                    try: # Process with existing pipeline
                        response, contexts = st.session_state.pipeline.process_query(user_query)
                        eval_elapsed_time = time.time() - start_eval_time

                        # Run evaluation metrics
                        evaluation_results = {}
                        avg_score = 0
                        valid_scores = []
                        if ground_truth:
                             try:
                                evaluator = EvaluatorFactory.create_evaluator(EvaluationBackendType.RAGAS, EvaluationMetricType.get_metrics_for_backend(EvaluationBackendType.RAGAS))
                                evaluation_results = evaluator.evaluate(query=user_query, response=response, contexts=contexts, ground_truth=ground_truth)
                                if evaluation_results and isinstance(evaluation_results, dict):
                                     valid_scores = [v for v in evaluation_results.values() if isinstance(v, (int, float))]
                                     if valid_scores: avg_score = sum(valid_scores) / len(valid_scores)
                                logging.info(f"Single config evaluation scores: {evaluation_results}")
                             except Exception as eval_e:
                                logging.error(f"Evaluation failed for single config run: {eval_e}", exc_info=True)
                                st.warning(f"Evaluation metrics failed: {eval_e}")
                                evaluation_results = {"error": str(eval_e)}
                        else: logging.warning("No ground truth provided, skipping evaluation metrics.")

                        # --- Display Single Result in Col 2 ---
                        st.session_state.permutation_df = None # Clear any old permutation results
                        st.session_state.permutation_results = None
                        with col2:
                            st.subheader("Evaluation Result (Current Config)")
                            st.markdown(f"**Configuration:** `{st.session_state.embedding_model} | {st.session_state.vector_store} | {st.session_state.reranker} | {st.session_state.llm_model} | {st.session_state.chunking_strategy}`")
                            st.write(f"**Processing Time:** {eval_elapsed_time:.2f} seconds")
                            with st.expander("Response", expanded=True): st.write(response)
                            with st.expander("Retrieved Contexts", expanded=False):
                                 if contexts:
                                    for i, ctx in enumerate(contexts): st.markdown(f"**Context {i+1}:**"); st.text(ctx)
                                 else: st.write("No contexts were retrieved.")

                            # Display scores
                            if evaluation_results and isinstance(evaluation_results, dict) and "error" not in evaluation_results:
                                 st.subheader("Evaluation Scores")
                                 if evaluation_results:
                                     metric_cols = st.columns(len(evaluation_results))
                                     i = 0
                                     for metric, score in evaluation_results.items():
                                         with metric_cols[i]:
                                             score_display = f"{score:.2f}" if isinstance(score, (int, float)) else "N/A"
                                             st.metric(label=metric.replace('_', ' ').title(), value=score_display)
                                         i += 1
                                     if avg_score > 0 or len(valid_scores) > 0:
                                         st.metric("Overall Average Score", f"{avg_score:.2f}")
                                 else: st.info("No scores generated.")
                            elif "error" in evaluation_results: st.warning(f"Scores not calculated: {evaluation_results['error']}")
                            elif ground_truth: st.warning("Scores could not be calculated.")
                            else: st.info("Provide ground truth to see scores.")

                    except Exception as e:
                        logging.error(f"Error running single evaluation: {e}", exc_info=True)
                        st.error(f"Error processing evaluation: {str(e)}")
                        with col2: st.error(f"Failed to evaluate: {e}")

    # --- Process Permutations ---
    elif permutation_button and user_query:
        if not st.session_state.file_path: st.warning("Please upload a document first.")
        elif st.session_state.pipeline is None: st.warning("Pipeline not initialized. Please initialize.")
        else:
            # Check potential keys needed for permutations
            st.info("Checking API keys potentially needed for permutations...")
            perm_emb = [EmbeddingModelType.OPENAI, EmbeddingModelType.GEMINI, EmbeddingModelType.MISTRAL]
            perm_rerank = list(RerankerModelType)
            perm_llm = [LLMModelType.CLAUDE_37_SONNET, LLMModelType.GEMINI]
            potential_missing = set()
            for e_perm, r_perm, l_perm in itertools.product(perm_emb, perm_rerank, perm_llm):
                potential_missing.update(check_api_keys(e_perm, VectorStoreType.FAISS, r_perm, l_perm)) # Use dummy VS
            if potential_missing: st.warning(f"Heads up! Missing potential keys: {', '.join(potential_missing)}. Some permutations might fail.")
            else: st.success("Looks like all potentially required API keys are present!")

            try: chunking_strategy_enum = ChunkingStrategyType.from_string(st.session_state.chunking_strategy)
            except ValueError as e: st.error(f"Invalid chunking strategy: {e}"); st.stop()

            # Run permutations
            with st.spinner("Running all permutations... This might take a while! ☕️"):
                 results_df, all_results = run_all_permutations(
                    file_path=st.session_state.file_path, user_query=user_query, ground_truth=ground_truth,
                    chunk_size=st.session_state.chunk_size, chunk_overlap=st.session_state.chunk_overlap,
                    top_k=st.session_state.top_k, hybrid_alpha=st.session_state.hybrid_alpha,
                    chunking_strategy_enum=chunking_strategy_enum
                )
            st.session_state.permutation_df = results_df
            st.session_state.permutation_results = all_results
            logging.info("Permutations completed.")
            st.success("All permutations complete! Results below.")
            st.rerun() # Update display in col2

    elif (process_button or permutation_button) and not user_query:
        st.warning("⚠️ Please enter a question to evaluate.")

    # --- Display Permutation Results in Col 2 ---
    with col2:
        if st.session_state.permutation_df is not None and not st.session_state.permutation_df.empty:
            st.subheader("Permutation Results Summary")
            st.markdown(get_csv_download_link(st.session_state.permutation_df), unsafe_allow_html=True)

            # Show top results sorted by score
            results_to_display = st.session_state.permutation_df.copy()
            results_to_display['avg_score_numeric'] = results_to_display['avg_score'].fillna(-1)
            top_results = results_to_display.sort_values('avg_score_numeric', ascending=False).head(10)

            # Define display columns dynamically
            display_cols = ["embedding_model", "vector_store", "reranker", "llm_model", "chunking_strategy", "avg_score", "elapsed_time"]
            metric_cols_exist = sorted([col for col in results_to_display.columns if col.startswith("metric_")])
            display_cols.extend(metric_cols_exist)

            format_dict = {'avg_score': "{:.2f}", 'elapsed_time': "{:.2f}"}
            for col in metric_cols_exist: format_dict[col] = "{:.2f}"
            st.dataframe(top_results[display_cols].style.format(format_dict, na_rep="N/A"))

            # Explore individual results
            st.markdown("---")
            st.subheader("Explore Individual Results")
            config_labels = []
            if st.session_state.permutation_results:
                for index, row_data in enumerate(st.session_state.permutation_results):
                    label = (f"{index}: {row_data.get('embedding_model','?')} / {row_data.get('vector_store','?')} / "
                             f"{row_data.get('reranker','?')} / {row_data.get('llm_model','?')} "
                             f"(Score: {row_data.get('avg_score', 0):.2f}, Time: {row_data.get('elapsed_time', 0):.1f}s)")
                    config_labels.append(label)

                if config_labels:
                     selected_index = st.selectbox(
                         "Select Configuration to View Details:", options=range(len(config_labels)),
                         format_func=lambda index: config_labels[index], key="permutation_select"
                     )
                     if selected_index is not None and selected_index < len(st.session_state.permutation_results):
                        selected_result = st.session_state.permutation_results[selected_index]
                        st.markdown(f"**Details for Configuration {selected_index}:**")
                        st.markdown(f"**Models:** `{selected_result.get('embedding_model','N/A')} | {selected_result.get('vector_store','N/A')} | {selected_result.get('reranker','N/A')} | {selected_result.get('llm_model','N/A')}`")
                        st.markdown(f"**Chunking:** `{selected_result.get('chunking_strategy','N/A')}`")
                        st.write(f"**Processing Time:** {selected_result.get('elapsed_time', 0):.2f} seconds")
                        with st.expander("Response", expanded=True): st.write(selected_result.get('response', 'N/A'))
                        with st.expander("Retrieved Contexts", expanded=False):
                             contexts = selected_result.get('contexts', [])
                             if contexts:
                                 for i, ctx in enumerate(contexts): st.markdown(f"**Context {i+1}:**"); st.text(ctx)
                             else: st.write("No contexts available.")

                        # Display scores for selected permutation
                        eval_scores = selected_result.get('evaluation_scores', {})
                        if isinstance(eval_scores, dict) and "error" not in eval_scores and eval_scores:
                            st.subheader("Evaluation Scores")
                            metric_cols_detail = st.columns(len(eval_scores))
                            i = 0; valid_scores_detail = []
                            for metric, score in eval_scores.items():
                                 with metric_cols_detail[i]:
                                     score_display = f"{score:.2f}" if isinstance(score, (int, float)) else "N/A"
                                     st.metric(label=metric.replace('_', ' ').title(), value=score_display)
                                     if isinstance(score, (int, float)): valid_scores_detail.append(score)
                                 i+=1
                            avg_score_detail = selected_result.get('avg_score', 0)
                            if avg_score_detail > 0 or len(valid_scores_detail) > 0:
                                 st.metric("Overall Average Score", f"{avg_score_detail:.2f}")
                        elif isinstance(eval_scores, dict) and "error" in eval_scores: st.warning(f"Eval failed: {eval_scores['error']}")
                        elif ground_truth: st.warning("Scores not available (failed/skipped?).")
                        else: st.info("Provide ground truth during permutation run for scores.")
                else: st.info("No permutation results available.")
        elif st.session_state.permutation_results is not None:
             st.info("Permutation run finished, but no valid results generated.")
        # else: st.info("Run permutations or evaluate single config to see results.")


def display_settings_panel():
    st.sidebar.image("https://www.nicepng.com/png/detail/972-9721863_raising-hand-icon-png.png", width=80)
    st.sidebar.title("JEFF's Controls")

    # Mode selector
    mode_options = {"💬 Chat with JEFF": "chat", "🧪 Test Setups (Evaluation)": "evaluation"}
    current_mode = st.session_state.get('mode', 'chat')
    if current_mode not in mode_options.values(): current_mode = 'chat'; st.session_state.mode = 'chat'
    current_mode_index = list(mode_options.values()).index(current_mode)
    selected_mode_label = st.sidebar.radio(
        "Select Mode", options=list(mode_options.keys()), index=current_mode_index,
        key="mode_radio", help="Switch between chatting and testing configurations."
    )
    new_mode = mode_options[selected_mode_label]
    if new_mode != st.session_state.mode:
        st.session_state.mode = new_mode
        st.session_state.messages = [] # Reset chat
        st.session_state.pipeline = None # Reset pipeline
        st.session_state.permutation_results = None # Clear eval results
        st.session_state.permutation_df = None
        logging.info(f"Mode changed to: {st.session_state.mode}. Resetting state.")
        st.rerun()

    # File upload
    st.sidebar.header("📚 Load Textbook")
    uploaded_file = st.sidebar.file_uploader("Upload .txt file", type=['txt'], key="file_uploader")
    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.get('last_uploaded_filename', None):
            logging.info(f"New file upload detected: {uploaded_file.name}")
            with st.spinner(f"Processing '{uploaded_file.name}'..."):
                st.session_state.file_path = save_uploaded_file(uploaded_file)
            if st.session_state.file_path:
                 st.session_state.last_uploaded_filename = uploaded_file.name
                 st.session_state.pipeline = None # Reset pipeline on new file
                 st.session_state.messages = []
                 st.session_state.permutation_results = None; st.session_state.permutation_df = None
                 st.sidebar.success(f"'{uploaded_file.name}' loaded!")
                 logging.info("New file loaded, reset state. Rerunning.")
                 st.rerun()
            else:
                 st.sidebar.error("Failed to process uploaded file.")
                 st.session_state.file_path = None; st.session_state.last_uploaded_filename = None; st.session_state.pipeline = None

    # System status
    st.sidebar.header("🚦 System Status")
    with st.sidebar.container(border=True):
        if st.session_state.file_path and os.path.exists(st.session_state.file_path):
            st.success(f"✅ Textbook: {st.session_state.last_uploaded_filename}")
        else: st.warning("⚠️ No textbook loaded")
        if st.session_state.pipeline: st.success("✅ JEFF is ready!")
        else: st.warning("⏳ JEFF needs setup (Initialize)")

    # Configuration Options Expander
    st.sidebar.markdown("---")
    if st.session_state.mode == "evaluation":
        IsInEvaluationMode = True
        st.sidebar.header("🛠️ Evaluation Config")
        st.sidebar.info("Adjust settings for Evaluation Mode. Press 'Initialize JEFF' after changing.")
        config_expander_expanded = True
    else: # Chat mode
        IsInEvaluationMode = False
        st.sidebar.header("⚙️ Current Setup")
        st.sidebar.info("JEFF uses this setup. Switch to Evaluation Mode to change.")
        config_expander_expanded = False

    with st.sidebar.expander("RAG Configuration Details", expanded=config_expander_expanded):
        disable_widgets = (st.session_state.mode == "chat")
        embedding_options = EmbeddingModelType.list()
        reranker_options = RerankerModelType.list()
        llm_options = LLMModelType.list()
        vector_store_options = VectorStoreType.list()
        chunking_strategy_options = ChunkingStrategyType.list()

        def get_safe_index(options_list, current_value, default_index=0):
             try: return options_list.index(current_value)
             except ValueError: return default_index

        # Use session state for default selection and updates
        st.session_state.embedding_model = st.selectbox("Embedding Model", options=embedding_options, index=get_safe_index(embedding_options, st.session_state.embedding_model), key="sb_embedding_model", disabled=disable_widgets)
        st.session_state.reranker = st.selectbox("Re-ranker Model", options=reranker_options, index=get_safe_index(reranker_options, st.session_state.reranker), key="sb_reranker", disabled=disable_widgets)
        st.session_state.llm_model = st.selectbox("LLM Model", options=llm_options, index=get_safe_index(llm_options, st.session_state.llm_model), key="sb_llm_model", disabled=disable_widgets)
        st.session_state.vector_store = st.selectbox("Vector Store", options=vector_store_options, index=get_safe_index(vector_store_options, st.session_state.vector_store), key="sb_vector_store", disabled=disable_widgets)
        st.session_state.chunking_strategy = st.selectbox("Chunking Strategy", options=chunking_strategy_options, index=get_safe_index(chunking_strategy_options, st.session_state.chunking_strategy), key="sb_chunking_strategy", disabled=disable_widgets)

        try: # Show chunking description
            selected_strategy_enum = ChunkingStrategyType.from_string(st.session_state.chunking_strategy)
            selected_strategy_obj = ChunkingStrategyFactory.get_strategy(selected_strategy_enum.value)
            if selected_strategy_obj: st.caption(f"**{selected_strategy_enum.value.replace('_', ' ').title()}:** {selected_strategy_obj.description}")
        except Exception as e: logging.warning(f"Failed to get chunking description: {e}")

        # Sliders using session state
        st.session_state.chunk_size = st.slider("Chunk Size (chars)", 200, 4000, st.session_state.chunk_size, 50, key="sb_chunk_size", disabled=disable_widgets)
        st.session_state.chunk_overlap = st.slider("Chunk Overlap (chars)", 0, 1000, st.session_state.chunk_overlap, 25, key="sb_chunk_overlap", disabled=disable_widgets, help="Overlap < Chunk Size")
        st.session_state.top_k = st.slider("Docs to Retrieve (Top K)", 1, 15, st.session_state.top_k, 1, key="sb_top_k", disabled=disable_widgets)

        try: selected_vector_store_enum = VectorStoreType.from_string(st.session_state.vector_store)
        except ValueError: selected_vector_store_enum = None
        if selected_vector_store_enum == VectorStoreType.HYBRID:
             st.caption("Hybrid search mixes keyword and vector search.")
             st.session_state.hybrid_alpha = st.slider("Vector Weight (alpha)", 0.0, 1.0, st.session_state.hybrid_alpha, 0.05, key="sb_hybrid_alpha", disabled=disable_widgets, help="1.0=vector, 0.0=keyword")
             kw_weight = 1.0 - float(st.session_state.get('hybrid_alpha', 0.5))
             st.write(f"Keyword Weight: {kw_weight:.2f}")
        # else: # Ensure default alpha exists even if slider hidden
        #     if 'hybrid_alpha' not in st.session_state: st.session_state.hybrid_alpha = DEFAULT_HYBRID_ALPHA

    # Chat Mode Specific Settings
    st.sidebar.markdown("---")
    if st.session_state.mode == "chat":
        if st.sidebar.button("Clear Chat History", key="clear_chat"):
            st.session_state.messages = []; logging.info("Chat history cleared."); st.rerun()
        show_contexts_now = st.sidebar.toggle("Show JEFF's sources?", value=st.session_state.show_contexts, key="toggle_context_display")
        if show_contexts_now != st.session_state.show_contexts:
             st.session_state.show_contexts = show_contexts_now; st.rerun() # Rerun needed to update display

    # API Keys Status Expander
    st.sidebar.markdown("---")
    with st.sidebar.expander("🔑 API Key Status", expanded=False):
        try: # Check keys based on currently selected models
             embedding_val = st.session_state.get('embedding_model', DEFAULT_EMBEDDING_MODEL.value)
             vs_val = st.session_state.get('vector_store', DEFAULT_VECTOR_STORE.value)
             reranker_val = st.session_state.get('reranker', DEFAULT_RERANKER_MODEL.value)
             llm_val = st.session_state.get('llm_model', DEFAULT_LLM_MODEL.value)
             embedding_enum = EmbeddingModelType.from_string(embedding_val)
             vs_enum = VectorStoreType.from_string(vs_val)
             reranker_enum = RerankerModelType.from_string(reranker_val)
             llm_enum = LLMModelType.from_string(llm_val)
             check_api_keys(embedding_enum, vs_enum, reranker_enum, llm_enum)
        except ValueError as e: st.error(f"Error checking keys (invalid model): {e}")
        except Exception as e_api: st.error(f"Error checking keys: {e_api}"); logging.error(f"API Key check fail: {e_api}", exc_info=True)

        if st.session_state.api_key_status:
            missing_keys_found = False
            sorted_key_names = sorted(st.session_state.api_key_status.keys())
            for key_name in sorted_key_names:
                status = st.session_state.api_key_status[key_name]
                icon = "✅" if status == "Available" else "❌"; color = "green" if status == "Available" else "red"
                st.markdown(f"{icon} {key_name}: <span style='color:{color};'>{status}</span>", unsafe_allow_html=True)
                if status == "Missing": missing_keys_found = True
            if missing_keys_found: st.warning("Missing keys needed for current config.", icon="🔑"); st.caption("Add to `.env` & restart if needed.")
        else: st.info("No external API keys currently required.")

    # Initialize Button
    st.sidebar.markdown("---")
    disable_init = not st.session_state.file_path
    if st.sidebar.button("🚀 Initialize JEFF", key="init_pipeline", help="Load textbook with current settings.", disabled=disable_init):
        try: # Read config from session state safely
            embedding_val = st.session_state.get('embedding_model', DEFAULT_EMBEDDING_MODEL.value)
            vs_val = st.session_state.get('vector_store', DEFAULT_VECTOR_STORE.value)
            reranker_val = st.session_state.get('reranker', DEFAULT_RERANKER_MODEL.value)
            llm_val = st.session_state.get('llm_model', DEFAULT_LLM_MODEL.value)
            cs_val = st.session_state.get('chunking_strategy', DEFAULT_CHUNKING_STRATEGY.value)
            embedding_enum = EmbeddingModelType.from_string(embedding_val)
            vs_enum = VectorStoreType.from_string(vs_val)
            reranker_enum = RerankerModelType.from_string(reranker_val)
            llm_enum = LLMModelType.from_string(llm_val)
            cs_enum = ChunkingStrategyType.from_string(cs_val)
            hybrid_alpha_val = float(st.session_state.get('hybrid_alpha', DEFAULT_HYBRID_ALPHA))
            chunk_size_val = int(st.session_state.get('chunk_size', DEFAULT_CHUNK_SIZE))
            chunk_overlap_val = int(st.session_state.get('chunk_overlap', DEFAULT_CHUNK_OVERLAP))
            top_k_val = int(st.session_state.get('top_k', DEFAULT_TOP_K))
        except (ValueError, TypeError) as e: st.sidebar.error(f"Invalid config: {e}"); logging.error(f"Config error on Init: {e}"); st.stop()

        missing_keys = check_api_keys(embedding_enum, vs_enum, reranker_enum, llm_enum)
        if missing_keys: st.sidebar.error(f"Cannot initialize. Missing keys: {', '.join(missing_keys)}", icon="🔑")
        else:
            with st.spinner("Warming up JEFF's brain..."):
                pipeline_instance = initialize_pipeline(
                    file_path=st.session_state.file_path, embedding_model_enum=embedding_enum,
                    vector_store_enum=vs_enum, reranker_enum=reranker_enum, llm_enum=llm_enum,
                    chunking_strategy_enum=cs_enum, hybrid_alpha=hybrid_alpha_val,
                    chunk_size=chunk_size_val, chunk_overlap=chunk_overlap_val, top_k=top_k_val
                )
            if pipeline_instance:
                st.sidebar.success("JEFF is initialized!")
                st.session_state.permutation_df = None; st.session_state.permutation_results = None # Clear old results
                st.rerun() # Update main page status
            else: st.sidebar.error("Initialization failed. Check logs.")
    elif disable_init: st.sidebar.caption("Upload textbook to enable.")

    st.sidebar.markdown("---")


def attempt_automatic_initialization():
    """Tries to initialize RAG pipeline automatically on startup if possible."""
    if st.session_state.pipeline is None and st.session_state.file_path and os.path.exists(st.session_state.file_path):
        logging.info("Attempting automatic RAG pipeline initialization on startup.")
        init_placeholder = st.empty(); init_placeholder.info("Trying auto-setup...")

        # Use DEFAULT enums
        default_embedding_enum = DEFAULT_EMBEDDING_MODEL; default_vs_enum = DEFAULT_VECTOR_STORE
        default_reranker_enum = DEFAULT_RERANKER_MODEL; default_llm_enum = DEFAULT_LLM_MODEL
        default_cs_enum = DEFAULT_CHUNKING_STRATEGY

        missing_keys = check_api_keys(default_embedding_enum, default_vs_enum, default_reranker_enum, default_llm_enum)
        if missing_keys:
            init_placeholder.warning(f"Auto-setup skipped: Missing keys ({', '.join(missing_keys)}). Initialize manually.", icon="🔑")
            logging.warning(f"Auto-init skipped due to missing keys: {missing_keys}")
        else:
            logging.info("Default keys found. Proceeding with auto-initialization.")
            init_placeholder.empty()
            with st.spinner("JEFF is warming up... (Auto-initializing)"):
                 try: # Initialize using defaults (values might be in state or use constants)
                    pipeline_instance = initialize_pipeline(
                        file_path=st.session_state.file_path,
                        embedding_model_enum=default_embedding_enum, vector_store_enum=default_vs_enum,
                        reranker_enum=default_reranker_enum, llm_enum=default_llm_enum,
                        chunking_strategy_enum=default_cs_enum,
                        hybrid_alpha=st.session_state.get('hybrid_alpha', DEFAULT_HYBRID_ALPHA),
                        chunk_size=st.session_state.get('chunk_size', DEFAULT_CHUNK_SIZE),
                        chunk_overlap=st.session_state.get('chunk_overlap', DEFAULT_CHUNK_OVERLAP),
                        top_k=st.session_state.get('top_k', DEFAULT_TOP_K)
                    )
                    if pipeline_instance:
                        st.success("JEFF automatically initialized!"); logging.info("Auto-init successful."); st.rerun()
                    else: st.error("Auto-init failed. Try manual."); logging.error("Auto-init failed.")
                 except Exception as e:
                     st.error(f"Auto-init error: {e}. Try manual."); logging.error(f"Auto-init error: {e}", exc_info=True)
    # Clear any leftover placeholder (might happen if check failed early)
    # if 'init_placeholder' in locals() and hasattr(init_placeholder, 'empty'): init_placeholder.empty()


def main():
    st.title("👋 Hey! I'm JEFF, Your Study Buddy")

    # Attempt Automatic Initialization only once per session start if needed
    if not st.session_state.auto_init_attempted:
         st.session_state.auto_init_attempted = True
         attempt_automatic_initialization()

    # Display Sidebar (controls interactions, should render early)
    display_settings_panel()

    # --- Main Content Area ---
    if st.session_state.mode == "chat":
        st.markdown("""
        Got that big exam coming up? Don't sweat it! 😅 Upload your textbook (.txt format) using the sidebar,
        hit **Initialize JEFF**, and then ask me anything. Choose **Read Response** or **Hear Response** below my answers. Let's ace this thing! 🚀
        """)
        display_chat_interface() # Use the updated function with tabs
    else: # Evaluation mode
        st.markdown("""
        Alright, let's put different study strategies to the test! 🔬 Upload your textbook, use the **Evaluation Config** in the sidebar,
        **Initialize JEFF**, then ask a question, provide the perfect answer (ground truth), and see how each setup performs.
        Test the current config or run the **All Permutations Test**.
        """)
        display_evaluation_interface()

    # Display Chunking Strategy info at the bottom
    st.markdown("---")
    with st.expander("📚 About Chunking Strategies"):
        try:
            chunking_strategies = ChunkingStrategyFactory.get_all_strategies()
            for name, strategy in chunking_strategies.items():
                st.markdown(f"##### {name.replace('_', ' ').title()}")
                st.markdown(strategy.description); st.markdown("---")
        except Exception as e: st.warning(f"Could not load chunking descriptions: {e}")


if __name__ == "__main__":
    # Ensure necessary directories exist if needed by vector stores (e.g., Chroma)
    # if not os.path.exists("./vector_store_data"):
    #    os.makedirs("./vector_store_data")
    main()