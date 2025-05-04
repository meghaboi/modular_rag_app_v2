from gtts import gTTS
import io
from typing import Optional
import re
from openai import OpenAI
import httpx # Import httpx for OpenAI client timeout

import streamlit as st
import os
import tempfile
from typing import List, Dict, Any, Optional # Make sure Optional is here
import pandas as pd
from dotenv import load_dotenv
import itertools
import time
import datetime
import base64
import logging # Added for better debugging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from enums import (
    EmbeddingModelType,
    RerankerModelType,
    LLMModelType,
    VectorStoreType,
    ChunkingStrategyType,
    EvaluationBackendType,
    EvaluationMetricType
)

# Import our custom modules
from embedding_models import EmbeddingModelFactory
from rerankers import RerankerFactory
from vector_stores import VectorStoreFactory
from llm_models import LLMFactory
from evaluator import EvaluatorFactory
from rag_pipeline import RAGPipeline, ChunkingStrategyFactory

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

# --- Initialize Session State ---
# ... (session state initialization remains the same) ...
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
if 'is_settings_open' not in st.session_state:
    st.session_state.is_settings_open = False # Consider removing if not used
if 'show_contexts' not in st.session_state:
    st.session_state.show_contexts = False
if 'api_key_status' not in st.session_state:
    st.session_state.api_key_status = {}

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
        # Use a more specific suffix if possible, otherwise keep .txt
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

def toggle_mode(): # Consider renaming or linking directly to radio button
    st.session_state.mode = "evaluation" if st.session_state.mode == "chat" else "chat"
    st.session_state.messages = [] # Reset messages when switching modes
    st.session_state.pipeline = None # Reset pipeline as defaults might apply differently
    logging.info(f"Switched mode to: {st.session_state.mode}")
    st.rerun()

def toggle_contexts():
    st.session_state.show_contexts = not st.session_state.show_contexts
    st.rerun()

def check_api_keys(embedding_model_enum, vector_store_enum, reranker_enum, llm_enum):
    """Check if required API keys are available in environment"""
    # ... (check_api_keys function remains the same) ...
    api_keys_status = {}
    missing_keys_list = [] # Store missing keys for easier checking

    # --- Determine required keys based on selections ---
    openai_needed = (embedding_model_enum == EmbeddingModelType.OPENAI or
                     llm_enum in [LLMModelType.OPENAI_GPT35, LLMModelType.OPENAI_GPT4])
    cohere_needed = (embedding_model_enum == EmbeddingModelType.COHERE or
                     reranker_enum in [RerankerModelType.COHERE_V2, RerankerModelType.COHERE_V3, RerankerModelType.COHERE_MULTILINGUAL])
    gemini_needed = (embedding_model_enum == EmbeddingModelType.GEMINI or
                     llm_enum == LLMModelType.GEMINI)
    anthropic_needed = (llm_enum in [LLMModelType.CLAUDE_3_OPUS, LLMModelType.CLAUDE_37_SONNET])
    mistral_needed = (embedding_model_enum == EmbeddingModelType.MISTRAL or
                      llm_enum in [LLMModelType.MISTRAL_LARGE, LLMModelType.MISTRAL_MEDIUM, LLMModelType.MISTRAL_SMALL])
    voyage_needed = (embedding_model_enum == EmbeddingModelType.VOYAGE or
                     reranker_enum in [RerankerModelType.VOYAGE, RerankerModelType.VOYAGE_2])
    openai_needed = True # Always True because of TTS

    # --- Check and record status ---
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
    # Log only if status changes or first time? For now, log every check.
    # logging.info(f"API Key Status Check: {api_keys_status}")
    return missing_keys_list # Return list of missing keys

def initialize_pipeline(file_path, embedding_model_enum, vector_store_enum, reranker_enum, llm_enum,
                        chunking_strategy_enum, hybrid_alpha, chunk_size, chunk_overlap, top_k):
    """Initialize RAG pipeline with selected configuration"""
    # ... (initialize_pipeline function remains the same) ...
    logging.info(f"Attempting to initialize RAG pipeline with config:")
    logging.info(f"  Embedding: {embedding_model_enum.value}, Vector Store: {vector_store_enum.value}, Reranker: {reranker_enum.value}, LLM: {llm_enum.value}")
    logging.info(f"  Chunking: {chunking_strategy_enum.value}, Size: {chunk_size}, Overlap: {chunk_overlap}, Top K: {top_k}, Hybrid Alpha: {hybrid_alpha}")

    if not file_path or not os.path.exists(file_path):
        st.error("Cannot initialize pipeline: Document file path is invalid or missing.")
        logging.error("Pipeline initialization failed: Invalid file path.")
        return None

    try:
        # Initialize components based on selection
        embedding_model_instance = EmbeddingModelFactory.create_model(embedding_model_enum)

        if vector_store_enum == VectorStoreType.HYBRID:
            # Ensure alpha is passed if store type is hybrid
            vector_store_instance = VectorStoreFactory.create_store(vector_store_enum, alpha=hybrid_alpha)
        else:
            vector_store_instance = VectorStoreFactory.create_store(vector_store_enum)

        reranker_instance = None
        if reranker_enum != RerankerModelType.NONE:
            reranker_instance = RerankerFactory.create_reranker(reranker_enum)

        # --- Pass JEFF PERSONA PROMPT to LLM ---
        # This is crucial for the persona. Adapt based on your LLMFactory implementation.
        # Option 1: Pass as system prompt if supported by the factory/LLM
        llm_instance = LLMFactory.create_llm(llm_enum)
        # Option 2: If system prompt isn't directly supported, you might need to
        # modify your RAGPipeline.process_query to prepend it or use a specific
        # method on the llm_instance before generating the final answer.
        # Example (Conceptual - requires changes in RAGPipeline):
        # llm_instance = LLMFactory.create_llm(llm_enum)
        # # In RAGPipeline.process_query:
        # #   final_prompt = f"{JEFF_PERSONA_PROMPT}\n\nContext:\n{context_string}\n\nQuestion: {query}\n\nAnswer:"
        # #   response = self.llm.generate(final_prompt)
        # --- Choose the method that fits your LLMFactory/RAGPipeline structure ---

        chunking_strategy_instance = ChunkingStrategyFactory.get_strategy(chunking_strategy_enum.value)

        # Create RAG pipeline
        pipeline = RAGPipeline(
            embedding_model=embedding_model_instance,
            vector_store=vector_store_instance,
            reranker=reranker_instance,
            llm=llm_instance,
            top_k=top_k,
            chunking_strategy=chunking_strategy_instance
        )

        # Index documents
        logging.info(f"Indexing documents from: {file_path}")
        index_start_time = time.time()
        # Ensure index_documents can handle potential file reading errors
        try:
             # Pass chunk size/overlap during indexing
             pipeline.index_documents(file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        except Exception as index_e:
             logging.error(f"Error during document indexing: {index_e}", exc_info=True)
             st.error(f"Failed to index document: {index_e}")
             return None # Stop initialization if indexing fails
        index_end_time = time.time()
        logging.info(f"Document indexing completed in {index_end_time - index_start_time:.2f} seconds.")

        st.session_state.pipeline = pipeline # Store the initialized pipeline
        return pipeline

    except Exception as e:
        logging.error(f"Error initializing RAG pipeline: {e}", exc_info=True)
        st.error(f"Failed to initialize RAG pipeline: {e}")
        st.session_state.pipeline = None # Ensure pipeline is None on failure
        return None

# --- Run Single Config & Permutations ---
# ... (run_pipeline_with_config and run_all_permutations remain the same) ...
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
    # ... (function body is unchanged) ...
    config_str = f"{embedding_model_enum.value}, {vector_store_enum.value}, {reranker_enum.value}, {llm_enum.value}, {chunking_strategy_enum.value}"
    logging.info(f"Running pipeline with config: {config_str}")
    start_run_time = time.time()
    try:
        # Initialize components
        embedding_model_instance = EmbeddingModelFactory.create_model(embedding_model_enum)
        vector_store_instance = VectorStoreFactory.create_store(vector_store_enum, alpha=hybrid_alpha) if vector_store_enum == VectorStoreType.HYBRID else VectorStoreFactory.create_store(vector_store_enum)
        reranker_instance = RerankerFactory.create_reranker(reranker_enum) if reranker_enum != RerankerModelType.NONE else None
        # Pass Persona Prompt Here too for Evaluation runs
        llm_instance = LLMFactory.create_llm(llm_enum)
        chunking_strategy_instance = ChunkingStrategyFactory.get_strategy(chunking_strategy_enum.value)

        pipeline = RAGPipeline(
            embedding_model=embedding_model_instance,
            vector_store=vector_store_instance,
            reranker=reranker_instance,
            llm=llm_instance,
            top_k=top_k,
            chunking_strategy=chunking_strategy_instance
        )

        # Index (consider if re-indexing is needed every time or can be reused)
        # For evaluation, re-indexing per config might be desired for isolation
        pipeline.index_documents(file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Process query
        start_query_time = time.time()
        response, contexts = pipeline.process_query(user_query)
        query_elapsed_time = time.time() - start_query_time
        logging.info(f"Query processed in {query_elapsed_time:.2f}s. Response length: {len(response)}")

        # Run evaluation
        evaluation_results = {}
        avg_score = 0
        # Only evaluate if ground truth is provided
        if ground_truth:
            try:
                evaluator = EvaluatorFactory.create_evaluator(
                    EvaluationBackendType.RAGAS,
                    EvaluationMetricType.get_metrics_for_backend(EvaluationBackendType.RAGAS)
                )
                evaluation_results = evaluator.evaluate(
                    query=user_query,
                    response=response,
                    contexts=contexts,
                    ground_truth=ground_truth
                )
                # Calculate average score only if results are not empty
                if evaluation_results and isinstance(evaluation_results, dict):
                    valid_scores = [v for v in evaluation_results.values() if isinstance(v, (int, float))]
                    if valid_scores:
                        avg_score = sum(valid_scores) / len(valid_scores)
                logging.info(f"Evaluation scores: {evaluation_results}")
            except Exception as eval_e:
                logging.error(f"Error during evaluation for config {config_str}: {eval_e}", exc_info=True)
                st.warning(f"Evaluation failed for this configuration: {eval_e}")
                evaluation_results = {"error": str(eval_e)} # Store error
        else:
             logging.warning("No ground truth provided, skipping RAGAS evaluation.")


        total_elapsed_time = time.time() - start_run_time
        logging.info(f"Total run time for config {config_str}: {total_elapsed_time:.2f}s")

        return {
            "embedding_model": embedding_model_enum.value,
            "vector_store": vector_store_enum.value,
            "reranker": reranker_enum.value,
            "llm_model": llm_enum.value,
            "chunking_strategy": chunking_strategy_enum.value,
            "response": response,
            "evaluation_scores": evaluation_results, # May contain error string
            "avg_score": avg_score,
            "elapsed_time": total_elapsed_time, # Use total time for the run
            "contexts": contexts
        }
    except Exception as e:
        total_elapsed_time = time.time() - start_run_time
        logging.error(f"Error running pipeline config {config_str}: {e}", exc_info=True)
        st.error(f"Error with configuration - {config_str}: {str(e)}")
        return {
            "embedding_model": embedding_model_enum.value,
            "vector_store": vector_store_enum.value,
            "reranker": reranker_enum.value,
            "llm_model": llm_enum.value,
            "chunking_strategy": chunking_strategy_enum.value,
            "response": f"ERROR: {str(e)}",
            "evaluation_scores": {"error": str(e)},
            "avg_score": 0,
            "elapsed_time": total_elapsed_time,
            "contexts": []
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
    # ... (function body is unchanged) ...
    logging.info("Starting 'Run All Permutations'")
    # Define the models to include in permutations (as requested in original code)
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
        # Check progress bar exists before updating
        if hasattr(progress_bar, 'progress'):
            try:
                 progress_bar.progress((i + 1) / num_permutations, text=progress_text)
            except Exception as pb_e:
                 logging.warning(f"Could not update progress bar: {pb_e}")


        # API Key Check for current permutation (optional, but good practice)
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
                 file_path=file_path,
                 user_query=user_query,
                 ground_truth=ground_truth,
                 embedding_model_enum=embedding_model,
                 vector_store_enum=vector_store,
                 reranker_enum=reranker,
                 llm_enum=llm_model,
                 chunking_strategy_enum=chunking_strategy_enum,
                 hybrid_alpha=hybrid_alpha,
                 chunk_size=chunk_size,
                 chunk_overlap=chunk_overlap,
                 top_k=top_k
             )

        # Add individual metric scores to the result dictionary
        if "evaluation_scores" in result and isinstance(result["evaluation_scores"], dict):
             for metric, score in result["evaluation_scores"].items():
                 # Only add if score is numeric, ignore errors stored as values
                 if isinstance(score, (int, float)):
                     result[f"metric_{metric}"] = score

        results.append(result)

    end_permutations_time = time.time()
    total_time = end_permutations_time - start_permutations_time
    logging.info(f"All {num_permutations} permutations completed in {total_time:.2f} seconds.")
    # Check progress bar exists before updating
    if hasattr(progress_bar, 'progress'):
        try:
            progress_bar.progress(1.0, text="Permutations complete!")
            time.sleep(1) # Keep message visible
            progress_bar.empty()
        except Exception as pb_e:
             logging.warning(f"Could not update/empty progress bar: {pb_e}")


    # Convert to dataframe
    results_df = pd.DataFrame(results)

    # Define columns for CSV export (keep all relevant info)
    base_columns = ["embedding_model", "vector_store", "reranker", "llm_model", "chunking_strategy", "avg_score", "elapsed_time"]
    # Dynamically find all metric columns present in the DataFrame
    metric_columns = sorted([col for col in results_df.columns if col.startswith("metric_")])
    csv_columns = base_columns + metric_columns + ["response"] # Include response in CSV for review

    # Ensure all expected columns exist, add if missing (e.g., if evaluation failed)
    for col in csv_columns:
        if col not in results_df.columns:
            results_df[col] = pd.NA # Use pandas NA for missing values

    # Select only the desired columns for the final DataFrame to be returned and displayed
    # Make sure avg_score and elapsed_time are numeric for sorting later
    results_df['avg_score'] = pd.to_numeric(results_df['avg_score'], errors='coerce')
    results_df['elapsed_time'] = pd.to_numeric(results_df['elapsed_time'], errors='coerce')
    for col in metric_columns:
        results_df[col] = pd.to_numeric(results_df[col], errors='coerce')

    display_df = results_df[csv_columns].copy() # Create a copy for display

    return display_df, results # Return DF for display/CSV and raw results list

# --- NEW TTS Helper Function ---
# --- NEW TTS Helper Function ---
@st.cache_data(show_spinner=False) # Cache TTS results
def text_to_speech(text: str) -> Optional[bytes]:
    """Generates speech from text using OpenAI TTS and returns audio bytes."""
    if not text or not isinstance(text, str):
        logging.warning("TTS skipped: Input text is empty or not a string.")
        return None

    # *** ADDED: Clean the text ***
    # Remove markdown characters like *, #
    cleaned_text = re.sub(r'[#*]', '', text)
    # Optional: Remove URLs (simple version)
    cleaned_text = re.sub(r'http[s]?://\S+', '', cleaned_text)
    # Optional: Replace multiple spaces/newlines resulting from cleaning
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    if not cleaned_text:
        logging.warning("TTS skipped: Text is empty after cleaning.")
        return None

    # Check if OpenAI API key is available (essential for this function)
    if not os.getenv("OPENAI_API_KEY"):
        logging.error("OpenAI API Key not found. Cannot generate audio.")
        # Don't flood UI, maybe show a single warning elsewhere or rely on sidebar check
        # st.warning("OpenAI API Key missing. Cannot generate audio.", icon="🔑")
        return None

    try:
        # Increase timeout for potentially longer TTS requests
        # Note: Adjust timeout as needed (e.g., 60 seconds)
        client = OpenAI(timeout=httpx.Timeout(30.0, connect=10.0))

        # Choose a voice (alloy, echo, fable, onyx, nova, shimmer)
        # 'nova' or 'alloy' are often good starting points
        selected_voice = "alloy"
        # Choose a model ('tts-1' standard, 'tts-1-hd' high definition)
        selected_model = "tts-1"

        logging.info(f"Requesting OpenAI TTS: voice='{selected_voice}', model='{selected_model}', text length (cleaned): {len(cleaned_text)}")

        response = client.audio.speech.create(
            model=selected_model,
            voice=selected_voice,
            input=cleaned_text,
            response_format="mp3" # Specify format (mp3, opus, aac, flac)
        )

        # The response object itself doesn't directly contain bytes in v1.x.
        # You need to stream or read the content. .read() gets all bytes.
        audio_bytes = response.read()

        logging.info(f"OpenAI TTS audio generated successfully ({len(audio_bytes)} bytes).")
        return audio_bytes

    except ImportError:
        logging.error("OpenAI library not installed. Cannot generate audio.")
        st.warning("Audio generation requires `openai`. Please install it (`pip install openai`).", icon="⚠️")
        return None
    except Exception as e:
        logging.error(f"Error generating OpenAI TTS audio: {e}", exc_info=True)
        # Avoid flooding the UI with warnings
        # st.warning(f"Couldn't generate audio: {e}", icon="🔇")
        return None


# --- UI Display Functions ---

def display_chat_interface():
    st.header("💬 Chat with JEFF")
    st.markdown("Hey! Got questions about your textbook? Lay 'em on me. I'll break it down for ya.")

    # Initialize welcome message if chat is empty
    if not st.session_state.messages:
         welcome_msg = "Alright, let's get this study session started! What's on your mind?"
         # Generate audio for welcome message? Optional, but adds polish.
         welcome_audio_bytes = text_to_speech(welcome_msg)
         st.session_state.messages.append({
             "role": "assistant",
             "content": welcome_msg,
             "audio": welcome_audio_bytes # Store pre-generated audio
         })

    # Display message history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            # Play audio if it exists for the message (for history and welcome)
            if message["role"] == "assistant" and message.get("audio"):
                st.audio(message["audio"], format="audio/mp3")

            # Display contexts if available and show_contexts is True
            if st.session_state.show_contexts and message["role"] == "assistant" and "contexts" in message and message["contexts"]:
                with st.expander("🧠 Check out the textbook bits I used:"):
                    for i, context in enumerate(message["contexts"]):
                        st.markdown(f"**Snippet {i+1}:**")
                        # Use st.text or st.code for potentially long context strings
                        st.text(context) # st.text preserves formatting better for code/logs

            # Display processing time if available
            if message["role"] == "assistant" and "elapsed_time" in message:
                 st.write(f"_(JEFF cooked that up in {message['elapsed_time']:.2f} seconds)_")


    # User input
    user_query = st.chat_input("Type your question here...")

    if user_query:
        logging.info(f"User query received: {user_query}")
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Display user message immediately
        with st.chat_message("user"):
            st.write(user_query)

        # Check if pipeline is initialized
        if st.session_state.pipeline is None:
            logging.warning("Chat query received, but pipeline not initialized.")
            # Generate TTS for the warning message
            warning_msg = "Whoa there! Looks like we haven't loaded your textbook into my brain yet. Upload it and hit 'Initialize' in the sidebar first!"
            warning_audio = text_to_speech(warning_msg)
            with st.chat_message("assistant"):
                st.warning(warning_msg)
                if warning_audio:
                    st.audio(warning_audio, format="audio/mp3")
            st.session_state.messages.append(
                {"role": "assistant", "content": warning_msg, "audio": warning_audio}
            )
            st.stop() # Stop execution for this run if pipeline isn't ready

        # Process the query and get response
        with st.spinner("JEFF's thinking... 🤔"):
            try:
                start_time = time.time()
                logging.info("Processing query with RAG pipeline...")
                response, contexts = st.session_state.pipeline.process_query(user_query)
                elapsed_time = time.time() - start_time
                logging.info(f"Query processed successfully in {elapsed_time:.2f}s.")

                # --- Generate TTS Audio for the new response ---
                logging.info("Generating TTS audio for the response...")
                tts_start_time = time.time()
                audio_bytes = text_to_speech(response)
                tts_elapsed_time = time.time() - tts_start_time
                logging.info(f"TTS generation took {tts_elapsed_time:.2f}s.")
                # --- End TTS Generation ---


                # Display assistant response and audio
                with st.chat_message("assistant"):
                    st.write(response) # Display text
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3") # Play audio

                    # Display metadata (time, contexts)
                    st.write(f"_(JEFF cooked that up in {elapsed_time:.2f} seconds)_")
                    if st.session_state.show_contexts and contexts:
                         with st.expander("🧠 Check out the textbook bits I used:"):
                            for i, context in enumerate(contexts):
                                st.markdown(f"**Snippet {i+1}:**")
                                st.text(context) # Use st.text

                # Add assistant message to chat history including audio
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "contexts": contexts,
                    "elapsed_time": elapsed_time,
                    "audio": audio_bytes # Store generated audio bytes
                })
                st.rerun() # Rerun to clear the input box and show the latest message properly

            except Exception as e:
                logging.error(f"Error processing query or generating audio: {e}", exc_info=True)
                error_msg = f"Oof, hit a snag trying to answer that. Maybe try rephrasing? Error: {str(e)}"
                error_audio = text_to_speech(error_msg) # Try generating audio for the error too
                with st.chat_message("assistant"):
                    st.error(error_msg)
                    if error_audio:
                        st.audio(error_audio, format="audio/mp3")
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg, "audio": error_audio }
                )


def display_evaluation_interface():
    st.header("🧪 RAG Evaluation Mode")
    # ... (display_evaluation_interface remains the same) ...
    st.markdown("Let's test out different setups. Give me a question and the perfect answer (ground truth) to see how well various RAG configurations perform.")

    # Ensure pipeline is initialized before allowing evaluation actions
    if st.session_state.pipeline is None and st.session_state.file_path:
        st.warning("💡 Looks like you've uploaded a document, but the RAG pipeline isn't active yet. Please hit 'Initialize JEFF' in the sidebar using the default or evaluation settings before running evaluations.")
    elif not st.session_state.file_path:
         st.warning("💡 Upload a document first using the sidebar!")


    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Evaluation Inputs")
        user_query = st.text_area("Enter your question:", height=100, key="eval_query")
        ground_truth = st.text_area("Enter the ideal 'ground truth' answer:", height=100, key="eval_ground_truth")
        st.info("Providing ground truth enables detailed RAGAS evaluation scores.")

        # Buttons - disable if pipeline not ready or no file
        disable_buttons = not st.session_state.file_path
        # Disable Initialize if no file path
        # Disable Eval/Permutation buttons if pipeline not initialized *after* file upload
        # This allows initializing first.
        disable_eval_buttons = st.session_state.pipeline is None or disable_buttons

        process_button = st.button("Evaluate Current Config", disabled=disable_eval_buttons)
        permutation_button = st.button("Run All Permutations Test", disabled=disable_eval_buttons)

    # Process single query or run permutations (logic moved outside col1)
    if process_button and user_query:
        if not st.session_state.file_path:
            st.warning("Please upload a document first.")
        elif st.session_state.pipeline is None:
             st.warning("Pipeline not initialized. Please initialize from the sidebar.")
        else:
            # Get current configuration from session state (used by the initialized pipeline)
            try:
                 embedding_model_enum = EmbeddingModelType.from_string(st.session_state.embedding_model)
                 vector_store_enum = VectorStoreType.from_string(st.session_state.vector_store)
                 reranker_enum = RerankerModelType.from_string(st.session_state.reranker)
                 llm_enum = LLMModelType.from_string(st.session_state.llm_model)
                 chunking_strategy_enum = ChunkingStrategyType.from_string(st.session_state.chunking_strategy)
            except ValueError as e:
                 st.error(f"Error reading current configuration: {e}")
                 st.stop()

            # Check API keys for the CURRENTLY LOADED pipeline's config
            missing_keys = check_api_keys(embedding_model_enum, vector_store_enum, reranker_enum, llm_enum)
            if missing_keys:
                st.error(f"Cannot proceed. Missing API keys for the current configuration: {', '.join(missing_keys)}")
            else:
                # Process query using the *already initialized* pipeline
                with st.spinner("Evaluating current configuration..."):
                    logging.info("Evaluating single query using the existing pipeline.")
                    start_eval_time = time.time()
                    try:
                        response, contexts = st.session_state.pipeline.process_query(user_query)
                        eval_elapsed_time = time.time() - start_eval_time

                        # Run evaluation metrics
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
                                logging.info(f"Single config evaluation scores: {evaluation_results}")
                             except Exception as eval_e:
                                logging.error(f"Evaluation failed for single config run: {eval_e}", exc_info=True)
                                st.warning(f"Evaluation metrics failed: {eval_e}")
                                evaluation_results = {"error": str(eval_e)}
                        else:
                             logging.warning("No ground truth provided for single config run, skipping evaluation metrics.")


                        # --- Display Results Directly ---
                        # Clear previous permutation results when running single eval
                        st.session_state.permutation_df = None
                        st.session_state.permutation_results = None

                        # Update Column 2 with single result
                        with col2:
                            st.subheader("Evaluation Result (Current Config)")
                            st.markdown(f"**Configuration:** `{st.session_state.embedding_model} | {st.session_state.vector_store} | {st.session_state.reranker} | {st.session_state.llm_model} | {st.session_state.chunking_strategy}`")
                            st.write(f"**Processing Time:** {eval_elapsed_time:.2f} seconds")

                            with st.expander("Response", expanded=True):
                                st.write(response)

                            with st.expander("Retrieved Contexts", expanded=False):
                                 if contexts:
                                    for i, context in enumerate(contexts):
                                        st.markdown(f"**Context {i+1}:**")
                                        st.text(context)
                                 else:
                                     st.write("No contexts were retrieved.")

                            if evaluation_results and isinstance(evaluation_results, dict) and "error" not in evaluation_results:
                                 st.subheader("Evaluation Scores")
                                 metric_cols = st.columns(len(evaluation_results))
                                 i = 0
                                 for metric, score in evaluation_results.items():
                                     with metric_cols[i]:
                                         # Format score, handle potential non-numeric gracefully
                                         score_display = f"{score:.2f}" if isinstance(score, (int, float)) else "N/A"
                                         st.metric(label=metric.replace('_', ' ').title(), value=score_display)
                                     i += 1
                                 # Display average score if calculated
                                 if avg_score > 0 or len(valid_scores) > 0:
                                     st.metric("Overall Average Score", f"{avg_score:.2f}")
                            elif "error" in evaluation_results:
                                st.warning(f"Evaluation scores could not be calculated: {evaluation_results['error']}")
                            elif ground_truth:
                                 st.warning("Evaluation scores could not be calculated (no metrics returned).")
                            else:
                                 st.info("Provide ground truth to see evaluation scores.")

                    except Exception as e:
                        logging.error(f"Error running single evaluation: {e}", exc_info=True)
                        st.error(f"Error processing evaluation: {str(e)}")
                        with col2:
                            st.error(f"Failed to evaluate: {e}") # Show error in results area


    elif permutation_button and user_query:
        if not st.session_state.file_path:
            st.warning("Please upload a document first.")
        elif st.session_state.pipeline is None:
             st.warning("Pipeline not initialized. Please initialize from the sidebar (this ensures the document is processed at least once).")
        else:
            # Check for potentially needed API keys BEFORE starting permutations
            st.info("Checking for API keys potentially needed for permutations...")
            # Check keys needed by the permutation list defined in run_all_permutations
            perm_emb = [EmbeddingModelType.OPENAI, EmbeddingModelType.GEMINI, EmbeddingModelType.MISTRAL]
            perm_rerank = list(RerankerModelType)
            perm_llm = [LLMModelType.CLAUDE_37_SONNET, LLMModelType.GEMINI]

            potential_missing = set()
            for e, r, l in itertools.product(perm_emb, perm_rerank, perm_llm):
                # Use dummy vector store as it doesn't require keys for the check itself
                potential_missing.update(check_api_keys(e, VectorStoreType.FAISS, r, l))

            if potential_missing:
                st.warning(f"Heads up! Some permutations might fail or be skipped. Missing potential API keys: {', '.join(potential_missing)}. Make sure they are in your .env file if needed.")
            else:
                st.success("Looks like all potentially required API keys are present!")

            # Get chunking strategy from session state
            try:
                 chunking_strategy_enum = ChunkingStrategyType.from_string(st.session_state.chunking_strategy)
            except ValueError as e:
                 st.error(f"Invalid chunking strategy selected: {e}")
                 st.stop()


            # Run permutations
            with st.spinner("Running all permutations... This could take a while, grab a coffee! ☕️"):
                 results_df, all_results = run_all_permutations(
                    file_path=st.session_state.file_path,
                    user_query=user_query,
                    ground_truth=ground_truth, # Pass ground truth here
                    chunk_size=st.session_state.chunk_size,
                    chunk_overlap=st.session_state.chunk_overlap,
                    top_k=st.session_state.top_k,
                    hybrid_alpha=st.session_state.hybrid_alpha,
                    chunking_strategy_enum=chunking_strategy_enum
                )

            # Store results in session state
            st.session_state.permutation_df = results_df
            st.session_state.permutation_results = all_results
            logging.info("Permutations completed and results stored in session state.")

            st.success("All permutations complete! Results are shown below.")
            # Use rerun to ensure the results display updates cleanly in col2
            st.rerun()

    elif (process_button or permutation_button) and not user_query:
        st.warning("⚠️ Please enter a question to evaluate.")


    # Display Permutation Results in Column 2 (if available)
    with col2:
        if st.session_state.permutation_df is not None and not st.session_state.permutation_df.empty:
            st.subheader("Permutation Results Summary")

            # Display download link for CSV
            st.markdown(get_csv_download_link(st.session_state.permutation_df), unsafe_allow_html=True)

            # Show top results sorted by average score (handle cases with no score or NaN)
            results_to_display = st.session_state.permutation_df.copy()
            # Use the numeric avg_score column, fill NaN with a low value for sorting
            results_to_display['avg_score_numeric'] = results_to_display['avg_score'].fillna(-1)
            top_results = results_to_display.sort_values('avg_score_numeric', ascending=False).head(10) # Show top 10

            # Select columns to display in the summary table
            display_cols = ["embedding_model", "vector_store", "reranker", "llm_model", "chunking_strategy", "avg_score", "elapsed_time"]
            # Add metric columns if they exist and were created in run_all_permutations
            metric_cols_exist = sorted([col for col in results_to_display.columns if col.startswith("metric_")])
            display_cols.extend(metric_cols_exist)

            # Format numeric columns for display
            format_dict = {'avg_score': "{:.2f}", 'elapsed_time': "{:.2f}"}
            for col in metric_cols_exist:
                format_dict[col] = "{:.2f}"

            # Use st.dataframe for better display and handling of NaNs
            st.dataframe(top_results[display_cols].style.format(format_dict, na_rep="N/A"))

            st.markdown("---")
            st.subheader("Explore Individual Results")

            # Allow user to select a specific configuration to view details
            # Create readable labels for the selectbox
            config_labels = []
            if st.session_state.permutation_results: # Check if raw results exist
                for index, row_data in enumerate(st.session_state.permutation_results):
                    # Use .get() for safety in case keys are missing in error results
                    label = (f"{index}: {row_data.get('embedding_model','?')} / {row_data.get('vector_store','?')} / "
                             f"{row_data.get('reranker','?')} / {row_data.get('llm_model','?')} "
                             f"(Score: {row_data.get('avg_score', 0):.2f}, Time: {row_data.get('elapsed_time', 0):.1f}s)")
                    config_labels.append(label)

                # Use index as the selection key, format_func provides the display label
                # Check if config_labels is not empty before creating selectbox
                if config_labels:
                     selected_index = st.selectbox(
                         "Select Configuration to View Details:",
                         options=range(len(config_labels)), # Use simple range index
                         format_func=lambda index: config_labels[index], # Map index to label
                         key="permutation_select"
                     )

                     if selected_index is not None and selected_index < len(st.session_state.permutation_results):
                        # Retrieve the raw result dictionary using the selected index
                        selected_result = st.session_state.permutation_results[selected_index]

                        st.markdown(f"**Details for Configuration {selected_index}:**")
                        st.markdown(f"**Models:** `{selected_result.get('embedding_model','N/A')} | {selected_result.get('vector_store','N/A')} | {selected_result.get('reranker','N/A')} | {selected_result.get('llm_model','N/A')}`")
                        st.markdown(f"**Chunking:** `{selected_result.get('chunking_strategy','N/A')}`")
                        st.write(f"**Processing Time:** {selected_result.get('elapsed_time', 0):.2f} seconds")

                        with st.expander("Response", expanded=True):
                            st.write(selected_result.get('response', 'N/A'))

                        with st.expander("Retrieved Contexts", expanded=False):
                             contexts = selected_result.get('contexts', [])
                             if contexts:
                                 for i, context in enumerate(contexts):
                                    st.markdown(f"**Context {i+1}:**")
                                    st.text(context)
                             else:
                                 st.write("No contexts available for this result.")

                        # Display evaluation scores if they exist and are a dict
                        eval_scores = selected_result.get('evaluation_scores', {})
                        if isinstance(eval_scores, dict) and "error" not in eval_scores and eval_scores:
                            st.subheader("Evaluation Scores")
                            metric_cols_detail = st.columns(len(eval_scores))
                            i = 0
                            avg_score_detail = 0
                            valid_scores_detail = []
                            for metric, score in eval_scores.items():
                                 with metric_cols_detail[i]:
                                     score_display = f"{score:.2f}" if isinstance(score, (int, float)) else "N/A"
                                     st.metric(label=metric.replace('_', ' ').title(), value=score_display)
                                     if isinstance(score, (int, float)):
                                         valid_scores_detail.append(score)
                                 i+=1
                            # Display average score if available in the raw result or calculable
                            avg_score_detail = selected_result.get('avg_score', 0)
                            if avg_score_detail > 0 or len(valid_scores_detail) > 0:
                                 st.metric("Overall Average Score", f"{avg_score_detail:.2f}")

                        elif isinstance(eval_scores, dict) and "error" in eval_scores:
                            st.warning(f"Evaluation failed for this run: {eval_scores['error']}")
                        elif ground_truth: # Only show warning if ground truth was provided but scores are missing
                             st.warning("Evaluation scores are not available for this result (might have failed or been skipped).")
                        else:
                             st.info("Provide ground truth during the permutation run to see evaluation scores.")
                else:
                    st.info("No permutation results available to select.")

        elif st.session_state.permutation_results is not None: # Handle case where results exist but df is empty (e.g., all skipped)
             st.info("Permutation run finished, but no valid results were generated (check logs for errors or skipped runs).")
        # else: # No results have been generated yet in this session for permutations
            # st.info("Run permutations or evaluate a single config to see results here.")


def display_settings_panel():
    st.sidebar.image("https://www.nicepng.com/png/detail/972-9721863_raising-hand-icon-png.png", width=80) # Placeholder 'cool friend' icon
    st.sidebar.title("JEFF's Controls")

    # Mode selector (Chat vs Evaluation)
    # ... (mode selector logic remains the same) ...
    mode_options = {"💬 Chat with JEFF": "chat", "🧪 Test Setups (Evaluation)": "evaluation"}
    # Guard against potential state inconsistencies if mode somehow gets invalid value
    current_mode = st.session_state.get('mode', 'chat')
    if current_mode not in mode_options.values():
        current_mode = 'chat' # Default to chat if invalid state
        st.session_state.mode = 'chat'
    current_mode_index = list(mode_options.values()).index(current_mode)

    selected_mode_label = st.sidebar.radio(
        "Select Mode",
        options=list(mode_options.keys()),
        index=current_mode_index,
        key="mode_radio",
        help="Switch between asking JEFF questions and testing different RAG configurations."
    )
    # Update mode if changed
    new_mode = mode_options[selected_mode_label]
    if new_mode != st.session_state.mode:
        st.session_state.mode = new_mode
        st.session_state.messages = [] # Reset chat
        # Reset pipeline ONLY if switching away from Chat, maybe keep if switching TO Chat?
        # Let's reset always for simplicity, user needs to re-init anyway if config changed.
        st.session_state.pipeline = None
        st.session_state.permutation_results = None # Clear eval results
        st.session_state.permutation_df = None
        logging.info(f"Mode changed to: {st.session_state.mode}. Resetting state.")
        st.rerun()

    # File upload
    # ... (file upload logic remains the same) ...
    st.sidebar.header("📚 Load Textbook")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your textbook (.txt file)",
        type=['txt'],
        key="file_uploader",
        help="Upload the text content you want JEFF to study."
        )

    if uploaded_file is not None:
        # Check if it's a new file upload by comparing names OR potentially file content hash? Name is simpler.
        if uploaded_file.name != st.session_state.get('last_uploaded_filename', None):
            logging.info(f"New file upload detected: {uploaded_file.name}")
            # Display spinner during save
            with st.spinner(f"Processing '{uploaded_file.name}'..."):
                st.session_state.file_path = save_uploaded_file(uploaded_file)

            if st.session_state.file_path: # Only update if save was successful
                 st.session_state.last_uploaded_filename = uploaded_file.name
                 st.session_state.pipeline = None # Crucial: Reset pipeline when new file is uploaded
                 st.session_state.messages = [] # Reset chat history as context changed
                 st.session_state.permutation_results = None # Clear old eval results
                 st.session_state.permutation_df = None
                 st.sidebar.success(f"'{uploaded_file.name}' loaded!")
                 logging.info("New file loaded, pipeline and chat reset. Rerunning to update status.")
                 # Rerun needed to update System Status and clear chat display
                 st.rerun()
            else:
                 st.sidebar.error("Failed to process uploaded file.")
                 # Keep old file path if save failed? Or set to None? Setting to None is safer.
                 st.session_state.file_path = None
                 st.session_state.last_uploaded_filename = None
                 st.session_state.pipeline = None


    # Display system status
    # ... (system status logic remains the same) ...
    st.sidebar.header("🚦 System Status")
    with st.sidebar.container(border=True): # Use container for better layout
        if st.session_state.file_path and os.path.exists(st.session_state.file_path):
            st.success(f"✅ Textbook loaded: {st.session_state.last_uploaded_filename}")
        else:
            st.warning("⚠️ No textbook loaded") # Changed to warning

        if st.session_state.pipeline:
            st.success("✅ JEFF is ready!")
        else:
            st.warning("⏳ JEFF needs setup (Initialize)") # Changed wording slightly


    # --- Configuration Options (Show based on mode) ---
    st.sidebar.markdown("---")
    # Evaluation mode shows full config
    if st.session_state.mode == "evaluation":
        st.sidebar.header("🛠️ Evaluation Config")
        st.sidebar.info("Adjust these settings to test different RAG setups in Evaluation Mode. Press 'Initialize JEFF' after changing settings.")
        config_expander_expanded = True # Expand by default in eval mode
    # Chat mode shows simplified/read-only view or hides it
    else: # Chat mode
        st.sidebar.header("⚙️ Current Setup")
        st.sidebar.info("JEFF is using this setup. To change it, switch to Evaluation Mode.")
        config_expander_expanded = False # Collapse by default in chat mode

    # Use an expander for config options
    with st.sidebar.expander("RAG Configuration Details", expanded=config_expander_expanded):
        disable_widgets = (st.session_state.mode == "chat") # Disable widgets in chat mode

        # --- Models & Storage ---
        embedding_options = EmbeddingModelType.list()
        reranker_options = RerankerModelType.list()
        llm_options = LLMModelType.list()
        vector_store_options = VectorStoreType.list()
        chunking_strategy_options = ChunkingStrategyType.list()

        # Helper to get index safely
        def get_safe_index(options_list, current_value, default_index=0):
             try:
                 return options_list.index(current_value)
             except ValueError:
                 # If current value isn't valid (e.g., from old session state), log and return default
                 logging.warning(f"Value '{current_value}' not found in options {options_list}. Using default index {default_index}.")
                 # Attempt to update session state to a valid default? Risky during render.
                 # Let's just return default index for this render pass.
                 return default_index

        # Use session state for default selection via index and key for updates
        st.session_state.embedding_model = st.selectbox(
            "Embedding Model", options=embedding_options,
            index=get_safe_index(embedding_options, st.session_state.embedding_model),
            key="sb_embedding_model", disabled=disable_widgets
        )
        st.session_state.reranker = st.selectbox(
            "Re-ranker Model", options=reranker_options,
            index=get_safe_index(reranker_options, st.session_state.reranker),
            key="sb_reranker", disabled=disable_widgets
        )
        st.session_state.llm_model = st.selectbox(
            "LLM Model", options=llm_options,
            index=get_safe_index(llm_options, st.session_state.llm_model),
            key="sb_llm_model", disabled=disable_widgets
        )
        st.session_state.vector_store = st.selectbox(
            "Vector Store", options=vector_store_options,
            index=get_safe_index(vector_store_options, st.session_state.vector_store),
            key="sb_vector_store", disabled=disable_widgets
        )
        st.session_state.chunking_strategy = st.selectbox(
            "Chunking Strategy", options=chunking_strategy_options,
            index=get_safe_index(chunking_strategy_options, st.session_state.chunking_strategy),
            key="sb_chunking_strategy", disabled=disable_widgets
        )

        # Show description of selected chunking strategy if valid
        try:
            selected_strategy_enum = ChunkingStrategyType.from_string(st.session_state.chunking_strategy)
            selected_strategy_obj = ChunkingStrategyFactory.get_strategy(selected_strategy_enum.value)
            if selected_strategy_obj:
                st.caption(f"**{selected_strategy_enum.value.replace('_', ' ').title()}:** {selected_strategy_obj.description}")
        except Exception as e:
            st.caption("Could not load chunking strategy description.")
            logging.warning(f"Failed to get chunking strategy description: {e}")


        # --- Indexing & Retrieval ---
        st.session_state.chunk_size = st.slider("Chunk Size (chars)", 200, 4000, st.session_state.chunk_size, 50, key="sb_chunk_size", disabled=disable_widgets)
        st.session_state.chunk_overlap = st.slider("Chunk Overlap (chars)", 0, 1000, st.session_state.chunk_overlap, 25, key="sb_chunk_overlap", disabled=disable_widgets, help="How many characters chunks should share. Should be less than Chunk Size.")
        st.session_state.top_k = st.slider("Docs to Retrieve (Top K)", 1, 15, st.session_state.top_k, 1, key="sb_top_k", disabled=disable_widgets, help="How many text chunks JEFF looks at initially.")

        # Hybrid search settings (only show if relevant)
        try:
             selected_vector_store_enum = VectorStoreType.from_string(st.session_state.vector_store)
        except ValueError:
             selected_vector_store_enum = None # Handle invalid state

        if selected_vector_store_enum == VectorStoreType.HYBRID:
             st.caption("Hybrid search mixes keyword and vector search.")
             st.session_state.hybrid_alpha = st.slider(
                 "Vector Weight (alpha)", 0.0, 1.0, st.session_state.hybrid_alpha, 0.05,
                 key="sb_hybrid_alpha", disabled=disable_widgets,
                 help="1.0 = pure vector, 0.0 = pure keyword (BM25)"
             )
             # Show keyword weight for clarity, ensure it's calculated safely
             kw_weight = 1.0 - float(st.session_state.get('hybrid_alpha', 0.5))
             st.write(f"Keyword Weight: {kw_weight:.2f}")
        else:
             # Ensure default alpha is in session state even if slider not shown
             if 'hybrid_alpha' not in st.session_state: st.session_state.hybrid_alpha = DEFAULT_HYBRID_ALPHA

    # --- Settings applicable to both modes ---
    st.sidebar.markdown("---")

    # Reset Chat Button (Only makes sense in chat mode?)
    if st.session_state.mode == "chat":
        if st.sidebar.button("Clear Chat History", key="clear_chat", help="Wipe the current conversation."):
            st.session_state.messages = []
            logging.info("Chat history cleared by user.")
            st.rerun() # Rerun to clear chat display

    # Show Contexts Toggle (useful in chat mode primarily)
    if st.session_state.mode == "chat":
        show_contexts_now = st.sidebar.toggle(
            "Show JEFF's sources?",
            value=st.session_state.show_contexts,
            key="toggle_context_display",
            help="See the parts of the textbook JEFF used to answer."
         )
        # No need to rerun on toggle, display logic handles it per message
        if show_contexts_now != st.session_state.show_contexts:
             st.session_state.show_contexts = show_contexts_now
             # Rerun IS needed if you want existing messages to immediately show/hide context
             st.rerun()


    # --- API Keys Status (Always Visible) ---
    st.sidebar.markdown("---")
    with st.sidebar.expander("🔑 API Key Status", expanded=False):
        # Check keys based on *currently selected* models in session state
        try:
             # Use .get() with default values from constants to prevent errors if state is missing temporarily
             embedding_val = st.session_state.get('embedding_model', DEFAULT_EMBEDDING_MODEL.value)
             vs_val = st.session_state.get('vector_store', DEFAULT_VECTOR_STORE.value)
             reranker_val = st.session_state.get('reranker', DEFAULT_RERANKER_MODEL.value)
             llm_val = st.session_state.get('llm_model', DEFAULT_LLM_MODEL.value)

             embedding_enum = EmbeddingModelType.from_string(embedding_val)
             vs_enum = VectorStoreType.from_string(vs_val)
             reranker_enum = RerankerModelType.from_string(reranker_val)
             llm_enum = LLMModelType.from_string(llm_val)
             check_api_keys(embedding_enum, vs_enum, reranker_enum, llm_enum) # Updates st.session_state.api_key_status
        except ValueError as e:
             st.error(f"Error checking API keys due to invalid model selection: {e}")
        except Exception as e_api: # Catch other potential errors during check
             st.error(f"Error checking API keys: {e_api}")
             logging.error(f"API Key check failed: {e_api}", exc_info=True)


        if st.session_state.api_key_status:
            missing_keys_found = False
            # Sort keys for consistent display order
            sorted_key_names = sorted(st.session_state.api_key_status.keys())
            for key_name in sorted_key_names:
                status = st.session_state.api_key_status[key_name]
                status_icon = "✅" if status == "Available" else "❌"
                status_color = "green" if status == "Available" else "red"
                st.markdown(f"{status_icon} {key_name}: <span style='color:{status_color};'>{status}</span>", unsafe_allow_html=True)
                if status == "Missing":
                     missing_keys_found = True

            if missing_keys_found:
                st.warning("Missing API keys needed for the current configuration.", icon="🔑")
                st.caption("Add required keys to a `.env` file in the app directory (e.g., `OPENAI_API_KEY=sk-...`). You might need to restart the app after adding keys.")
        else:
            st.info("No external API keys are currently required for the selected local/free models.")


    # --- Initialize Pipeline Button (Always Visible but disable if no file) ---
    st.sidebar.markdown("---")
    disable_init = not st.session_state.file_path # Disable if no file uploaded

    if st.sidebar.button("🚀 Initialize JEFF", key="init_pipeline", help="Load the textbook with the current settings. Required before chatting or evaluating.", disabled=disable_init):
        # This button should always be clickable if a file exists, even if already initialized,
        # to allow re-initializing with changed settings.
        # if not st.session_state.file_path or not os.path.exists(st.session_state.file_path):
        #     st.sidebar.error("Please upload a textbook first!") # Should be disabled anyway
        # else:
        # Always use values from session state, which reflect defaults or evaluation settings
        try:
            # Use .get() with defaults for robustness during initialization
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

            # Also get numeric params safely from state
            hybrid_alpha_val = float(st.session_state.get('hybrid_alpha', DEFAULT_HYBRID_ALPHA))
            chunk_size_val = int(st.session_state.get('chunk_size', DEFAULT_CHUNK_SIZE))
            chunk_overlap_val = int(st.session_state.get('chunk_overlap', DEFAULT_CHUNK_OVERLAP))
            top_k_val = int(st.session_state.get('top_k', DEFAULT_TOP_K))

        except (ValueError, TypeError) as e:
             st.sidebar.error(f"Invalid configuration selected: {e}")
             logging.error(f"Config error on Initialize: {e}", exc_info=True)
             st.stop() # Stop if config is bad

        # Check API keys for the selected configuration before initializing
        missing_keys = check_api_keys(embedding_enum, vs_enum, reranker_enum, llm_enum)
        if missing_keys:
            st.sidebar.error(f"Cannot initialize. Missing keys: {', '.join(missing_keys)}", icon="🔑")
        else:
            # Initialize pipeline using values from session state
            # Show spinner within the sidebar button area if possible, or globally
            with st.spinner("Warming up JEFF's brain... (Initializing RAG pipeline)"):
                pipeline_instance = initialize_pipeline(
                    file_path=st.session_state.file_path,
                    embedding_model_enum=embedding_enum,
                    vector_store_enum=vs_enum,
                    reranker_enum=reranker_enum,
                    llm_enum=llm_enum,
                    chunking_strategy_enum=cs_enum,
                    hybrid_alpha=hybrid_alpha_val,
                    chunk_size=chunk_size_val,
                    chunk_overlap=chunk_overlap_val,
                    top_k=top_k_val
                )

            if pipeline_instance:
                st.sidebar.success("JEFF is initialized and ready!")
                # Clear any old evaluation results when re-initializing
                st.session_state.permutation_df = None
                st.session_state.permutation_results = None
                # Rerun to update the main page status/interface
                st.rerun()
            else:
                st.sidebar.error("Initialization failed. Check logs.")
    # Add a small note if disabled
    elif disable_init:
         st.sidebar.caption("Upload a textbook to enable initialization.")

    st.sidebar.markdown("---")

def attempt_automatic_initialization():
    """Tries to initialize the RAG pipeline automatically on startup if conditions are met."""
    # ... (attempt_automatic_initialization remains the same) ...
    if st.session_state.pipeline is None and st.session_state.file_path and os.path.exists(st.session_state.file_path):
        logging.info("Attempting automatic RAG pipeline initialization on startup.")

        # Use placeholder for immediate feedback
        init_placeholder = st.empty()
        init_placeholder.info("Detected textbook. Trying to set up JEFF automatically with default settings...")

        # Use the DEFAULT enums for the check and initialization
        default_embedding_enum = DEFAULT_EMBEDDING_MODEL
        default_vs_enum = DEFAULT_VECTOR_STORE
        default_reranker_enum = DEFAULT_RERANKER_MODEL
        default_llm_enum = DEFAULT_LLM_MODEL
        default_cs_enum = DEFAULT_CHUNKING_STRATEGY

        # Check API keys specifically for the default configuration
        missing_keys = check_api_keys(default_embedding_enum, default_vs_enum, default_reranker_enum, default_llm_enum)

        if missing_keys:
            init_placeholder.warning(f"Auto-setup skipped: Missing default API keys ({', '.join(missing_keys)}). Add to .env or Initialize manually.", icon="🔑")
            logging.warning(f"Automatic initialization skipped due to missing default API keys: {missing_keys}")
        else:
            logging.info("Required API keys for default config found. Proceeding with automatic initialization.")
            # Replace placeholder with spinner
            init_placeholder.empty() # Clear the info message
            with st.spinner("JEFF is warming up... (Auto-initializing)"):
                 try:
                    # Initialize using default values stored in session state (or constants if state not yet populated)
                    pipeline_instance = initialize_pipeline(
                        file_path=st.session_state.file_path,
                        embedding_model_enum=default_embedding_enum,
                        vector_store_enum=default_vs_enum,
                        reranker_enum=default_reranker_enum,
                        llm_enum=default_llm_enum,
                        chunking_strategy_enum=default_cs_enum,
                        hybrid_alpha=st.session_state.get('hybrid_alpha', DEFAULT_HYBRID_ALPHA),
                        chunk_size=st.session_state.get('chunk_size', DEFAULT_CHUNK_SIZE),
                        chunk_overlap=st.session_state.get('chunk_overlap', DEFAULT_CHUNK_OVERLAP),
                        top_k=st.session_state.get('top_k', DEFAULT_TOP_K)
                    )

                    if pipeline_instance:
                        st.success("JEFF automatically initialized with default settings!")
                        logging.info("Automatic RAG pipeline initialization successful.")
                        # Rerun needed to update status indicators and clear spinner
                        st.rerun()
                    else:
                        # Keep spinner? No, replace with error.
                        st.error("Automatic initialization failed. Try initializing manually from the sidebar.")
                        logging.error("Automatic RAG pipeline initialization failed.")

                 except Exception as e:
                     # Ensure spinner is cleared on error
                     st.error(f"Automatic initialization encountered an error: {e}. Try initializing manually.")
                     logging.error(f"Error during automatic initialization: {e}", exc_info=True)

    # If auto-init didn't run or is done, clear any leftover placeholder
    if 'init_placeholder' in locals() and hasattr(init_placeholder, 'empty'):
         pass # Spinner/message would have replaced or cleared it


def main():
    # --- Page Title and Intro ---
    st.title("👋 Hey! I'm JEFF, Your Study Buddy")

    # --- Attempt Automatic Initialization ---
    # Run this early, only once per session if pipeline isn't set and file exists
    if 'auto_init_attempted' not in st.session_state:
         st.session_state.auto_init_attempted = True # Mark that we've tried
         attempt_automatic_initialization()


    # --- Display Sidebar ---
    # Sidebar setup needs to happen before main content to capture interactions
    display_settings_panel() # Sidebar setup

    # --- Main Content Area ---
    if st.session_state.mode == "chat":
        st.markdown("""
        Got that big exam coming up? Don't sweat it! 😅 Upload your textbook (.txt format) using the sidebar,
        hit **Initialize JEFF**, and then ask me anything. I'll explain stuff in a way that actually makes sense, and you can even hear my responses! Let's ace this thing! 🚀
        """)
        display_chat_interface()
    else: # Evaluation mode
        st.markdown("""
        Alright, let's put different study strategies to the test! 🔬 Upload your textbook, then use the **Evaluation Config** in the sidebar
        to choose different models and settings. **Initialize JEFF** with those settings. Then, ask a question, provide the perfect answer (ground truth), and see how each setup performs.
        You can test the current config or run the **All Permutations Test**.
        """)
        display_evaluation_interface()

    # Display about information at the bottom (optional)
    st.markdown("---") # Separator before the expander
    with st.expander("📚 About Chunking Strategies"):
        try:
            chunking_strategies = ChunkingStrategyFactory.get_all_strategies()
            for strategy_name, strategy in chunking_strategies.items():
                # Use markdown for potential formatting in descriptions
                st.markdown(f"##### {strategy_name.replace('_', ' ').title()}")
                st.markdown(strategy.description)
                st.markdown("---") # Separator between strategies
        except Exception as e:
            st.warning(f"Could not load chunking strategy descriptions: {e}")


if __name__ == "__main__":
    main()