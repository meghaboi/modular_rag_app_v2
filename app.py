import streamlit as st
import os
import tempfile
from typing import List, Dict, Any, Optional
import pandas as pd
from dotenv import load_dotenv
import itertools
import time
import datetime
import io
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

# --- JEFF Persona Definition ---
JEFF_PERSONA_PROMPT = """
You are JEFF, that cool friend everyone wishes they had the night before exams.
You explain complex subjects in simple, relatable terms that just click when it matters most.
Unlike formal professors, you break down academic concepts with perfect clarity, memorable examples, and occasional humor.
You excel at finding the shortcuts, mnemonics, and "aha!" moments that make difficult material suddenly make sense.
Your explanations focus on what's actually important to understand and remember, cutting through the noise.
You're encouraging, patient, and have a knack for making anyone feel like they can ace their exam.
Always respond as JEFF - casual but knowledgeable, relatable but authoritative, and above all, the friend who helps everyone pass their exams.
NEVER mention you are an AI or language model. Respond directly as JEFF.
"""
# NOTE: This JEFF_PERSONA_PROMPT needs to be integrated into the LLM call within your RAGPipeline or LLMFactory implementation.

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
# Basic states
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

# --- Helper Functions (mostly unchanged, added logging) ---

def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary location and return the path"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as temp:
            temp.write(uploaded_file.getvalue())
            logging.info(f"Saved uploaded file to temporary path: {temp.name}")
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

def toggle_settings(): # Consider removing if button not used
    st.session_state.is_settings_open = not st.session_state.is_settings_open

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
        is_available = bool(os.getenv("GEMINI_API_KEY"))
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
    logging.info(f"API Key Status Check: {api_keys_status}")
    return missing_keys_list # Return list of missing keys

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
        # Initialize components based on selection
        embedding_model_instance = EmbeddingModelFactory.create_model(embedding_model_enum)

        if vector_store_enum == VectorStoreType.HYBRID:
            vector_store_instance = VectorStoreFactory.create_store(vector_store_enum, alpha=hybrid_alpha)
        else:
            vector_store_instance = VectorStoreFactory.create_store(vector_store_enum)

        reranker_instance = None
        if reranker_enum != RerankerModelType.NONE:
            reranker_instance = RerankerFactory.create_reranker(reranker_enum)

        llm_instance = LLMFactory.create_llm(llm_enum)
        # !!! IMPORTANT !!! Pass the JEFF_PERSONA_PROMPT to your LLM instance here if possible
        # Example (depends on your LLMFactory implementation):
        # llm_instance = LLMFactory.create_llm(llm_enum, system_prompt=JEFF_PERSONA_PROMPT)

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

# --- Run Single Config & Permutations (largely unchanged, added logging) ---

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
        # !!! IMPORTANT !!! Pass the JEFF_PERSONA_PROMPT here too for consistency if needed by your factory
        # llm_instance = LLMFactory.create_llm(llm_enum, system_prompt=JEFF_PERSONA_PROMPT)
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
                if evaluation_results:
                   avg_score = sum(evaluation_results.values()) / len(evaluation_results)
                logging.info(f"Evaluation scores: {evaluation_results}")
            except Exception as eval_e:
                logging.error(f"Error during evaluation for config {config_str}: {eval_e}", exc_info=True)
                st.warning(f"Evaluation failed for this configuration: {eval_e}")
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
            "evaluation_scores": evaluation_results,
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
            "evaluation_scores": {},
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
        progress_bar.progress((i + 1) / num_permutations, text=progress_text)

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
        if "evaluation_scores" in result and result["evaluation_scores"]:
             for metric, score in result["evaluation_scores"].items():
                 result[f"metric_{metric}"] = score # Keep all metrics for detailed view

        results.append(result)

    end_permutations_time = time.time()
    total_time = end_permutations_time - start_permutations_time
    logging.info(f"All {num_permutations} permutations completed in {total_time:.2f} seconds.")
    progress_bar.progress(1.0, text="Permutations complete!")

    # Convert to dataframe
    results_df = pd.DataFrame(results)

    # Define columns for CSV export (keep all relevant info)
    base_columns = ["embedding_model", "vector_store", "reranker", "llm_model", "chunking_strategy", "avg_score", "elapsed_time"]
    metric_columns = sorted([col for col in results_df.columns if col.startswith("metric_")])
    csv_columns = base_columns + metric_columns + ["response"] # Include response in CSV for review

    # Ensure all expected columns exist, add if missing (e.g., if evaluation failed)
    for col in csv_columns:
        if col not in results_df.columns:
            results_df[col] = None # Or pd.NA

    csv_df = results_df[csv_columns]

    return csv_df, results # Return both DF and raw results list

# --- UI Display Functions ---

def display_chat_interface():
    st.header("💬 Chat with JEFF")
    st.markdown("Hey! Got questions about your textbook? Lay 'em on me. I'll break it down for ya.")

    # Initialize welcome message if chat is empty
    if not st.session_state.messages:
         welcome_msg = "Alright, let's get this study session started! What's on your mind?"
         st.session_state.messages.append({"role": "assistant", "content": welcome_msg})

    # Display message history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            # Display contexts if available and show_contexts is True
            if st.session_state.show_contexts and message["role"] == "assistant" and "contexts" in message and message["contexts"]:
                with st.expander("🧠 Check out the textbook bits I used:"):
                    for i, context in enumerate(message["contexts"]):
                        st.markdown(f"**Snippet {i+1}:**")
                        st.text(context)
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
            with st.chat_message("assistant"):
                st.warning("Whoa there! Looks like we haven't loaded your textbook into my brain yet. Upload it and hit 'Initialize' in the sidebar first!")
            st.session_state.messages.append(
                {"role": "assistant", "content": "Whoa there! Looks like we haven't loaded your textbook into my brain yet. Upload it and hit 'Initialize' in the sidebar first!"}
            )
            st.stop() # Stop execution for this run if pipeline isn't ready

        # Process the query and get response
        with st.spinner("JEFF's thinking... 🤔"):
            try:
                start_time = time.time()
                logging.info("Processing query with RAG pipeline...")
                # Pass the user query to the pipeline
                response, contexts = st.session_state.pipeline.process_query(user_query)
                elapsed_time = time.time() - start_time
                logging.info(f"Query processed successfully in {elapsed_time:.2f}s.")

                # Display assistant response
                with st.chat_message("assistant"):
                    st.write(response)
                    st.write(f"_(JEFF cooked that up in {elapsed_time:.2f} seconds)_")

                    # Display contexts if enabled
                    if st.session_state.show_contexts and contexts:
                         with st.expander("🧠 Check out the textbook bits I used:"):
                            for i, context in enumerate(contexts):
                                st.markdown(f"**Snippet {i+1}:**")
                                st.text(context)

                # Add assistant message to chat history including context and time
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "contexts": contexts,
                    "elapsed_time": elapsed_time
                })
                st.rerun() # Rerun to clear the input box and show the latest message properly

            except Exception as e:
                logging.error(f"Error processing query: {e}", exc_info=True)
                with st.chat_message("assistant"):
                    st.error(f"Oof, hit a snag trying to answer that. Maybe try rephrasing? Error: {str(e)}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"Oof, hit a snag trying to answer that. Maybe try rephrasing? Error: {str(e)}"}
                )

def display_evaluation_interface():
    st.header("🧪 RAG Evaluation Mode")
    st.markdown("Let's test out different setups. Give me a question and the perfect answer (ground truth) to see how well various RAG configurations perform.")

    # Ensure pipeline is initialized before allowing evaluation actions
    if st.session_state.pipeline is None and st.session_state.file_path:
        st.warning("💡 Looks like you've uploaded a document, but the RAG pipeline isn't active yet. Please hit 'Initialize RAG Pipeline' in the sidebar using the default settings before running evaluations.")
    elif not st.session_state.file_path:
         st.warning("💡 Upload a document first using the sidebar!")


    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Evaluation Inputs")
        user_query = st.text_area("Enter your question:", height=100, key="eval_query")
        ground_truth = st.text_area("Enter the ideal 'ground truth' answer:", height=100, key="eval_ground_truth")
        st.info("Providing ground truth enables detailed RAGAS evaluation scores.")

        # Buttons - disable if pipeline not ready or no file
        disable_buttons = st.session_state.pipeline is None or not st.session_state.file_path
        process_button = st.button("Evaluate Current Config", disabled=disable_buttons)
        permutation_button = st.button("Run All Permutations Test", disabled=disable_buttons)

    # Process single query or run permutations (logic moved outside col1)
    if process_button and user_query:
        if not st.session_state.file_path:
            st.warning("Please upload a document first.")
        elif st.session_state.pipeline is None:
             st.warning("Pipeline not initialized. Please initialize from the sidebar.")
        else:
            # Get current configuration from session state (used by the initialized pipeline)
            # We don't need to re-read from widgets as the pipeline holds the active config
            # However, we DO need the enum types for the API key check
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
                                if evaluation_results:
                                    avg_score = sum(evaluation_results.values()) / len(evaluation_results)
                                logging.info(f"Single config evaluation scores: {evaluation_results}")
                             except Exception as eval_e:
                                logging.error(f"Evaluation failed for single config run: {eval_e}", exc_info=True)
                                st.warning(f"Evaluation metrics failed: {eval_e}")
                        else:
                             logging.warning("No ground truth provided for single config run, skipping evaluation metrics.")


                        # --- Display Results Directly ---
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

                        if evaluation_results:
                             st.subheader("Evaluation Scores")
                             cols = st.columns(len(evaluation_results))
                             i = 0
                             for metric, score in evaluation_results.items():
                                 with cols[i]:
                                     st.metric(label=metric.replace('_', ' ').title(), value=f"{score:.2f}")
                                 i += 1
                             st.metric("Overall Average Score", f"{avg_score:.2f}") # Display average if scores exist
                        elif ground_truth:
                             st.warning("Evaluation scores could not be calculated.")
                        else:
                             st.info("Provide ground truth to see evaluation scores.")

                    except Exception as e:
                        logging.error(f"Error running single evaluation: {e}", exc_info=True)
                        st.error(f"Error processing evaluation: {str(e)}")

    elif permutation_button and user_query:
        if not st.session_state.file_path:
            st.warning("Please upload a document first.")
        elif st.session_state.pipeline is None:
             st.warning("Pipeline not initialized. Please initialize from the sidebar (this ensures the document is processed).")
        else:
            # Check for potentially needed API keys BEFORE starting permutations
            # This is a broad check; individual runs might skip if specific keys are missing
            st.info("Checking for API keys potentially needed for permutations...")
            # Check keys needed by the permutation list defined in run_all_permutations
            perm_emb = [EmbeddingModelType.OPENAI, EmbeddingModelType.GEMINI, EmbeddingModelType.MISTRAL]
            perm_rerank = list(RerankerModelType)
            perm_llm = [LLMModelType.CLAUDE_37_SONNET, LLMModelType.GEMINI]
            
            potential_missing = set()
            for e, r, l in itertools.product(perm_emb, perm_rerank, perm_llm):
                # Use dummy vector store as it doesn't require keys
                potential_missing.update(check_api_keys(e, VectorStoreType.FAISS, r, l)) 
                
            if potential_missing:
                st.warning(f"Heads up! Some permutations might fail. Missing potential API keys: {', '.join(potential_missing)}. Make sure they are in your .env file if needed.")
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
            st.rerun() # Rerun to update the display in col2

    elif (process_button or permutation_button) and not user_query:
        st.warning("⚠️ Please enter a question to evaluate.")


    # Display Permutation Results in Column 2 (if available)
    with col2:
        if st.session_state.permutation_df is not None and not st.session_state.permutation_df.empty:
            st.subheader("Permutation Results Summary")

            # Display download link for CSV
            st.markdown(get_csv_download_link(st.session_state.permutation_df), unsafe_allow_html=True)

            # Show top results sorted by average score (handle cases with no score)
            results_to_display = st.session_state.permutation_df.copy()
            # Convert avg_score to numeric, coercing errors to NaN, then fill NaN with a low value for sorting
            results_to_display['avg_score_numeric'] = pd.to_numeric(results_to_display['avg_score'], errors='coerce').fillna(-1)
            top_results = results_to_display.sort_values('avg_score_numeric', ascending=False).head(10) # Show top 10

            # Select columns to display in the summary table
            display_cols = ["embedding_model", "vector_store", "reranker", "llm_model", "chunking_strategy", "avg_score", "elapsed_time"]
            # Add metric columns if they exist
            metric_cols_exist = [col for col in results_to_display.columns if col.startswith("metric_")]
            display_cols.extend(metric_cols_exist)

            st.dataframe(top_results[display_cols]) # Display selected columns

            st.markdown("---")
            st.subheader("Explore Individual Results")

            # Allow user to select a specific configuration to view details
            # Create readable labels for the selectbox
            config_labels = []
            for index, row in st.session_state.permutation_df.iterrows():
                 label = (f"{index}: {row['embedding_model']} / {row['vector_store']} / "
                          f"{row['reranker']} / {row['llm_model']} "
                          f"(Score: {row['avg_score']:.2f}, Time: {row['elapsed_time']:.1f}s)")
                 config_labels.append(label)

            # Use index as the selection key, format_func provides the display label
            selected_index = st.selectbox(
                 "Select Configuration to View Details:",
                 options=st.session_state.permutation_df.index, # Use DataFrame index
                 format_func=lambda index: config_labels[index] # Map index to label
            )


            if selected_index is not None and selected_index in st.session_state.permutation_df.index:
                # Retrieve the raw result dictionary using the selected index
                selected_result = st.session_state.permutation_results[selected_index]

                st.markdown(f"**Details for Configuration {selected_index}:**")
                st.markdown(f"**Models:** `{selected_result['embedding_model']} | {selected_result['vector_store']} | {selected_result['reranker']} | {selected_result['llm_model']}`")
                st.markdown(f"**Chunking:** `{selected_result['chunking_strategy']}`")
                st.write(f"**Processing Time:** {selected_result['elapsed_time']:.2f} seconds")

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

                # Display evaluation scores if they exist
                eval_scores = selected_result.get('evaluation_scores', {})
                if eval_scores:
                    st.subheader("Evaluation Scores")
                    cols = st.columns(len(eval_scores))
                    i = 0
                    for metric, score in eval_scores.items():
                         with cols[i]:
                             st.metric(label=metric.replace('_', ' ').title(), value=f"{score:.2f}")
                         i+=1
                    # Display average score if available in the raw result
                    if 'avg_score' in selected_result:
                         st.metric("Overall Average Score", f"{selected_result['avg_score']:.2f}")
                elif ground_truth: # Only show warning if ground truth was provided but scores are missing
                     st.warning("Evaluation scores are not available for this result (might have failed or been skipped).")
                else:
                     st.info("Provide ground truth during the permutation run to see evaluation scores.")
        elif st.session_state.permutation_results is not None: # Handle case where results exist but are empty
             st.info("Permutation run finished, but no results were generated (check logs for errors).")


def display_settings_panel():
    st.sidebar.image("https://www.nicepng.com/png/detail/972-9721863_raising-hand-icon-png.png", width=80) # Placeholder 'cool friend' icon
    st.sidebar.title("JEFF's Controls")

    # Mode selector (Chat vs Evaluation)
    mode_options = {"💬 Chat with JEFF": "chat", "🧪 Test Setups (Evaluation)": "evaluation"}
    current_mode_index = list(mode_options.values()).index(st.session_state.mode)

    selected_mode_label = st.sidebar.radio(
        "Select Mode",
        options=list(mode_options.keys()),
        index=current_mode_index,
        help="Switch between asking JEFF questions and testing different RAG configurations."
    )

    # Update mode if changed
    new_mode = mode_options[selected_mode_label]
    if new_mode != st.session_state.mode:
        st.session_state.mode = new_mode
        st.session_state.messages = [] # Reset chat
        st.session_state.pipeline = None # Reset pipeline on mode switch
        st.session_state.permutation_results = None # Clear eval results
        st.session_state.permutation_df = None
        logging.info(f"Mode changed to: {st.session_state.mode}. Resetting state.")
        st.rerun()

    # File upload
    st.sidebar.header("📚 Load Textbook")
    uploaded_file = st.sidebar.file_uploader("Upload your textbook (.txt file)", type=['txt'], key="file_uploader")

    if uploaded_file is not None:
        # Check if it's a new file upload
        if uploaded_file.name != st.session_state.get('last_uploaded_filename', None):
            logging.info(f"New file uploaded: {uploaded_file.name}")
            st.session_state.file_path = save_uploaded_file(uploaded_file)
            if st.session_state.file_path: # Only update if save was successful
                 st.session_state.last_uploaded_filename = uploaded_file.name
                 st.session_state.pipeline = None # Crucial: Reset pipeline when new file is uploaded
                 st.sidebar.success(f"'{uploaded_file.name}' loaded!")
                 logging.info("New file loaded, pipeline reset. Rerunning.")
                 # No automatic rerun here, let the auto-init logic handle it or user press button
            else:
                 st.sidebar.error("Failed to process uploaded file.")
                 # Keep old file path if save failed? Or set to None? Setting to None is safer.
                 st.session_state.file_path = None
                 st.session_state.last_uploaded_filename = None
                 st.session_state.pipeline = None


    # Display system status
    st.sidebar.header("🚦 System Status")
    with st.sidebar.container(): # Use container for better layout
        if st.session_state.file_path and os.path.exists(st.session_state.file_path):
            st.success(f"✅ Textbook loaded: {st.session_state.last_uploaded_filename}")
        else:
            st.error("❌ No textbook loaded")

        if st.session_state.pipeline:
            st.success("✅ JEFF is ready!")
        else:
            st.warning("⏳ JEFF needs setup (Initialize Pipeline)")


    # --- Configuration Options (Only show in Evaluation Mode) ---
    if st.session_state.mode == "evaluation":
        st.sidebar.markdown("---")
        st.sidebar.header("🛠️ Evaluation Config")
        st.sidebar.info("Adjust these settings to test different RAG setups in Evaluation Mode.")

        with st.sidebar.expander("Models & Storage", expanded=True):
            # Get current lists and find default indices
            embedding_options = EmbeddingModelType.list()
            reranker_options = RerankerModelType.list()
            llm_options = LLMModelType.list()
            vector_store_options = VectorStoreType.list()
            chunking_strategy_options = ChunkingStrategyType.list()

            try:
                emb_index = embedding_options.index(st.session_state.embedding_model)
            except ValueError: emb_index = 0 # Fallback
            try:
                rerank_index = reranker_options.index(st.session_state.reranker)
            except ValueError: rerank_index = 0 # Fallback (often 'none')
            try:
                llm_index = llm_options.index(st.session_state.llm_model)
            except ValueError: llm_index = 0 # Fallback
            try:
                vs_index = vector_store_options.index(st.session_state.vector_store)
            except ValueError: vs_index = 0 # Fallback
            try:
                cs_index = chunking_strategy_options.index(st.session_state.chunking_strategy)
            except ValueError: cs_index = 0 # Fallback

            # Use session state for default selection via index and key for updates
            st.session_state.embedding_model = st.selectbox(
                "Embedding Model", options=embedding_options, index=emb_index, key="sb_embedding_model"
            )
            st.session_state.reranker = st.selectbox(
                "Re-ranker Model", options=reranker_options, index=rerank_index, key="sb_reranker"
            )
            st.session_state.llm_model = st.selectbox(
                "LLM Model", options=llm_options, index=llm_index, key="sb_llm_model"
            )
            st.session_state.vector_store = st.selectbox(
                "Vector Store", options=vector_store_options, index=vs_index, key="sb_vector_store"
            )
            st.session_state.chunking_strategy = st.selectbox(
                "Chunking Strategy", options=chunking_strategy_options, index=cs_index, key="sb_chunking_strategy"
            )

            # Show description of selected chunking strategy
            selected_strategy_enum = ChunkingStrategyType.from_string(st.session_state.chunking_strategy)
            selected_strategy_obj = ChunkingStrategyFactory.get_strategy(selected_strategy_enum.value)
            if selected_strategy_obj:
                st.info(f"**{selected_strategy_enum.value}:** {selected_strategy_obj.description}")

        with st.sidebar.expander("Indexing & Retrieval"):
             # Use session state for default value and key for updates
             st.session_state.chunk_size = st.slider("Chunk Size (characters)", 500, 5000, st.session_state.chunk_size, key="sb_chunk_size")
             st.session_state.chunk_overlap = st.slider("Chunk Overlap (characters)", 0, 1000, st.session_state.chunk_overlap, key="sb_chunk_overlap")
             st.session_state.top_k = st.slider("Documents to Retrieve (Top K)", 1, 10, st.session_state.top_k, key="sb_top_k")

             # Hybrid search settings (only show if relevant)
             selected_vector_store_enum = VectorStoreType.from_string(st.session_state.vector_store)
             if selected_vector_store_enum == VectorStoreType.HYBRID:
                 st.info("Hybrid search combines vector similarity with keyword matching.")
                 st.session_state.hybrid_alpha = st.slider(
                     "Vector Search Weight (alpha)", min_value=0.0, max_value=1.0, value=st.session_state.hybrid_alpha, key="sb_hybrid_alpha",
                     help="1.0 = pure vector search, 0.0 = pure keyword search (BM25)"
                 )
                 st.write(f"Keyword Search Weight: {1 - st.session_state.hybrid_alpha:.2f}")
             else:
                 # Ensure default alpha is in session state even if slider not shown
                 if 'hybrid_alpha' not in st.session_state: st.session_state.hybrid_alpha = DEFAULT_HYBRID_ALPHA

    # --- Settings applicable to both modes ---
    st.sidebar.markdown("---")

    # Reset Chat Button
    if st.sidebar.button("Clear Chat History", key="clear_chat"):
        st.session_state.messages = []
        logging.info("Chat history cleared by user.")
        st.rerun()

    # Show Contexts Toggle (useful in chat mode primarily)
    if st.session_state.mode == "chat":
        show_contexts_now = st.sidebar.toggle(
            "Show JEFF's sources?",
            value=st.session_state.show_contexts,
            key="toggle_context_display",
            help="See the parts of the textbook JEFF used to answer."
         )
        if show_contexts_now != st.session_state.show_contexts:
             st.session_state.show_contexts = show_contexts_now
             st.rerun() # Rerun to apply the change immediately


    # --- API Keys Status (Always Visible) ---
    st.sidebar.markdown("---")
    with st.sidebar.expander("🔑 API Key Status"):
        # Check keys based on *currently selected* models in session state
        try:
             embedding_enum = EmbeddingModelType.from_string(st.session_state.embedding_model)
             vs_enum = VectorStoreType.from_string(st.session_state.vector_store)
             reranker_enum = RerankerModelType.from_string(st.session_state.reranker)
             llm_enum = LLMModelType.from_string(st.session_state.llm_model)
             check_api_keys(embedding_enum, vs_enum, reranker_enum, llm_enum) # Updates st.session_state.api_key_status
        except ValueError as e:
             st.error(f"Error checking API keys due to invalid model selection: {e}")

        if st.session_state.api_key_status:
            missing_keys_found = False
            for key_name, status in st.session_state.api_key_status.items():
                status_icon = "✅" if status == "Available" else "❌"
                status_color = "green" if status == "Available" else "red"
                st.markdown(f"{status_icon} {key_name}: <span style='color:{status_color};'>{status}</span>", unsafe_allow_html=True)
                if status == "Missing":
                     missing_keys_found = True

            if missing_keys_found:
                st.warning("Missing API keys needed for the current configuration.")
                st.info("Add required keys to a `.env` file in the app directory (e.g., `OPENAI_API_KEY=sk-...`). You might need to restart the app after adding keys.")
        else:
            st.info("No API keys are currently required for the selected configuration.")


    # --- Initialize Pipeline Button (Always Visible) ---
    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 Initialize JEFF", key="init_pipeline", help="Load the textbook with the current settings. Required before chatting or evaluating."):
        if not st.session_state.file_path or not os.path.exists(st.session_state.file_path):
            st.sidebar.error("Please upload a textbook first!")
        else:
            # Always use values from session state, which reflect defaults or evaluation settings
            try:
                 embedding_enum = EmbeddingModelType.from_string(st.session_state.embedding_model)
                 vs_enum = VectorStoreType.from_string(st.session_state.vector_store)
                 reranker_enum = RerankerModelType.from_string(st.session_state.reranker)
                 llm_enum = LLMModelType.from_string(st.session_state.llm_model)
                 cs_enum = ChunkingStrategyType.from_string(st.session_state.chunking_strategy)
            except ValueError as e:
                 st.sidebar.error(f"Invalid configuration selected: {e}")
                 st.stop() # Stop if config is bad

            # Check API keys for the selected configuration before initializing
            missing_keys = check_api_keys(embedding_enum, vs_enum, reranker_enum, llm_enum)
            if missing_keys:
                st.sidebar.error(f"Cannot initialize. Missing keys: {', '.join(missing_keys)}")
            else:
                # Initialize pipeline using values from session state
                with st.spinner("Warming up JEFF's brain... (Initializing RAG pipeline)"):
                    pipeline_instance = initialize_pipeline(
                        file_path=st.session_state.file_path,
                        embedding_model_enum=embedding_enum,
                        vector_store_enum=vs_enum,
                        reranker_enum=reranker_enum,
                        llm_enum=llm_enum,
                        chunking_strategy_enum=cs_enum,
                        hybrid_alpha=st.session_state.hybrid_alpha,
                        chunk_size=st.session_state.chunk_size,
                        chunk_overlap=st.session_state.chunk_overlap,
                        top_k=st.session_state.top_k
                    )

                if pipeline_instance:
                    st.sidebar.success("JEFF is initialized and ready!")
                    # Don't rerun here, allow the main page to update naturally
                else:
                    st.sidebar.error("Initialization failed. Check logs.")
    st.sidebar.markdown("---")

def attempt_automatic_initialization():
    """Tries to initialize the RAG pipeline automatically on startup if conditions are met."""
    if st.session_state.pipeline is None and st.session_state.file_path and os.path.exists(st.session_state.file_path):
        logging.info("Attempting automatic RAG pipeline initialization on startup.")
        st.info("Detected textbook. Trying to set up JEFF automatically with default settings...")

        # Use the DEFAULT enums for the check and initialization
        default_embedding_enum = DEFAULT_EMBEDDING_MODEL
        default_vs_enum = DEFAULT_VECTOR_STORE
        default_reranker_enum = DEFAULT_RERANKER_MODEL
        default_llm_enum = DEFAULT_LLM_MODEL
        default_cs_enum = DEFAULT_CHUNKING_STRATEGY

        # Check API keys specifically for the default configuration
        missing_keys = check_api_keys(default_embedding_enum, default_vs_enum, default_reranker_enum, default_llm_enum)

        if missing_keys:
            st.warning(f"Automatic setup skipped: Missing API keys required for default settings ({', '.join(missing_keys)}). Please add them to .env or initialize manually via the sidebar.")
            logging.warning(f"Automatic initialization skipped due to missing default API keys: {missing_keys}")
        else:
            logging.info("Required API keys for default config found. Proceeding with automatic initialization.")
            progress_bar = st.progress(0, text="JEFF is warming up... (Auto-initializing)")
            try:
                # Initialize using default values stored in session state
                pipeline_instance = initialize_pipeline(
                    file_path=st.session_state.file_path,
                    embedding_model_enum=default_embedding_enum,
                    vector_store_enum=default_vs_enum,
                    reranker_enum=default_reranker_enum,
                    llm_enum=default_llm_enum,
                    chunking_strategy_enum=default_cs_enum,
                    hybrid_alpha=st.session_state.hybrid_alpha, # Use session state defaults
                    chunk_size=st.session_state.chunk_size,     # Use session state defaults
                    chunk_overlap=st.session_state.chunk_overlap, # Use session state defaults
                    top_k=st.session_state.top_k              # Use session state defaults
                )
                progress_bar.progress(1.0, text="JEFF is ready!")
                time.sleep(1) # Keep message visible briefly
                progress_bar.empty() # Remove progress bar

                if pipeline_instance:
                    st.success("JEFF automatically initialized with default settings!")
                    logging.info("Automatic RAG pipeline initialization successful.")
                    st.rerun() # Rerun to update status indicators immediately
                else:
                    st.error("Automatic initialization failed. Try initializing manually from the sidebar.")
                    logging.error("Automatic RAG pipeline initialization failed.")

            except Exception as e:
                 progress_bar.empty()
                 st.error(f"Automatic initialization encountered an error: {e}. Try initializing manually.")
                 logging.error(f"Error during automatic initialization: {e}", exc_info=True)


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
        hit **Initialize JEFF**, and then ask me anything. I'll explain stuff in a way that actually makes sense. Let's ace this thing! 🚀
        """)
        display_chat_interface()
    else: # Evaluation mode
        st.markdown("""
        Alright, let's put different study strategies to the test! 🔬 Upload your textbook, then use the **Evaluation Config** in the sidebar
        to choose different models and settings. Ask a question, provide the perfect answer (ground truth), and see how each setup performs.
        You can test one config at a time or run the **All Permutations Test**.
        """)
        display_evaluation_interface()

    # Display about information at the bottom (optional)
    with st.expander("📚 About Chunking Strategies"):
        try:
            chunking_strategies = ChunkingStrategyFactory.get_all_strategies()
            for strategy_name, strategy in chunking_strategies.items():
                st.subheader(strategy_name.replace("_", " ").title())
                st.markdown(strategy.description)
        except Exception as e:
            st.warning(f"Could not load chunking strategy descriptions: {e}")


if __name__ == "__main__":
    main()