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

st.set_page_config(page_title="RAG Chat & Evaluation System", layout="wide")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'file_path' not in st.session_state:
    st.session_state.file_path = None
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'mode' not in st.session_state:
    st.session_state.mode = "chat"  # Default to chat mode
if 'permutation_results' not in st.session_state:
    st.session_state.permutation_results = None
if 'permutation_df' not in st.session_state:
    st.session_state.permutation_df = None
if 'is_settings_open' not in st.session_state:
    st.session_state.is_settings_open = False
if 'show_contexts' not in st.session_state:
    st.session_state.show_contexts = False
if 'api_key_status' not in st.session_state:
    st.session_state.api_key_status = {}

def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary location and return the path"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as temp:
        temp.write(uploaded_file.getvalue())
        return temp.name

def get_csv_download_link(df, filename="permutation_results.csv"):
    """Generate a download link for a pandas dataframe as CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href

def toggle_settings():
    st.session_state.is_settings_open = not st.session_state.is_settings_open

def toggle_mode():
    st.session_state.mode = "evaluation" if st.session_state.mode == "chat" else "chat"
    # Reset messages when switching modes
    st.session_state.messages = []

def toggle_contexts():
    st.session_state.show_contexts = not st.session_state.show_contexts

def check_api_keys(embedding_model_enum, vector_store_enum, reranker_enum, llm_enum):
    """Check if required API keys are available in environment"""
    api_keys_status = {}
    
    if embedding_model_enum == EmbeddingModelType.OPENAI or llm_enum in [LLMModelType.OPENAI_GPT35, LLMModelType.OPENAI_GPT4]:
        api_keys_status["OpenAI API Key"] = "Available" if os.getenv("OPENAI_API_KEY") else "Missing"
    
    if embedding_model_enum == EmbeddingModelType.COHERE or reranker_enum in [RerankerModelType.COHERE_V2, RerankerModelType.COHERE_V3, RerankerModelType.COHERE_MULTILINGUAL]:
        api_keys_status["Cohere API Key"] = "Available" if os.getenv("COHERE_API_KEY") else "Missing"
        
    if embedding_model_enum == EmbeddingModelType.GEMINI or llm_enum == LLMModelType.GEMINI:
        api_keys_status["Gemini API Key"] = "Available" if os.getenv("GEMINI_API_KEY") else "Missing"
        
    if llm_enum in [LLMModelType.CLAUDE_3_OPUS, LLMModelType.CLAUDE_37_SONNET]:
        api_keys_status["Anthropic API Key"] = "Available" if os.getenv("ANTHROPIC_API_KEY") else "Missing"
    
    if llm_enum in [LLMModelType.MISTRAL_LARGE, LLMModelType.MISTRAL_MEDIUM, LLMModelType.MISTRAL_SMALL] or embedding_model_enum == EmbeddingModelType.MISTRAL:
        api_keys_status["Mistral API Key"] = "Available" if os.getenv("MISTRAL_API_KEY") else "Missing"
            
    if embedding_model_enum == EmbeddingModelType.VOYAGE or reranker_enum in [RerankerModelType.VOYAGE, RerankerModelType.VOYAGE_2]:
        api_keys_status["Voyage AI API Key"] = "Available" if os.getenv("VOYAGE_API_KEY") else "Missing"
    
    st.session_state.api_key_status = api_keys_status
    
    # Return list of missing keys
    return [key for key, status in api_keys_status.items() if status == "Missing"]

def initialize_pipeline(file_path, embedding_model_enum, vector_store_enum, reranker_enum, llm_enum, 
                        chunking_strategy_enum, hybrid_alpha, chunk_size, chunk_overlap, top_k):
    """Initialize RAG pipeline with selected configuration"""
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
    with st.spinner("Indexing documents..."):
        pipeline.index_documents(file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    return pipeline

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
        
        chunking_strategy_instance = ChunkingStrategyFactory.get_strategy(chunking_strategy_enum.value)
        
        # Create and run RAG pipeline
        pipeline = RAGPipeline(
            embedding_model=embedding_model_instance,
            vector_store=vector_store_instance,
            reranker=reranker_instance,
            llm=llm_instance,
            top_k=top_k,
            chunking_strategy=chunking_strategy_instance
        )
        
        # Index documents
        pipeline.index_documents(file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # Process query
        start_time = time.time()
        response, contexts = pipeline.process_query(user_query)
        elapsed_time = time.time() - start_time
        
        # Run evaluation with RAGAS
        evaluator = EvaluatorFactory.create_evaluator(
            EvaluationBackendType.RAGAS,
            EvaluationMetricType.get_metrics_for_backend(EvaluationBackendType.RAGAS)
        )
        
        evaluation_results = evaluator.evaluate(
            query=user_query,
            response=response,
            contexts=contexts,
            ground_truth=ground_truth if ground_truth else None
        )
        
        # Calculate average score
        avg_score = sum(evaluation_results.values()) / len(evaluation_results) if evaluation_results else 0
        
        return {
            "embedding_model": embedding_model_enum.value,
            "vector_store": vector_store_enum.value,
            "reranker": reranker_enum.value,
            "llm_model": llm_enum.value,
            "chunking_strategy": chunking_strategy_enum.value,
            "response": response,
            "evaluation_scores": evaluation_results,
            "avg_score": avg_score,
            "elapsed_time": elapsed_time,
            "contexts": contexts
        }
    except Exception as e:
        st.error(f"Error with configuration - {embedding_model_enum.value}, {vector_store_enum.value}, {reranker_enum.value}, {llm_enum.value}: {str(e)}")
        return {
            "embedding_model": embedding_model_enum.value,
            "vector_store": vector_store_enum.value,
            "reranker": reranker_enum.value,
            "llm_model": llm_enum.value,
            "chunking_strategy": chunking_strategy_enum.value,
            "response": f"ERROR: {str(e)}",
            "evaluation_scores": {},
            "avg_score": 0,
            "elapsed_time": 0,
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
    
    # Get all possible combinations but exclude specific models
    embedding_models = [
        EmbeddingModelType.OPENAI, EmbeddingModelType.GEMINI, EmbeddingModelType.MISTRAL
    ]
    
    vector_stores = [
        VectorStoreType.FAISS, VectorStoreType.CHROMA
    ]
    
    rerankers = list(RerankerModelType)
    
    llm_models = [
        LLMModelType.CLAUDE_37_SONNET, LLMModelType.GEMINI
    ]
    
    all_permutations = list(itertools.product(embedding_models, vector_stores, rerankers, llm_models))
    
    progress_bar = st.progress(0)
    
    results = []
    for i, (embedding_model, vector_store, reranker, llm_model) in enumerate(all_permutations):
        st.info(f"Running permutation {i+1}/{len(all_permutations)}: {embedding_model.value}, {vector_store.value}, {reranker.value}, {llm_model.value}")
        
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
        
        # Add individual metric scores to the result dictionary (excluding answer_relevancy)
        for metric, score in result["evaluation_scores"].items():
            if metric != "answer_relevance":  # Skip answer_relevancy metric
                result[f"metric_{metric}"] = score
        
        results.append(result)
        
        # Update progress bar
        progress_bar.progress((i + 1) / len(all_permutations))
    
    # Convert to dataframe
    results_df = pd.DataFrame(results)
    
    # Extract only the necessary columns for CSV (excluding answer_relevancy)
    csv_columns = [
        "embedding_model", "vector_store", "reranker", "llm_model", 
        "chunking_strategy", "avg_score", "elapsed_time"
    ] + [col for col in results_df.columns if col.startswith("metric_") and not col.endswith("answer_relevancy")]
    
    csv_df = results_df[csv_columns]
    
    return csv_df, results

def display_chat_interface():
    st.header("RAG Chat")
    
    # Display message history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Display contexts if available and show_contexts is True
            if st.session_state.show_contexts and message["role"] == "assistant" and "contexts" in message:
                with st.expander("View Retrieved Contexts"):
                    for i, context in enumerate(message["contexts"]):
                        st.markdown(f"**Context {i+1}:**")
                        st.text(context)
    
    # User input
    user_query = st.chat_input("Ask a question about your documents")
    
    if user_query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_query)
        
        # Check if pipeline is initialized
        if st.session_state.pipeline is None:
            with st.chat_message("assistant"):
                st.write("Please upload a document and configure the RAG pipeline first in the Settings panel.")
            st.session_state.messages.append(
                {"role": "assistant", "content": "Please upload a document and configure the RAG pipeline first in the Settings panel."}
            )
            return
        
        # Process the query and get response
        with st.spinner("Thinking..."):
            try:
                start_time = time.time()
                response, contexts = st.session_state.pipeline.process_query(user_query)
                elapsed_time = time.time() - start_time
                
                # Display assistant response with spinning animation
                with st.chat_message("assistant"):
                    st.write(response)
                    st.write(f"_Processing time: {elapsed_time:.2f} seconds_")
                    
                    # Display contexts if enabled
                    if st.session_state.show_contexts:
                        with st.expander("View Retrieved Contexts"):
                            for i, context in enumerate(contexts):
                                st.markdown(f"**Context {i+1}:**")
                                st.text(context)
                
                # Add assistant message to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "contexts": contexts,
                    "elapsed_time": elapsed_time
                })
                
            except Exception as e:
                with st.chat_message("assistant"):
                    st.error(f"Error processing query: {str(e)}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"Error processing query: {str(e)}"}
                )

def display_evaluation_interface():
    st.header("RAG Evaluation")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Evaluation Query")
        user_query = st.text_area("Enter your question:", height=100)
        
        st.subheader("Ground Truth")
        ground_truth = st.text_area("Enter ground truth answer for evaluation:", height=100)
        
        # Create two buttons side by side
        process_button = st.button("Process Single Query")
        permutation_button = st.button("Run All Permutations")
        
    with col2:
        if st.session_state.permutation_results is not None:
            st.subheader("Permutation Results")
            
            # Display download link for CSV
            st.markdown(get_csv_download_link(st.session_state.permutation_df), unsafe_allow_html=True)
            
            # Show top 5 results sorted by average score
            top_results = st.session_state.permutation_df.sort_values('avg_score', ascending=False).head(5)
            st.dataframe(top_results)
            
            # Allow user to select a specific configuration to view details
            all_configs = [f"{row['embedding_model']} + {row['vector_store']} + {row['reranker']} + {row['llm_model']}" 
                         for _, row in st.session_state.permutation_df.iterrows()]
            
            selected_config_index = st.selectbox("Select Configuration", range(len(all_configs)), format_func=lambda i: all_configs[i])
            
            if selected_config_index is not None:
                selected_result = st.session_state.permutation_results[selected_config_index]
                
                with st.expander("Configuration Details", expanded=True):
                    st.write(f"Embedding Model: {selected_result['embedding_model']}")
                    st.write(f"Vector Store: {selected_result['vector_store']}")
                    st.write(f"Reranker: {selected_result['reranker']}")
                    st.write(f"LLM Model: {selected_result['llm_model']}")
                    st.write(f"Processing Time: {selected_result['elapsed_time']:.2f} seconds")
                    
                    st.subheader("Response")
                    st.write(selected_result['response'])
                    
                    # Removed the nested expander and replaced with a subheader
                    st.subheader("Retrieved Contexts")
                    for i, context in enumerate(selected_result['contexts']):
                        st.markdown(f"**Context {i+1}:**")
                        st.text(context)
                    
                    st.subheader("Evaluation Scores")
                    for metric, score in selected_result['evaluation_scores'].items():
                        st.write(f"{metric.replace('_', ' ').title()}: {score:.2f}")
                    
                    st.metric("Overall Score", f"{selected_result['avg_score']:.2f}/5.0")
    
    # Process single query or run permutations
    if st.session_state.file_path is not None and process_button and user_query:
        # Get configuration from settings
        embedding_model_enum = EmbeddingModelType.from_string(st.session_state.embedding_model)
        vector_store_enum = VectorStoreType.from_string(st.session_state.vector_store)  
        reranker_enum = RerankerModelType.from_string(st.session_state.reranker)
        llm_enum = LLMModelType.from_string(st.session_state.llm_model)
        chunking_strategy_enum = ChunkingStrategyType.from_string(st.session_state.chunking_strategy)
        
        # Check for missing API keys
        missing_keys = check_api_keys(embedding_model_enum, vector_store_enum, reranker_enum, llm_enum)
        if missing_keys:
            st.error(f"Cannot proceed. Missing required API keys: {', '.join(missing_keys)}")
            return
        
        # Process query
        with st.spinner("Evaluating single query..."):
            result = run_pipeline_with_config(
                file_path=st.session_state.file_path,
                user_query=user_query,
                ground_truth=ground_truth,
                embedding_model_enum=embedding_model_enum,
                vector_store_enum=vector_store_enum,
                reranker_enum=reranker_enum,
                llm_enum=llm_enum,
                chunking_strategy_enum=chunking_strategy_enum,
                hybrid_alpha=st.session_state.hybrid_alpha,
                chunk_size=st.session_state.chunk_size,
                chunk_overlap=st.session_state.chunk_overlap,
                top_k=st.session_state.top_k
            )
        
        # Display results
        with st.expander("Results", expanded=True):
            st.subheader("Answer")
            st.write(result["response"])
            st.write(f"Processing time: {result['elapsed_time']:.2f} seconds")
            
            # Replace nested expander with a subheader
            st.subheader("Retrieved Contexts")
            for i, context in enumerate(result["contexts"]):
                st.markdown(f"**Context {i+1}:**")
                st.text(context)
            
            st.subheader("Evaluation Scores")
            for metric, score in result["evaluation_scores"].items():
                st.write(f"{metric.replace('_', ' ').title()}: {score:.2f}")
            
            st.metric("Overall Score", f"{result['avg_score']:.2f}/5.0")
    
    elif st.session_state.file_path is not None and permutation_button and user_query:
        # Check for all possible API keys before running permutations
        all_api_keys = {
            "OpenAI API Key": "OPENAI_API_KEY",
            "Cohere API Key": "COHERE_API_KEY",
            "Gemini API Key": "GEMINI_API_KEY",
            "Anthropic API Key": "ANTHROPIC_API_KEY",
            "Mistral API Key": "MISTRAL_API_KEY",
            "Voyage AI API Key": "VOYAGE_API_KEY"
        }
        
        missing_keys = []
        for key_name, env_var in all_api_keys.items():
            if not os.getenv(env_var):
                missing_keys.append(key_name)
        
        if missing_keys:
            st.warning(f"Missing API keys: {', '.join(missing_keys)}. Some permutations may fail.")
        
        # Run permutations
        chunking_strategy_enum = ChunkingStrategyType.from_string(st.session_state.chunking_strategy)
        
        with st.spinner("Running all permutations. This may take a while..."):
            results_df, all_results = run_all_permutations(
                file_path=st.session_state.file_path,
                user_query=user_query,
                ground_truth=ground_truth,
                chunk_size=st.session_state.chunk_size,
                chunk_overlap=st.session_state.chunk_overlap,
                top_k=st.session_state.top_k,
                hybrid_alpha=st.session_state.hybrid_alpha,
                chunking_strategy_enum=chunking_strategy_enum
            )
        
        # Store results in session state
        st.session_state.permutation_df = results_df
        st.session_state.permutation_results = all_results
        
        # Show success message
        st.success("All permutations completed!")
        st.rerun()
        
    elif (process_button or permutation_button) and not st.session_state.file_path:
        st.warning("Please upload a document in the Settings panel first.")
    elif (process_button or permutation_button) and not user_query:
        st.warning("Please enter a query.")

def display_settings_panel():
    st.sidebar.title("Settings")
    
    # Reset chat button
    if st.sidebar.button("Reset Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    # Mode selector (Chat vs Evaluation)
    mode_options = {"Chat Mode": "chat", "Evaluation Mode": "evaluation"}
    selected_mode = st.sidebar.radio(
        "Select Mode", 
        options=list(mode_options.keys()),
        index=0 if st.session_state.mode == "chat" else 1
    )
    
    if mode_options[selected_mode] != st.session_state.mode:
        st.session_state.mode = mode_options[selected_mode]
        st.session_state.messages = []
        st.rerun()
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload Knowledge Base (.txt file)", type=['txt'])
    
    if uploaded_file and (st.session_state.file_path is None or 
                          uploaded_file.name != getattr(st.session_state, 'last_uploaded_filename', None)):
        st.session_state.file_path = save_uploaded_file(uploaded_file)
        st.session_state.last_uploaded_filename = uploaded_file.name
        # Reset pipeline when new file is uploaded
        st.session_state.pipeline = None
        st.rerun()
    
    # Display system status
    with st.sidebar.expander("System Status", expanded=True):
        if st.session_state.file_path:
            st.success("✅ Document loaded")
        else:
            st.error("❌ No document loaded")
            
        if st.session_state.pipeline:
            st.success("✅ RAG pipeline initialized")
        else:
            st.error("❌ RAG pipeline not initialized")
    
    # Configuration options
    with st.sidebar.expander("Configuration", expanded=True):
        # Model selection using enums
        embedding_model = st.selectbox(
            "Embedding Model",
            options=EmbeddingModelType.list(),
            index=0,
            key="embedding_model"
        )
        
        reranker_model = st.selectbox(
            "Re-ranker Model",
            options=RerankerModelType.list(),
            index=0,
            key="reranker"
        )
        
        llm_model = st.selectbox(
            "LLM Model",
            options=LLMModelType.list(),
            index=0,
            key="llm_model"
        )
        
        vector_store = st.selectbox(
            "Vector Store",
            options=VectorStoreType.list(),
            index=0,
            key="vector_store"
        )
        
        # Chunking strategy selection
        chunking_strategy = st.selectbox(
            "Chunking Strategy",
            options=ChunkingStrategyType.list(),
            index=0,
            key="chunking_strategy"
        )
        
        # Show description of selected chunking strategy
        selected_strategy_enum = ChunkingStrategyType.from_string(chunking_strategy)
        selected_strategy_obj = ChunkingStrategyFactory.get_strategy(selected_strategy_enum.value)
        if selected_strategy_obj:
            st.info(selected_strategy_obj.description)
        
    # Advanced settings expander
    with st.sidebar.expander("Advanced Settings"):
        # Hybrid search settings
        selected_vector_store_enum = VectorStoreType.from_string(vector_store)
        hybrid_alpha = 0.5
        if selected_vector_store_enum == VectorStoreType.HYBRID:
            st.info("Hybrid search combines dense vector search with keyword-based BM25 search")
            hybrid_alpha = st.slider(
                "Vector search weight (alpha)", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.5,
                key="hybrid_alpha",
                help="Higher values favor vector search, lower values favor BM25 keyword search"
            )
            st.write(f"BM25 weight: {1 - hybrid_alpha:.2f}")
        else:
            # Set default even if not displayed
            st.session_state.hybrid_alpha = 0.5
        
        # Chunking parameters
        chunk_size = st.slider("Chunk Size (characters)", 500, 5000, 1000, key="chunk_size")
        chunk_overlap = st.slider("Chunk Overlap (characters)", 0, 1000, 200, key="chunk_overlap")
        
        top_k = st.slider("Number of documents to retrieve", 1, 10, 3, key="top_k")
        
        # In chat mode, option to show contexts
        if st.session_state.mode == "chat":
            show_contexts = st.checkbox("Show retrieved contexts", value=st.session_state.show_contexts)
            if show_contexts != st.session_state.show_contexts:
                st.session_state.show_contexts = show_contexts
                st.rerun()
    
    # API keys status expander
    with st.sidebar.expander("API Keys Status"):
        embedding_model_enum = EmbeddingModelType.from_string(embedding_model)
        vector_store_enum = VectorStoreType.from_string(vector_store)
        reranker_enum = RerankerModelType.from_string(reranker_model)
        llm_enum = LLMModelType.from_string(llm_model)
        
        missing_keys = check_api_keys(embedding_model_enum, vector_store_enum, reranker_enum, llm_enum)
        
        # Display API key status
        if st.session_state.api_key_status:
            for key_name, status in st.session_state.api_key_status.items():
                status_color = "green" if status == "Available" else "red"
                st.markdown(f"{key_name}: <span style='color:{status_color}'>{status}</span>", unsafe_allow_html=True)
            
            # Show help text for missing keys
            if missing_keys:
                st.warning(f"Missing API keys required for selected models.")
                st.info("Add them to your .env file with format:\n```\nOPENAI_API_KEY=sk-...\nCOHERE_API_KEY=...\n```")
    
    # Initialize pipeline button
    if st.sidebar.button("Initialize RAG Pipeline"):
        if st.session_state.file_path is None:
            st.sidebar.error("Please upload a document first")
        else:
            # Check for missing API keys
            embedding_model_enum = EmbeddingModelType.from_string(embedding_model)
            vector_store_enum = VectorStoreType.from_string(vector_store)
            reranker_enum = RerankerModelType.from_string(reranker_model)
            llm_enum = LLMModelType.from_string(llm_model)
            
            missing_keys = check_api_keys(embedding_model_enum, vector_store_enum, reranker_enum, llm_enum)
            if missing_keys:
                st.sidebar.error(f"Cannot initialize. Missing API keys: {', '.join(missing_keys)}")
            else:
                # Initialize pipeline
                try:
                    chunking_strategy_enum = ChunkingStrategyType.from_string(chunking_strategy)
                    
                    with st.spinner("Initializing RAG pipeline..."):
                        st.session_state.pipeline = initialize_pipeline(
                            file_path=st.session_state.file_path,
                            embedding_model_enum=embedding_model_enum,
                            vector_store_enum=vector_store_enum,
                            reranker_enum=reranker_enum,
                            llm_enum=llm_enum,
                            chunking_strategy_enum=chunking_strategy_enum,
                            hybrid_alpha=st.session_state.hybrid_alpha,
                            chunk_size=st.session_state.chunk_size,
                            chunk_overlap=st.session_state.chunk_overlap,
                            top_k=st.session_state.top_k
                        )
                    
                    st.sidebar.success("Pipeline initialized successfully!")
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Error initializing pipeline: {str(e)}")

def main():
    # Display page title and description
    st.title("RAG System")
    
    if st.session_state.mode == "chat":
        st.markdown("""
        This is a Retrieval-Augmented Generation (RAG) chat system. Upload a document, configure the pipeline, 
        and start asking questions about your document.
        """)
    else:
        st.markdown("""
        This is a RAG Evaluation system. Upload a document, configure the pipeline, and evaluate different
        model combinations with your queries and ground truth answers.
        """)
    
    # Display settings panel
    display_settings_panel()
    
    # Display appropriate interface based on selected mode
    if st.session_state.mode == "chat":
        display_chat_interface()
    else:
        display_evaluation_interface()
    
    # Display about information at the bottom
    with st.expander("About Chunking Strategies"):
        chunking_strategies = ChunkingStrategyFactory.get_all_strategies()
        for strategy_name, strategy in chunking_strategies.items():
            st.subheader(strategy_name)
            st.markdown(strategy.description)

if __name__ == "__main__":
    main()