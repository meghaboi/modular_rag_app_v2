import streamlit as st
import pandas as pd
from typing import Dict, Any
from ..services.pipeline_service import run_pipeline_with_config, run_all_permutations
from ..core.app import get_csv_download_link

def display_evaluation_interface():
    """Display the evaluation interface"""
    st.title("RAG Pipeline Evaluation")
    
    # File upload
    uploaded_file = st.file_uploader("Upload a document", type=['txt', 'pdf', 'docx'])
    if uploaded_file:
        file_path = save_uploaded_file(uploaded_file)
        st.session_state.file_path = file_path
        st.success(f"File uploaded: {uploaded_file.name}")
    
    # Evaluation parameters
    col1, col2 = st.columns(2)
    with col1:
        user_query = st.text_area("User Query", height=100)
        ground_truth = st.text_area("Ground Truth", height=100)
    
    with col2:
        chunk_size = st.number_input("Chunk Size", min_value=100, max_value=4000, value=1000)
        chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=1000, value=200)
        top_k = st.number_input("Top K", min_value=1, max_value=10, value=3)
        hybrid_alpha = st.slider("Hybrid Alpha", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    
    # Run evaluation
    if st.button("Run Evaluation"):
        if not st.session_state.file_path:
            st.error("Please upload a document first.")
            return
        if not user_query or not ground_truth:
            st.error("Please provide both user query and ground truth.")
            return
        
        with st.spinner("Running evaluation..."):
            results = run_all_permutations(
                st.session_state.file_path,
                user_query,
                ground_truth,
                chunk_size,
                chunk_overlap,
                top_k,
                hybrid_alpha,
                ChunkingStrategyType.HIERARCHICAL
            )
            
            st.session_state.permutation_results = results
            st.session_state.permutation_df = pd.DataFrame(results)
    
    # Display results
    if st.session_state.permutation_df is not None:
        st.subheader("Evaluation Results")
        st.dataframe(st.session_state.permutation_df)
        
        # Download results
        st.markdown(get_csv_download_link(st.session_state.permutation_df), unsafe_allow_html=True)
        
        # Display best configuration
        best_config = st.session_state.permutation_df.loc[
            st.session_state.permutation_df['score'].idxmax()
        ]
        st.subheader("Best Configuration")
        st.write(f"Embedding Model: {best_config['embedding_model']}")
        st.write(f"Vector Store: {best_config['vector_store']}")
        st.write(f"Reranker: {best_config['reranker']}")
        st.write(f"LLM: {best_config['llm']}")
        st.write(f"Score: {best_config['score']:.4f}") 