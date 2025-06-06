import streamlit as st
import logging
import os
from utils.subject_configs import (
    SUBJECT_CONFIGS,
    get_subject_config,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_VECTOR_STORE,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_CHUNKING_STRATEGY,
    DEFAULT_HYBRID_ALPHA,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K
)
from pipeline.subject_handler import update_rag_configuration
from pipeline.pipeline_utils import initialize_pipeline
from utils.enums import (
    EmbeddingModelType,
    RerankerModelType,
    LLMModelType,
    VectorStoreType,
    ChunkingStrategyType
)
from utils.utils import save_uploaded_file, check_api_keys
from models.embedding_models import EmbeddingModelFactory
from models.vector_stores import VectorStoreFactory
from models.rerankers import RerankerFactory
from models.llm_models import LLMFactory
from pipeline.rag_pipeline import RAGPipeline
from models.chunking_strategies import ChunkingStrategyFactory
from pipeline.summarizer_module import extract_main_points, generate_summary_for_point

def display_settings_panel():
    if "main_points" not in st.session_state:
        st.session_state.main_points = []
    if "current_summary" not in st.session_state:
        st.session_state.current_summary = ""
    if "point_extraction_llm_missing_keys" not in st.session_state:
        st.session_state.point_extraction_llm_missing_keys = False

    st.sidebar.image("https://i.postimg.cc/DfLpxwZJ/Chat-GPT-Image-May-7-2025-11-10-13-AM.png", width=80)
    st.sidebar.title("Ask-JEFF")

    subjects = list(SUBJECT_CONFIGS.keys())
    selected_subject = st.sidebar.selectbox(
        "Select Subject",
        subjects,
        index=subjects.index("general"),
        help="Choose the subject of your textbook for optimal RAG configuration"
    )

    if st.session_state.pipeline is not None and st.session_state.mode == "evaluation":
        update_rag_configuration(selected_subject, st.session_state.pipeline)

    mode_options = {"💬 Chat with JEFF": "chat", "🧪 Test Setups (Evaluation)": "evaluation"}
    current_mode = st.session_state.get('mode', 'chat')
    if current_mode not in mode_options.values():
        current_mode = 'chat'
        st.session_state.mode = 'chat'
    current_mode_index = list(mode_options.values()).index(current_mode)
    selected_mode_label = st.sidebar.radio(
        "Select Mode",
        options=list(mode_options.keys()),
        index=current_mode_index,
        key="mode_radio",
        help="Switch between chatting and testing configurations."
    )
    new_mode = mode_options[selected_mode_label]
    if new_mode != st.session_state.mode:
        st.session_state.mode = new_mode
        st.session_state.messages = []
        st.session_state.pipeline = None
        st.session_state.permutation_results = None
        st.session_state.permutation_df = None
        logging.info(f"Mode changed to: {st.session_state.mode}. Resetting state.")
        st.rerun()

    st.sidebar.header("📚 Load Textbook")
    uploaded_file = st.sidebar.file_uploader("Upload .txt or .pdf file", type=['txt', 'pdf'], key="file_uploader")
    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.get('last_uploaded_filename', None):
            logging.info(f"New file upload detected: {uploaded_file.name}")
            with st.spinner(f"Processing '{uploaded_file.name}'..."):
                st.session_state.file_path = save_uploaded_file(uploaded_file)
            if st.session_state.file_path:
                st.session_state.last_uploaded_filename = uploaded_file.name
                st.session_state.pipeline = None
                st.session_state.messages = []
                st.session_state.permutation_results = None
                st.session_state.permutation_df = None
                st.session_state.main_points = []
                st.session_state.current_summary = ""
                st.session_state.point_extraction_llm_missing_keys = False

                point_extraction_llm_type = LLMModelType.CLAUDE_37_SONNET

                current_embedding_model = EmbeddingModelType.from_string(
                    st.session_state.get('embedding_model', DEFAULT_EMBEDDING_MODEL.value))
                current_vector_store = VectorStoreType.from_string(
                    st.session_state.get('vector_store', DEFAULT_VECTOR_STORE.value))
                current_reranker = RerankerModelType.from_string(
                    st.session_state.get('reranker', DEFAULT_RERANKER_MODEL.value))

                missing_keys_for_extraction_check = check_api_keys(
                    embedding_model_enum=current_embedding_model,
                    vector_store_enum=current_vector_store,
                    reranker_enum=current_reranker,
                    llm_enum=point_extraction_llm_type
                )

                is_extraction_llm_key_missing = False
                if missing_keys_for_extraction_check:
                    if point_extraction_llm_type.name.startswith(
                            "CLAUDE") and "ANTHROPIC_API_KEY" in missing_keys_for_extraction_check:
                        is_extraction_llm_key_missing = True
                    elif point_extraction_llm_type.name.startswith(
                            "OPENAI") and "OPENAI_API_KEY" in missing_keys_for_extraction_check:
                        is_extraction_llm_key_missing = True
                    elif any(key.endswith("_API_KEY") for key in missing_keys_for_extraction_check):
                        is_extraction_llm_key_missing = True
                        expected_key_name = ""
                        if point_extraction_llm_type.name.startswith("CLAUDE"):
                            expected_key_name = "ANTHROPIC_API_KEY"
                        elif point_extraction_llm_type.name.startswith("OPENAI"):
                            expected_key_name = "OPENAI_API_KEY"
                        elif point_extraction_llm_type.name.startswith("GEMINI"):
                            expected_key_name = "GEMINI_API_KEY"
                        elif point_extraction_llm_type.name.startswith("MISTRAL"):
                            expected_key_name = "MISTRAL_API_KEY"

                        if expected_key_name and expected_key_name in missing_keys_for_extraction_check:
                            is_extraction_llm_key_missing = True
                        elif not expected_key_name and missing_keys_for_extraction_check:
                            pass

                if not is_extraction_llm_key_missing:
                    st.session_state.point_extraction_llm_missing_keys = False
                    try:
                        point_extraction_llm = LLMFactory.create_llm(point_extraction_llm_type)
                        with st.spinner("Extracting key points from document..."):
                            st.session_state.main_points = extract_main_points(st.session_state.file_path,
                                                                               point_extraction_llm)
                        if not st.session_state.main_points:
                            st.sidebar.warning("Could not extract key points automatically.")
                    except Exception as e:
                        st.sidebar.error(f"Error during point extraction: {str(e)}")
                        logging.error(f"Point extraction failed: {e}", exc_info=True)
                else:
                    st.session_state.point_extraction_llm_missing_keys = True
                    st.sidebar.warning(
                        f"API key for {point_extraction_llm_type.value} needed for automatic point extraction.")

                st.sidebar.success(f"'{uploaded_file.name}' loaded!")
                logging.info("New file loaded, reset state. Rerunning.")
                st.rerun()
            else:
                st.sidebar.error("Failed to process uploaded file.")
                st.session_state.file_path = None
                st.session_state.last_uploaded_filename = None
                st.session_state.pipeline = None

    st.sidebar.header("🚦 System Status")
    with st.sidebar.container(border=True):
        if st.session_state.file_path and os.path.exists(st.session_state.file_path):
            st.success(f"✅ Textbook: {st.session_state.last_uploaded_filename}")
        else:
            st.warning("⚠️ No textbook loaded")
        if st.session_state.pipeline:
            st.success("✅ JEFF is ready!")
        else:
            st.warning("⏳ JEFF needs setup (Initialize)")

    st.sidebar.markdown("---")

    st.sidebar.header("📄 Document Summary")
    if not st.session_state.get("file_path"):
        st.sidebar.caption("Upload a document to enable summarization.")
    elif st.session_state.get("point_extraction_llm_missing_keys"):
        st.sidebar.warning(f"API key for point extraction LLM is missing. Cannot display or summarize key points.")
    elif st.session_state.get("main_points"):
        selected_point = st.sidebar.selectbox(
            "Select Key Point to Summarize:",
            options=st.session_state.main_points,
            key="selected_main_point"
        )
        if st.sidebar.button("✨ Summarize Selected Point", key="summarize_point_button"):
            if not st.session_state.get("pipeline"):
                st.sidebar.error("JEFF is not initialized. Please initialize JEFF first from the settings below.")
            else:
                summarization_system_prompt = (
                    f"You are an expert summarizer. Based on the provided context from a larger document, "
                    f"generate a concise summary focusing on the topic: '{selected_point}'. "
                    "Highlight the most important information related to this specific topic within the given context."
                )
                with st.spinner(f"Summarizing '{selected_point}'..."):
                    summary_text = generate_summary_for_point(
                        selected_point,
                        st.session_state.pipeline,
                        summarization_system_prompt
                    )
                    st.session_state.current_summary = summary_text
    elif st.session_state.get("file_path") and not st.session_state.get("main_points"):
        st.sidebar.info("No key points were automatically extracted, or extraction needs API key.")

    if st.session_state.get("current_summary"):
        with st.sidebar.expander("🔍 View Summary", expanded=True):
            st.markdown(st.session_state.current_summary)
            if st.button("Clear Summary", key="clear_summary_button"):
                st.session_state.current_summary = ""
                st.rerun()

    st.sidebar.markdown("---")

    # Determine mode-specific settings
    if st.session_state.mode == "evaluation":
        st.sidebar.header("🧪 Evaluation Config")
        st.sidebar.info("Adjust settings for Evaluation Mode. Press 'Initialize JEFF' after changing.")
        config_expander_expanded = True
    else:
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
            try:
                return options_list.index(current_value)
            except ValueError:
                return default_index

        if st.session_state.mode == "chat":
            st.sidebar.text(f"Embedding Model: {DEFAULT_EMBEDDING_MODEL.value}")
            st.session_state.embedding_model = DEFAULT_EMBEDDING_MODEL.value
        else:
            st.session_state.embedding_model = st.selectbox(
                "Embedding Model",
                options=embedding_options,
                index=get_safe_index(embedding_options, st.session_state.embedding_model),
                key="sb_embedding_model"
            )

        if st.session_state.mode == "chat":
            st.sidebar.text(f"Re-ranker Model: {st.session_state.get('reranker', DEFAULT_RERANKER_MODEL.value)}")
            st.sidebar.text(f"LLM Model: {st.session_state.get('llm_model', DEFAULT_LLM_MODEL.value)}")
            st.sidebar.text(f"Vector Store: {st.session_state.get('vector_store', DEFAULT_VECTOR_STORE.value)}")
            st.sidebar.text(
                f"Chunking Strategy: {st.session_state.get('chunking_strategy', DEFAULT_CHUNKING_STRATEGY.value)}")
            st.sidebar.text(f"Chunk Size: {st.session_state.get('chunk_size', DEFAULT_CHUNK_SIZE)}")
            st.sidebar.text(f"Chunk Overlap: {st.session_state.get('chunk_overlap', DEFAULT_CHUNK_OVERLAP)}")
            st.sidebar.text(f"Top K: {st.session_state.get('top_k', DEFAULT_TOP_K)}")
        else:
            st.session_state.reranker = st.selectbox(
                "Re-ranker Model",
                options=reranker_options,
                index=get_safe_index(reranker_options, st.session_state.reranker),
                key="sb_reranker"
            )
            st.session_state.llm_model = st.selectbox(
                "LLM Model",
                options=llm_options,
                index=get_safe_index(llm_options, st.session_state.llm_model),
                key="sb_llm_model"
            )
            st.session_state.vector_store = st.selectbox(
                "Vector Store",
                options=vector_store_options,
                index=get_safe_index(vector_store_options, st.session_state.vector_store),
                key="sb_vector_store"
            )
            st.session_state.chunking_strategy = st.selectbox(
                "Chunking Strategy",
                options=chunking_strategy_options,
                index=get_safe_index(chunking_strategy_options, st.session_state.chunking_strategy),
                key="sb_chunking_strategy"
            )

            # Get subject-specific configuration
            subject_config = get_subject_config(selected_subject)

            # Update slider values based on subject configuration
            st.session_state.chunk_size = st.slider(
                "Chunk Size (tokens)",
                min_value=100,
                max_value=2000,
                value=subject_config.chunk_size,
                step=50,
                key="sb_chunk_size",
                help="Maximum number of tokens per chunk"
            )

            st.session_state.chunk_overlap = st.slider(
                "Chunk Overlap (tokens)",
                min_value=0,
                max_value=500,
                value=subject_config.chunk_overlap,
                step=25,
                key="sb_chunk_overlap",
                help="Number of tokens to overlap between chunks"
            )

            st.session_state.top_k = st.slider(
                "Docs to Retrieve (Top K)",
                min_value=1,
                max_value=15,
                value=subject_config.top_k if hasattr(subject_config, 'top_k') else DEFAULT_TOP_K,
                step=1,
                key="sb_top_k"
            )

            try:
                selected_vector_store_enum = VectorStoreType.from_string(st.session_state.vector_store)
            except ValueError:
                selected_vector_store_enum = None

            if selected_vector_store_enum == VectorStoreType.HYBRID:
                st.caption("Hybrid search mixes keyword and vector search.")
                st.session_state.hybrid_alpha = st.slider(
                    "Vector Weight (alpha)",
                    0.0,
                    1.0,
                    subject_config.hybrid_alpha if hasattr(subject_config, 'hybrid_alpha') else DEFAULT_HYBRID_ALPHA,
                    0.05,
                    key="sb_hybrid_alpha",
                    help="1.0=vector, 0.0=keyword"
                )
                kw_weight = 1.0 - float(st.session_state.get('hybrid_alpha', 0.5))
                st.write(f"Keyword Weight: {kw_weight:.2f}")

        if st.session_state.pipeline and hasattr(st.session_state.pipeline, 'last_evaluation_scores'):
            st.markdown("---")
            st.subheader("📊 Current Configuration Scores")

            scores = st.session_state.pipeline.last_evaluation_scores
            if scores and isinstance(scores, dict):
                metric_cols = st.columns(len(scores))
                i = 0
                valid_scores = []
                for metric, score in scores.items():
                    if isinstance(score, (int, float)):
                        with metric_cols[i]:
                            score_display = f"{score:.2f}"
                            st.metric(
                                label=metric.replace('_', ' ').title(),
                                value=score_display,
                                delta=None
                            )
                        valid_scores.append(score)
                    i += 1

                if valid_scores:
                    avg_score = sum(valid_scores) / len(valid_scores)
                    st.metric("Average Score", f"{avg_score:.2f}")
            else:
                st.info("No evaluation scores available yet.")
        elif st.session_state.pipeline:
            st.info("Run an evaluation to see configuration scores.")

    st.sidebar.markdown("---")

    # Mode-specific controls
    if st.session_state.mode == "chat":
        if st.sidebar.button("Clear Chat History", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()

        show_contexts_now = st.sidebar.checkbox("Show Contexts", value=st.session_state.show_contexts)
        if show_contexts_now != st.session_state.show_contexts:
            st.session_state.show_contexts = show_contexts_now
            st.rerun()

        with st.sidebar.expander("🔑 API Key Status", expanded=False):
            try:
                current_embedding_model = EmbeddingModelType.from_string(st.session_state.embedding_model)
                current_vector_store = VectorStoreType.from_string(st.session_state.vector_store)
                current_reranker = RerankerModelType.from_string(st.session_state.reranker)
                current_llm = LLMModelType.from_string(st.session_state.llm_model)
                st.session_state.api_key_status = check_api_keys(
                    embedding_model_enum=current_embedding_model,
                    vector_store_enum=current_vector_store,
                    reranker_enum=current_reranker,
                    llm_enum=current_llm
                )
            except ValueError as e:
                st.error(f"Error checking keys (invalid model): {e}")
            except Exception as e_api:
                st.error(f"Error checking keys: {e_api}")
                logging.error(f"API Key check fail: {e_api}", exc_info=True)

            if st.session_state.api_key_status:
                missing_keys_found = False
                for key_name, status in st.session_state.api_key_status.items():
                    if status == "Missing":
                        missing_keys_found = True
                    st.text(f"{key_name}: {status}")

                if missing_keys_found:
                    st.warning("Missing keys needed for current config.", icon="🔑")
                    st.caption("Add to `.env` & restart if needed.")
                else:
                    st.info("No external API keys currently required.")

    st.sidebar.markdown("---")

    # Initialize JEFF button
    disable_init = not st.session_state.file_path or st.session_state.pipeline is not None
    if st.sidebar.button("🚀 Initialize JEFF", key="init_pipeline", help="Load textbook with current settings.",
                         disabled=disable_init):
        try:
            embedding_model = EmbeddingModelFactory.create_model(
                EmbeddingModelType.from_string(st.session_state.embedding_model)
            )
            vector_store = VectorStoreFactory.create_store(
                VectorStoreType.from_string(st.session_state.vector_store)
            )
            reranker = RerankerFactory.create_reranker(
                RerankerModelType.from_string(st.session_state.reranker)
            )
            llm = LLMFactory.create_llm(
                LLMModelType.from_string(st.session_state.llm_model)
            )
            chunking_strategy = ChunkingStrategyFactory.get_strategy(
                st.session_state.chunking_strategy
            )
        except (ValueError, TypeError) as e:
            st.sidebar.error(f"Invalid config: {e}")
            logging.error(f"Config error on Init: {e}")
            st.stop()

        missing_keys = check_api_keys(
            embedding_model_enum=EmbeddingModelType.from_string(st.session_state.embedding_model),
            vector_store_enum=VectorStoreType.from_string(st.session_state.vector_store),
            reranker_enum=RerankerModelType.from_string(st.session_state.reranker),
            llm_enum=LLMModelType.from_string(st.session_state.llm_model)
        )

        if missing_keys:
            st.sidebar.error(f"Cannot initialize. Missing keys: {', '.join(missing_keys)}", icon="🔑")
        else:
            with st.spinner("Warming up JEFF's brain..."):
                pipeline_instance = initialize_pipeline(
                    embedding_model=embedding_model,
                    vector_store=vector_store,
                    reranker=reranker,
                    llm=llm,
                    chunking_strategy=chunking_strategy,
                    chunk_size=st.session_state.chunk_size,
                    chunk_overlap=st.session_state.chunk_overlap,
                    top_k=st.session_state.top_k,
                    hybrid_alpha=st.session_state.get('hybrid_alpha', DEFAULT_HYBRID_ALPHA)
                )

            if pipeline_instance:
                st.session_state.pipeline = pipeline_instance
                st.sidebar.success("JEFF is ready!")
                st.rerun()
            else:
                st.sidebar.error("Failed to initialize JEFF. Check logs for details.")
    elif disable_init:
        st.sidebar.caption("Upload textbook to enable.")

    st.sidebar.markdown("---")