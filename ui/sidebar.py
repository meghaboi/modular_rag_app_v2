import streamlit as st
import logging
import os
from utils.subject_configs import (
    SUBJECT_CONFIGS,
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
from pipeline.nature_handling import update_rag_configuration
from pipeline.utils.pipeline_initializer import PipelineInitializer
from pipeline.components.config import PipelineConfig
from utils.enums import (
    EmbeddingModelType,
    RerankerModelType,
    LLMModelType,
    VectorStoreType,
    ChunkingStrategyType
)
from utils.file_handling.file_utils import save_uploaded_file
from utils.api_management.api_utils import check_api_keys
from models.llm_models import LLMFactory
from pipeline.summarizer_module import extract_main_points
from prompts import get_provider

SIDEBAR_IMAGE_URL = "https://i.postimg.cc/DfLpxwZJ/Chat-GPT-Image-May-7-2025-11-10-13-AM.png"
SIDEBAR_IMAGE_WIDTH = 80
MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 2000
CHUNK_SIZE_STEP = 100
MAX_CHUNK_OVERLAP = 500
CHUNK_OVERLAP_STEP = 50
MIN_TOP_K = 1
MAX_TOP_K = 10
POINT_EXTRACTION_LLM = LLMModelType.CLAUDE_4_SONNET
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024 

class SessionStateManager:
    """Manages Streamlit session state initialization and updates."""

    @staticmethod
    def initialize_default_state():
        """Initialize session state with default values."""
        defaults = {
            'main_points': [],
            'current_summary': "",
            'point_extraction_llm_missing_keys': False,
            'mode': 'chat',
            'messages': [],
            'show_contexts': False,
            'config_changed': False,
            'embedding_model': DEFAULT_EMBEDDING_MODEL.value,
            'vector_store': DEFAULT_VECTOR_STORE.value,
            'reranker': DEFAULT_RERANKER_MODEL.value,
            'llm_model': DEFAULT_LLM_MODEL.value,
            'chunking_strategy': DEFAULT_CHUNKING_STRATEGY.value,
            'hybrid_alpha': DEFAULT_HYBRID_ALPHA,
            'chunk_size': DEFAULT_CHUNK_SIZE,
            'chunk_overlap': DEFAULT_CHUNK_OVERLAP,
            'top_k': DEFAULT_TOP_K
        }

        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    @staticmethod
    def reset_pipeline_state():
        """Reset pipeline-related session state."""
        st.session_state.pipeline = None
        st.session_state.messages = []
        st.session_state.permutation_results = None
        st.session_state.permutation_df = None

    @staticmethod
    def reset_document_state():
        """Reset document-related session state."""
        st.session_state.main_points = []
        st.session_state.current_summary = ""
        st.session_state.point_extraction_llm_missing_keys = False

class ModeManager:
    """Handles mode switching and related state management."""

    @staticmethod
    def get_mode_options():
        """Get available mode options."""
        return {
            "💬 Chat with JEFF": "chat",
            "🧪 Test Setups (Evaluation)": "evaluation"
        }

    @staticmethod
    def handle_mode_change(new_mode):
        """Handle mode switching with appropriate state management."""
        if new_mode == st.session_state.mode:
            return False

        st.session_state.mode = new_mode
        st.session_state.messages = []

        should_reset_pipeline = ModeManager._should_reset_pipeline()
        if should_reset_pipeline:
            SessionStateManager.reset_pipeline_state()
            logging.info("Configuration changed. Resetting pipeline state.")
        else:
            logging.info("Mode changed without config changes. Preserving pipeline state.")

        return True

    @staticmethod
    def _should_reset_pipeline():
        """Determine if pipeline should be reset based on configuration changes."""
        if st.session_state.pipeline is None:
            return False

        current_config = {
            'embedding_model': st.session_state.embedding_model,
            'vector_store': st.session_state.vector_store,
            'reranker': st.session_state.reranker,
            'llm_model': st.session_state.llm_model,
            'chunking_strategy': st.session_state.chunking_strategy,
            'hybrid_alpha': st.session_state.hybrid_alpha,
            'chunk_size': st.session_state.chunk_size,
            'chunk_overlap': st.session_state.chunk_overlap,
            'top_k': st.session_state.top_k
        }

        return current_config != st.session_state.pipeline.get_config()

class FileUploadHandler:
    """Handles file upload and processing logic."""

    @staticmethod
    def process_uploaded_file(uploaded_file):
        """Process uploaded file and update session state."""
        if not FileUploadHandler._is_new_file(uploaded_file):
            return

        logging.info(f"New file upload detected: {uploaded_file.name}")

        with st.spinner(f"Processing '{uploaded_file.name}'..."):
            file_path = save_uploaded_file(uploaded_file)

        if not file_path:
            FileUploadHandler._handle_upload_failure()
            return

        FileUploadHandler._handle_successful_upload(uploaded_file, file_path)

    @staticmethod
    def _is_new_file(uploaded_file):
        """Check if uploaded file is different from current file."""
        return uploaded_file.name != st.session_state.get('last_uploaded_filename', None)

    @staticmethod
    def _handle_upload_failure():
        """Handle failed file upload."""
        st.sidebar.error("Failed to process uploaded file.")
        st.session_state.file_path = None
        st.session_state.last_uploaded_filename = None
        st.session_state.pipeline = None

    @staticmethod
    def _handle_successful_upload(uploaded_file, file_path):
        """Handle successful file upload and point extraction."""
        st.session_state.file_path = file_path
        st.session_state.last_uploaded_filename = uploaded_file.name
        SessionStateManager.reset_pipeline_state()
        SessionStateManager.reset_document_state()

        FileUploadHandler._extract_document_points()
        st.sidebar.success(f"'{uploaded_file.name}' loaded!")
        st.rerun()

    @staticmethod
    def _extract_document_points():
        """Extract main points from uploaded document, then update config and initialize pipeline."""
        if FileUploadHandler._is_extraction_api_key_missing():
            st.session_state.point_extraction_llm_missing_keys = True
            st.sidebar.warning(f"API key for {POINT_EXTRACTION_LLM.value} needed for automatic point extraction.")
            return

        st.session_state.point_extraction_llm_missing_keys = False

        try:
            point_extraction_llm = LLMFactory.create_llm(POINT_EXTRACTION_LLM)
            with st.spinner("Extracting key points from document..."):
                st.session_state.main_points = extract_main_points(
                    st.session_state.file_path,
                    point_extraction_llm
                )

            if not st.session_state.main_points:
                st.sidebar.warning("Could not extract key points automatically.")
                return

            # --- Immediately update subject config and initialize pipeline ---
            # For subject config, we use the main points as a proxy for subject/topic.
            # If you have a more sophisticated mapping, replace this logic accordingly.
            # Here, we use the first main point as the subject hint.
            from pipeline.nature_handling import get_config_by_prompt_nature
            from pipeline.components.config import PipelineConfig
            from pipeline.utils.pipeline_initializer import PipelineInitializer
            from utils.enums import (
                EmbeddingModelType, VectorStoreType, RerankerModelType, LLMModelType, ChunkingStrategyType
            )

            # Determine subject config (use first main point as subject proxy)
            subject_hint = st.session_state.main_points[0] if st.session_state.main_points else "general"
            subject_config = get_config_by_prompt_nature(subject_hint)

            # Build PipelineConfig using current UI/model selections and subject config
            config = PipelineConfig(
                file_path=st.session_state.file_path,
                embedding_model_type=EmbeddingModelType.from_string(st.session_state.embedding_model),
                vector_store_type=VectorStoreType.from_string(st.session_state.vector_store),
                reranker_type=RerankerModelType.from_string(st.session_state.reranker),
                llm_type=LLMModelType.from_string(st.session_state.llm_model),
                chunking_strategy_type=ChunkingStrategyType.from_string(st.session_state.chunking_strategy),
                chunk_size=subject_config.chunk_size,
                chunk_overlap=subject_config.chunk_overlap,
                top_k=subject_config.top_k,
                hybrid_alpha=subject_config.hybrid_alpha,
                evaluation_mode=(st.session_state.mode == 'evaluation')
            )

            # Initialize pipeline and store in session state
            initializer = PipelineInitializer(config)
            try:
                pipeline_instance = initializer.initialize_pipeline()
                st.session_state.pipeline = pipeline_instance
                st.session_state.config_changed = False
                st.sidebar.success("Pipeline initialized with updated config!")
            except Exception as e:
                st.sidebar.error(f"Pipeline initialization failed: {str(e)}")
                logging.error(f"Pipeline initialization failed: {e}", exc_info=True)

        except Exception as e:
            st.sidebar.error(f"Error during point extraction: {str(e)}")
            logging.error(f"Point extraction failed: {e}", exc_info=True)

    @staticmethod
    def _is_extraction_api_key_missing():
        """Check if API key for point extraction is missing."""
        current_models = FileUploadHandler._get_current_model_enums()
        missing_keys = check_api_keys(**current_models, llm_enum=POINT_EXTRACTION_LLM)

        if not missing_keys:
            return False

        required_key = FileUploadHandler._get_required_api_key(POINT_EXTRACTION_LLM)
        return required_key in missing_keys

    @staticmethod
    def _get_current_model_enums():
        """Get current model enums for API key checking."""
        return {
            'embedding_model_enum': EmbeddingModelType.from_string(
                st.session_state.get('embedding_model', DEFAULT_EMBEDDING_MODEL.value)),
            'vector_store_enum': VectorStoreType.from_string(
                st.session_state.get('vector_store', DEFAULT_VECTOR_STORE.value)),
            'reranker_enum': RerankerModelType.from_string(
                st.session_state.get('reranker', DEFAULT_RERANKER_MODEL.value))
        }

    @staticmethod
    def _get_required_api_key(llm_type):
        """Get required API key name for LLM type."""
        if llm_type.name.startswith("CLAUDE"):
            return "ANTHROPIC_API_KEY"
        elif llm_type.name.startswith("OPENAI"):
            return "OPENAI_API_KEY"
        elif llm_type.name.startswith("GEMINI"):
            return "GEMINI_API_KEY"
        elif llm_type.name.startswith("MISTRAL"):
            return "MISTRAL_API_KEY"
        return ""

class ConfigurationManager:
    """Manages RAG configuration settings."""

    @staticmethod
    def render_configuration_panel(is_evaluation_mode):
        """Render configuration panel based on mode."""
        if is_evaluation_mode:
            st.sidebar.header("🧪 Evaluation Config")
            st.sidebar.info("Adjust settings for Evaluation Mode. Press 'Initialize JEFF' after changing.")
            ConfigurationManager._render_editable_config()
        else:
            st.sidebar.header("⚙️ Current Setup")
            st.sidebar.info("JEFF uses this setup. Switch to Evaluation Mode to change.")
            ConfigurationManager._render_readonly_config()

    @staticmethod
    def _render_editable_config():
        """Render editable configuration controls."""
        with st.sidebar.expander("RAG Configuration Details", expanded=True):
            ConfigurationManager._render_model_selectors()
            ConfigurationManager._render_parameter_controls()
            ConfigurationManager._render_evaluation_scores()

    @staticmethod
    def _render_readonly_config():
        """Render readonly configuration display."""
        with st.sidebar.expander("RAG Configuration Details", expanded=False):
            st.sidebar.text(f"Embedding Model: {DEFAULT_EMBEDDING_MODEL.value}")
            st.session_state.embedding_model = DEFAULT_EMBEDDING_MODEL.value
            ConfigurationManager._render_model_selectors(readonly=True)
            ConfigurationManager._render_parameter_controls(readonly=True)
            ConfigurationManager._render_evaluation_scores()

    @staticmethod
    def _render_model_selectors(readonly=False):
        """Render model selection controls."""
        options = ConfigurationManager._get_model_options()

        for model_type, (options_list, session_key, label) in options.items():
            if readonly and model_type == 'embedding':
                continue  # Skip embedding in readonly mode

            current_value = st.session_state.get(session_key, options_list[0])
            index = ConfigurationManager._get_safe_index(options_list, current_value)

            st.session_state[session_key] = st.selectbox(
                label,
                options=options_list,
                index=index,
                key=f"sb_{session_key}",
                on_change=ConfigurationManager._on_config_change,
                disabled=readonly
            )

    @staticmethod
    def _render_parameter_controls(readonly=False):
        """Render parameter input controls."""
        parameters = [
            ('chunk_size', 'Chunk Size', MIN_CHUNK_SIZE, MAX_CHUNK_SIZE, CHUNK_SIZE_STEP),
            ('chunk_overlap', 'Chunk Overlap', 0, MAX_CHUNK_OVERLAP, CHUNK_OVERLAP_STEP),
            ('top_k', 'Top K', MIN_TOP_K, MAX_TOP_K, 1)
        ]

        for param_key, label, min_val, max_val, step in parameters:
            st.session_state[param_key] = st.number_input(
                label,
                min_value=min_val,
                max_value=max_val,
                value=st.session_state.get(param_key, min_val),
                step=step,
                key=f"sb_{param_key}",
                on_change=ConfigurationManager._on_config_change,
                disabled=readonly
            )

        if st.session_state.vector_store == VectorStoreType.HYBRID.value:
            st.session_state.hybrid_alpha = st.slider(
                "Hybrid Alpha",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get('hybrid_alpha', DEFAULT_HYBRID_ALPHA),
                step=0.1,
                key="sb_hybrid_alpha",
                on_change=ConfigurationManager._on_config_change,
                disabled=readonly
            )

    @staticmethod
    def _render_evaluation_scores():
        """Render evaluation scores if available."""
        if not (st.session_state.pipeline and hasattr(st.session_state.pipeline, 'last_evaluation_scores')):
            return

        scores = st.session_state.pipeline.last_evaluation_scores
        if not (scores and isinstance(scores, dict)):
            st.info("No evaluation scores available yet.")
            return

        st.markdown("---")
        st.subheader("📊 Current Configuration Scores")

        valid_scores = ConfigurationManager._display_score_metrics(scores)
        ConfigurationManager._display_average_score(valid_scores)

    @staticmethod
    def _display_score_metrics(scores):
        """Display individual score metrics."""
        metric_cols = st.columns(len(scores))
        valid_scores = []

        for i, (metric, score) in enumerate(scores.items()):
            if isinstance(score, (int, float)):
                with metric_cols[i]:
                    st.metric(
                        label=metric.replace('_', ' ').title(),
                        value=f"{score:.2f}",
                        delta=None
                    )
                valid_scores.append(score)

        return valid_scores

    @staticmethod
    def _display_average_score(valid_scores):
        """Display average score if valid scores exist."""
        if valid_scores:
            avg_score = sum(valid_scores) / len(valid_scores)
            st.metric("Average Score", f"{avg_score:.2f}")

    @staticmethod
    def _get_model_options():
        """Get model options for configuration."""
        return {
            'embedding': (EmbeddingModelType.list(), 'embedding_model', 'Embedding Model'),
            'vector_store': (VectorStoreType.list(), 'vector_store', 'Vector Store'),
            'reranker': (RerankerModelType.list(), 'reranker', 'Reranker'),
            'llm': (LLMModelType.list(), 'llm_model', 'LLM Model'),
            'chunking': (ChunkingStrategyType.list(), 'chunking_strategy', 'Chunking Strategy')
        }

    @staticmethod
    def _get_safe_index(options_list, current_value, default_index=0):
        """Get safe index for selectbox options."""
        try:
            return options_list.index(current_value)
        except ValueError:
            return default_index

    @staticmethod
    def _on_config_change():
        """Handle configuration change."""
        st.session_state.config_changed = True

class SummaryManager:
    """Manages document summarization functionality."""

    @staticmethod
    def render_summary_panel():
        """Render document summary panel."""
        st.sidebar.header("📄 Document Summary")

        if not st.session_state.get("file_path"):
            st.sidebar.caption("Upload a document to enable summarization.")
            return

        if st.session_state.get("point_extraction_llm_missing_keys"):
            st.sidebar.warning("API key for point extraction LLM is missing. Cannot display or summarize key points.")
            return

        if not st.session_state.get("main_points"):
            st.sidebar.info("No key points were automatically extracted, or extraction needs API key.")
            return

        SummaryManager._render_point_selector()
        SummaryManager._render_current_summary()

    @staticmethod
    def _render_point_selector():
        """Render point selection and summarization controls."""
        selected_point = st.sidebar.selectbox(
            "Select Key Point to Summarize:",
            options=st.session_state.main_points,
            key="selected_main_point"
        )

        if st.sidebar.button("✨ Summarize Selected Point", key="summarize_point_button"):
            SummaryManager._handle_point_summarization(selected_point)

    @staticmethod
    def _handle_point_summarization(selected_point):
        """Handle point summarization request."""
        if not st.session_state.get("pipeline"):
            st.sidebar.error("JEFF is not initialized. Please initialize JEFF first from the settings below.")
            return

        with st.spinner(f"Summarizing '{selected_point}'..."):
            try:
                pipeline = st.session_state.pipeline
                contexts = pipeline.retrieve_context(selected_point)
                if not contexts:
                    st.sidebar.warning(f"No relevant context found for '{selected_point}'.")
                    st.session_state.current_summary = f"Could not find any information about '{selected_point}' in the document."
                    return

                context_str = "\n\n".join(contexts)

                summarizer_provider = get_provider('summarizer')
                system_prompt = summarizer_provider.get_prompt(
                    'point_summary',
                    topic=selected_point,
                    context=context_str
                )

                # The generate method now returns a tuple (text, usage_info)
                summary_text, _ = pipeline.llm.generate(prompt=system_prompt)
                st.session_state.current_summary = summary_text
            except Exception as e:
                logging.error(f"Error during point summarization for '{selected_point}': {e}", exc_info=True)
                st.sidebar.error(f"An error occurred while summarizing. Please check the logs.")
                st.session_state.current_summary = "Failed to generate summary due to an error."

    @staticmethod
    def _render_current_summary():
        """Render current summary display."""
        if not st.session_state.get("current_summary"):
            return

        with st.sidebar.expander("🔍 View Summary", expanded=True):
            st.markdown(st.session_state.current_summary)
            if st.button("Clear Summary", key="clear_summary_button"):
                st.session_state.current_summary = ""
                st.rerun()

class StatusManager:
    """Manages system status display."""

    @staticmethod
    def render_status_panel():
        """Render system status panel."""
        st.sidebar.header("🚦 System Status")

        with st.sidebar.container():
            StatusManager._render_textbook_status()
            StatusManager._render_pipeline_status()

    @staticmethod
    def _render_textbook_status():
        """Render textbook loading status."""
        if StatusManager._is_textbook_loaded():
            st.success(f"✅ Textbook: {st.session_state.last_uploaded_filename}")
        else:
            st.warning("⚠️ No textbook loaded")

    @staticmethod
    def _render_pipeline_status():
        """Render pipeline initialization status."""
        if st.session_state.pipeline:
            st.success("✅ JEFF is ready!")
        else:
            st.warning("⏳ JEFF needs setup (Initialize)")

    @staticmethod
    def _is_textbook_loaded():
        """Check if textbook is properly loaded."""
        return (st.session_state.file_path and
                os.path.exists(st.session_state.file_path))

class SidebarPipelineInitializer:
    """Handles pipeline initialization logic."""

    @staticmethod
    def render_initialization_controls():
        """Render pipeline initialization controls."""
        st.sidebar.markdown("---")

        disable_init = SidebarPipelineInitializer._should_disable_initialization()

        if st.sidebar.button(
                "🚀 Initialize JEFF",
                key="init_pipeline",
                help="Load textbook with current settings.",
                disabled=disable_init
        ):
            SidebarPipelineInitializer._handle_initialization_request()
        elif disable_init:
            st.sidebar.caption("Upload textbook to enable.")

    @staticmethod
    def _should_disable_initialization():
        """Determine if initialization should be disabled."""
        return (not st.session_state.file_path or
                (st.session_state.pipeline is not None and not st.session_state.config_changed))

    @staticmethod
    def _handle_initialization_request():
        """Handle pipeline initialization request."""
        model_enums_for_check = SidebarPipelineInitializer._get_model_enums()
        missing_keys = check_api_keys(**model_enums_for_check)

        if missing_keys:
            st.sidebar.error(f"Cannot initialize. Missing keys: {', '.join(missing_keys)}", icon="🔑")
            return

        # For PipelineConfig, keys must match the dataclass fields
        model_types_for_config = {
            'embedding_model_type': model_enums_for_check['embedding_model_enum'],
            'vector_store_type': model_enums_for_check['vector_store_enum'],
            'reranker_type': model_enums_for_check['reranker_enum'],
            'llm_type': model_enums_for_check['llm_enum'],
            'chunking_strategy_type': ChunkingStrategyType.from_string(st.session_state.chunking_strategy)
        }

        SidebarPipelineInitializer._initialize_pipeline(model_types_for_config)

    @staticmethod
    def _get_model_enums():
        """Get model enums for pipeline initialization."""
        return {
            'embedding_model_enum': EmbeddingModelType.from_string(st.session_state.embedding_model),
            'vector_store_enum': VectorStoreType.from_string(st.session_state.vector_store),
            'reranker_enum': RerankerModelType.from_string(st.session_state.reranker),
            'llm_enum': LLMModelType.from_string(st.session_state.llm_model),
        }
    @staticmethod
    def _initialize_pipeline(model_types):
        """Initialize the pipeline with current configuration."""
        with st.spinner("Warming up JEFF's brain..."):
            try:
                config = PipelineConfig(
                    file_path=st.session_state.file_path,
                    chunk_size=st.session_state.chunk_size,
                    chunk_overlap=st.session_state.chunk_overlap,
                    top_k=st.session_state.top_k,
                    hybrid_alpha=st.session_state.get('hybrid_alpha', DEFAULT_HYBRID_ALPHA),
                    evaluation_mode=(st.session_state.mode == 'evaluation'),
                    **model_types
                )
                initializer = PipelineInitializer(config)
                pipeline_instance = initializer.initialize_pipeline()
            except Exception as e:
                logging.error(f"Error during pipeline initialization: {e}", exc_info=True)
                st.sidebar.error(f"Initialization failed: {e}")
                pipeline_instance = None

        if pipeline_instance:
            st.session_state.pipeline = pipeline_instance
            st.session_state.config_changed = False
            st.sidebar.success("JEFF is ready!")
            st.rerun()
        else:
            st.sidebar.error("Failed to initialize JEFF. Check logs for details.")

def display_settings_panel():
    """Main function to display the settings panel."""
    SessionStateManager.initialize_default_state()

    _render_header()
    st.sidebar.header("🧑‍🎓 Subject Selection")
    _handle_subject_selection()
    st.sidebar.markdown("")

    st.sidebar.header("🧠 Mode Selection")
    _handle_mode_selection()
    st.sidebar.markdown("")

    # --- Contextual RAG Mode Button ---
    if st.sidebar.button("✨ Contextual RAG Mode", help="Switch to contextual chunking, hybrid retrieval, Cohere embedding, Voyage reranker, Claude 4 LLM"):
        st.session_state.chunking_strategy = "Contextual"
        st.session_state.embedding_model = DEFAULT_EMBEDDING_MODEL.value
        st.session_state.vector_store = "Hybrid"
        st.session_state.reranker = DEFAULT_RERANKER_MODEL.value
        st.session_state.llm_model = DEFAULT_LLM_MODEL.value
        st.session_state.config_changed = True
        st.sidebar.success("Contextual RAG configuration applied! Click 'Initialize JEFF' to activate.")

    st.sidebar.header("📚 Load Textbook")
    _handle_file_upload()
    st.sidebar.markdown("---")

    st.sidebar.header("🔎 System Status")
    StatusManager.render_status_panel()
    st.sidebar.markdown("---")

    st.sidebar.header("📝 Document Summary")
    SummaryManager.render_summary_panel()
    st.sidebar.markdown("---")

    st.sidebar.header("⚙️ Configuration")
    ConfigurationManager.render_configuration_panel(st.session_state.mode == "evaluation")
    st.sidebar.markdown("---")

    st.sidebar.header("🛠️ Mode-Specific Controls")
    _render_mode_specific_controls()
    st.sidebar.markdown("---")

    st.sidebar.header("🚦 Pipeline Initialization")
    SidebarPipelineInitializer.render_initialization_controls()
    st.sidebar.markdown("---")

def _render_header():
    """Render sidebar header."""
    st.sidebar.image(SIDEBAR_IMAGE_URL, width=SIDEBAR_IMAGE_WIDTH)
    st.sidebar.title("Ask-JEFF")

def _handle_subject_selection():
    """Handle subject selection."""
    subjects = list(SUBJECT_CONFIGS.keys())
    selected_subject = st.sidebar.selectbox(
        "Select Subject",
        subjects,
        index=subjects.index("general"),
        help="Choose the subject of your textbook for optimal RAG configuration."
    )

    if st.session_state.pipeline is not None and st.session_state.mode == "evaluation":
        update_rag_configuration(selected_subject)

def _handle_mode_selection():
    """Handle mode selection and switching."""
    mode_options = ModeManager.get_mode_options()
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
        help="Switch between chat and evaluation modes."
    )

    new_mode = mode_options[selected_mode_label]
    if ModeManager.handle_mode_change(new_mode):
        st.rerun()

def _handle_file_upload():
    """Handle file upload."""
    uploaded_file = st.sidebar.file_uploader(
        "Upload .txt or .pdf file",
        type=['txt', 'pdf'],
        key="file_uploader",
        help="Upload your textbook in .txt or .pdf format. Max size 50MB."
    )

    if uploaded_file is not None:
        with st.spinner("Processing uploaded file..."):
            FileUploadHandler.process_uploaded_file(uploaded_file)

def _render_mode_specific_controls():
    """Render controls specific to current mode."""
    if st.session_state.mode == "chat":
        _render_chat_controls()

def _render_chat_controls():
    """Render chat mode specific controls."""
    if st.sidebar.button("🗑️ Clear Chat History", key="clear_chat", help="Remove all previous chat messages."):
        st.session_state.messages = []
        st.rerun()

    show_contexts_now = st.sidebar.checkbox("Show Contexts", value=st.session_state.show_contexts, help="Display retrieved context passages in chat mode.")
    if show_contexts_now != st.session_state.show_contexts:
        st.session_state.show_contexts = show_contexts_now
        st.rerun()

    _render_api_key_status()

def _render_api_key_status():
    """Render API key status information."""
    with st.sidebar.expander("🔑 API Key Status", expanded=False):
        try:
            model_enums = {
                'embedding_model_enum': EmbeddingModelType.from_string(st.session_state.embedding_model),
                'vector_store_enum': VectorStoreType.from_string(st.session_state.vector_store),
                'reranker_enum': RerankerModelType.from_string(st.session_state.reranker),
                'llm_enum': LLMModelType.from_string(st.session_state.llm_model)
            }

            st.session_state.api_key_status = check_api_keys(**model_enums)

        except ValueError as e:
            st.error(f"Error checking keys (invalid model): {e}")
            return
        except Exception as e:
            st.error(f"Error checking keys: {e}")
            logging.error(f"API Key check fail: {e}", exc_info=True)
            return

        if st.session_state.api_key_status:
            missing_keys_found = any(
                status == "Missing" for status in st.session_state.api_key_status.values()
            )

            for key_name, status in st.session_state.api_key_status.items():
                st.text(f"{key_name}: {status}")

            if missing_keys_found:
                st.warning("Missing keys needed for current config.", icon="🔑")
                st.caption("Add to `.env` & restart if needed.")
            else:
                st.info("No external API keys currently required.")