# Modular RAG Application

A modular and extensible Retrieval-Augmented Generation (RAG) application that allows for flexible configuration and evaluation of different RAG components.

## Project Structure

### Main Application (`app.py`)
The main entry point of the application that:
- Initializes the Streamlit web interface
- Sets up session state variables for configuration
- Manages the application's main flow between chat and evaluation modes
- Handles environment variables and logging configuration

### UI Components (`ui/`)
- `sidebar.py`: Implements the settings panel with configuration options for all RAG components
- `ui_components.py`: Contains the main chat and evaluation interface components
- `components/`: Additional UI components for specific features

### Models (`models/`)
- `llm_models.py`: Implements various Large Language Model interfaces and configurations
- `rerankers.py`: Contains different reranking strategies for improving retrieval quality
- `embedding_models.py`: Implements various embedding models for text vectorization
- `vector_stores.py`: Manages different vector store implementations for document storage
- `evaluator.py`: Comprehensive evaluation framework for RAG pipeline components
- `chunking_strategies.py`: Implements different text chunking strategies for document processing

### Pipeline (`pipeline/`)
- `rag_pipeline.py`: Core RAG pipeline implementation
- `pipeline_manager.py`: Manages pipeline configuration and execution
- `summarizer_module.py`: Handles document summarization functionality
- `nature_handling.py`: Manages document nature and type-specific processing
- `components/`: Individual pipeline components
- `utils/`: Pipeline-specific utility functions
- `models/`: Pipeline-specific model implementations

### Utils (`utils/`)
- `enums.py`: Contains enumerations for configuration options
- `subject_configs.py`: Default configurations for different components
- `token_utils.py`: Token counting and management utilities
- `api_management/`: API key and service management
- `file_handling/`: Document processing and file management
- `analysis/`: Analysis tools for RAG performance
- `text_processing/`: Text processing utilities

### Prompts (`prompts/`)
Contains prompt templates and configurations for different LLM interactions

## Detailed Component Documentation

### Main Application

#### `app.py`
Main entry point of the application that initializes and manages the Streamlit interface.

Functions:
- `main()`: Entry point that initializes the application and manages the main flow
  - Sets up session state variables
  - Initializes default configurations
  - Manages the application's main flow between chat and evaluation modes

### UI Components (`ui/`)

#### `ui_components.py`
Contains the main UI components for the chat and evaluation interfaces.

Functions:
- `display_chat_interface()`: Renders the main chat interface
- `display_evaluation_interface()`: Renders the evaluation interface

#### `sidebar.py`
Implements the settings panel with configuration options.

Functions:
- `display_settings_panel()`: Renders the settings sidebar
- `handle_model_selection()`: Manages model selection changes
- `handle_configuration_changes()`: Processes configuration updates
- `display_api_key_status()`: Shows API key configuration status

### Models (`models/`)

#### `llm_models.py`
Implements various Large Language Model interfaces.

Functions:
- `BaseLLM`: Abstract base class for LLM implementations
- `OpenAILLM`: OpenAI model implementation
  - `generate()`: Generates text responses
  - `get_embeddings()`: Gets text embeddings
- `AnthropicLLM`: Anthropic model implementation
  - `generate()`: Generates text responses
  - `get_embeddings()`: Gets text embeddings
- `GoogleLLM`: Google model implementation
  - `generate()`: Generates text responses
  - `get_embeddings()`: Gets text embeddings

#### `rerankers.py`
Contains different reranking strategies.

Functions:
- `BaseReranker`: Abstract base class for rerankers
- `CohereReranker`: Cohere-based reranking implementation
  - `rerank()`: Reranks retrieved documents
- `CrossEncoderReranker`: Cross-encoder based reranking
  - `rerank()`: Reranks using cross-encoder models

#### `embedding_models.py`
Implements various embedding models.

Functions:
- `BaseEmbeddingModel`: Abstract base class for embedding models
- `OpenAIEmbedding`: OpenAI embedding implementation
  - `get_embeddings()`: Generates embeddings
- `CohereEmbedding`: Cohere embedding implementation
  - `get_embeddings()`: Generates embeddings
- `SentenceTransformerEmbedding`: Sentence transformer implementation
  - `get_embeddings()`: Generates embeddings

#### `vector_stores.py`
Manages different vector store implementations.

Functions:
- `BaseVectorStore`: Abstract base class for vector stores
- `FAISSVectorStore`: FAISS implementation
  - `add_documents()`: Adds documents to the store
  - `search()`: Searches for similar documents
- `MilvusVectorStore`: Milvus implementation
  - `add_documents()`: Adds documents to the store
  - `search()`: Searches for similar documents

#### `evaluator.py`
Comprehensive evaluation framework.

Functions:
- `RAGEvaluator`: Main evaluation class
  - `evaluate_retrieval()`: Evaluates retrieval performance
  - `evaluate_generation()`: Evaluates generation quality
  - `evaluate_end_to_end()`: Evaluates full pipeline
- `MetricsCalculator`: Calculates various metrics
  - `calculate_precision()`: Calculates precision
  - `calculate_recall()`: Calculates recall
  - `calculate_ndcg()`: Calculates NDCG

#### `chunking_strategies.py`
Implements different text chunking strategies.

Functions:
- `BaseChunkingStrategy`: Abstract base class for chunking
- `FixedSizeChunking`: Fixed-size chunking implementation
  - `chunk()`: Chunks text into fixed-size pieces
- `SemanticChunking`: Semantic-based chunking
  - `chunk()`: Chunks text based on semantic boundaries

### Pipeline (`pipeline/`)

#### `rag_pipeline.py`
Core RAG pipeline implementation.

Functions:
- `RAGPipeline`: Main pipeline class
  - `process_documents()`: Processes input documents
  - `retrieve()`: Retrieves relevant documents
  - `generate()`: Generates responses
- `PipelineConfig`: Configuration management
  - `validate_config()`: Validates pipeline configuration
  - `update_config()`: Updates pipeline settings

#### `pipeline_manager.py`
Manages pipeline configuration and execution.

Functions:
- `PipelineManager`: Main manager class
  - `initialize_pipeline()`: Initializes the pipeline
  - `update_components()`: Updates pipeline components
  - `execute_pipeline()`: Executes the pipeline

#### `summarizer_module.py`
Handles document summarization.

Functions:
- `DocumentSummarizer`: Summarization class
  - `summarize()`: Generates document summaries
  - `extract_key_points()`: Extracts key points

#### `nature_handling.py`
Manages document nature and type-specific processing.

Functions:
- `DocumentNatureHandler`: Nature handling class
  - `determine_nature()`: Determines document nature
  - `process_by_nature()`: Processes documents by nature

### Utils (`utils/`)

#### `enums.py`
Contains enumerations for configuration options.

Classes:
- `EmbeddingModelType`: Embedding model types
- `VectorStoreType`: Vector store types
- `RerankerType`: Reranker types
- `LLMModelType`: LLM model types
- `ChunkingStrategyType`: Chunking strategy types

#### `subject_configs.py`
Default configurations for different components.

Functions:
- `get_default_config()`: Gets default configuration
- `validate_config()`: Validates configuration
- `update_config()`: Updates configuration

#### `token_utils.py`
Token counting and management utilities.

Functions:
- `count_tokens()`: Counts tokens in text
- `truncate_by_tokens()`: Truncates text by token count
- `estimate_tokens()`: Estimates token count

#### `api_management/`
API key and service management.

Functions:
- `validate_api_key()`: Validates API keys
- `get_api_client()`: Gets API client
- `check_service_status()`: Checks service status

#### `file_handling/`
Document processing and file management.

Functions:
- `load_document()`: Loads documents
- `process_document()`: Processes documents
- `save_processed()`: Saves processed documents

#### `analysis/`
Analysis tools for RAG performance.

Functions:
- `analyze_retrieval()`: Analyzes retrieval performance
- `analyze_generation()`: Analyzes generation quality
- `generate_report()`: Generates analysis reports

#### `text_processing/`
Text processing utilities.

Functions:
- `clean_text()`: Cleans text
- `normalize_text()`: Normalizes text
- `extract_metadata()`: Extracts metadata

### Prompts (`prompts/`)
Contains prompt templates and configurations.

Files:
- `chat_prompts.py`: Chat-related prompts
- `evaluation_prompts.py`: Evaluation-related prompts
- `summarization_prompts.py`: Summarization-related prompts

## Key Features

1. **Modular Architecture**
   - Each component (embedding models, vector stores, rerankers, LLMs) can be swapped independently
   - Easy to add new implementations of any component

2. **Flexible Configuration**
   - Configurable chunking strategies
   - Adjustable retrieval parameters
   - Customizable LLM settings
   - Hybrid search capabilities

3. **Evaluation Framework**
   - Comprehensive evaluation metrics
   - Component-wise performance analysis
   - A/B testing capabilities

4. **User Interface**
   - Interactive chat interface
   - Real-time configuration changes
   - Performance visualization
   - Document upload and processing

## Configuration Options

### Embedding Models
- Support for various embedding models
- Configurable dimensions and parameters
- Easy integration of new models

### Vector Stores
- Multiple vector store implementations
- Configurable indexing strategies
- Support for different similarity metrics

### Rerankers
- Various reranking strategies
- Configurable reranking parameters
- Support for custom reranking logic

### LLM Models
- Multiple LLM provider support
- Configurable model parameters
- Temperature and context window settings

### Chunking Strategies
- Multiple chunking approaches
- Configurable chunk size and overlap
- Support for custom chunking logic

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
- Create a `.env` file with necessary API keys
- Configure model-specific settings

3. Run the application:
```bash
streamlit run app.py
```

## Development

### Adding New Components

1. **New Embedding Model**
   - Implement the model interface in `models/embedding_models.py`
   - Add configuration options in `utils/subject_configs.py`

2. **New Vector Store**
   - Implement the store interface in `models/vector_stores.py`
   - Add configuration options in `utils/subject_configs.py`

3. **New Reranker**
   - Implement the reranker interface in `models/rerankers.py`
   - Add configuration options in `utils/subject_configs.py`

4. **New LLM Model**
   - Implement the model interface in `models/llm_models.py`
   - Add configuration options in `utils/subject_configs.py`

### Testing

- Use the evaluation interface to test new components
- Compare performance metrics with existing implementations
- Validate integration with the pipeline

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## License

[Add your license information here]