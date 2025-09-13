# Modular RAG Application

A powerful and flexible Retrieval-Augmented Generation (RAG) system that combines the knowledge retrieval capabilities of search systems with the language understanding and generation abilities of large language models (LLMs).

## Core Concepts

### What is RAG?
Retrieval-Augmented Generation (RAG) enhances large language models by:
1. **Retrieving** relevant information from a knowledge base
2. **Augmenting** the model's context with this information
3. **Generating** responses that are grounded in the retrieved knowledge

### Why Use This Implementation?
- **Modular Design**: Swap components (models, retrievers, etc.) without changing core logic
- **Multi-Model Support**: Use various LLMs (OpenAI, Claude, Gemini, Mistral) with consistent interfaces
- **Configurable Pipeline**: Tune each component to optimize performance
- **Evaluation Framework**: Built-in tools to measure and improve system performance

## System Architecture

The application is structured into several key components that work together seamlessly:

### 1. User Interface Layer
- **Web Interface**: Interactive Streamlit-based UI for user interaction
- **Chat Interface**: Natural language input and response display
- **Configuration Panel**: Control panel for adjusting system settings
- **Evaluation Dashboard**: Monitor and analyze system performance

### UI Components (`ui/`)
- `sidebar.py`: Implements the settings panel with configuration options for all RAG components
- `ui_components.py`: Contains the main chat and evaluation interface components
- `components/`: Additional UI components for specific features

### 2. Core Components

#### Language Models
- Multiple LLM providers (OpenAI, Claude, Gemini, Mistral)
- Unified interface for different models
- Streaming support for real-time responses
- Token usage tracking and cost calculation

#### Document Processing
- **Chunking**: Breaks down documents into manageable pieces
- **Embedding**: Converts text into numerical vectors for semantic search
- **Vector Storage**: Efficiently stores and retrieves document vectors
- **Reranking**: Improves result quality by reordering retrieved documents

#### Knowledge Base
- Document ingestion and processing
- Vector database integration
- Semantic search capabilities
- Dynamic document updates

### 3. Processing Pipeline

#### Document Ingestion
1. **Document Loading**: Supports multiple file formats (PDF, DOCX, TXT, etc.)
2. **Preprocessing**: Cleans and normalizes text
3. **Chunking**: Splits documents into meaningful segments
4. **Embedding**: Converts text chunks into vector representations

#### Query Processing
1. **Query Understanding**: Analyzes and processes user questions
2. **Retrieval**: Finds relevant document chunks
3. **Reranking**: Orders results by relevance
4. **Context Formation**: Prepares information for the LLM
5. **Response Generation**: Creates coherent, informed answers

### 4. Support Systems

#### Configuration Management
- Component-specific settings
- Model configurations
- System parameters

#### Performance Monitoring
- Response time tracking
- Token usage and cost analysis
- Quality metrics

#### Utilities
- Text processing
- File handling
- API management
- Error handling

### 5. Prompt Engineering

#### Template System
- Dynamic prompt construction
- Context-aware templates
- Model-specific optimizations

#### Prompt Management
- Version control for prompts
- A/B testing framework
- Performance tracking

## How It All Works Together

1. **User Interaction**
   - User submits a query through the web interface
   - System processes and understands the intent

2. **Knowledge Retrieval**
   - Converts query to vector representation
   - Searches vector database for relevant information
   - Reranks results for maximum relevance

3. **Response Generation**
   - Combines query and retrieved context
   - Generates informed response using selected LLM
   - Streams response back to user in real-time

4. **Learning & Improvement**
   - Tracks interaction metrics
   - Updates knowledge base as needed
   - Adapts to improve future responses

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Set up your API keys in the configuration
3. Launch the application: `streamlit run app.py`
4. Start chatting or evaluating through the web interface

## Configuration

The system is highly configurable through:
- Environment variables
- Configuration files
- UI settings panel

## Contributing

We welcome contributions! Please see our contribution guidelines for more details.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

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