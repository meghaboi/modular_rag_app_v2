# Modular RAG Application

A modular and extensible Retrieval-Augmented Generation (RAG) application built with Streamlit. This application provides a flexible framework for implementing and evaluating different RAG configurations with various components.

## Project Overview

This project implements a modular RAG pipeline that allows users to:
- Upload and process documents
- Configure different components of the RAG pipeline
- Chat with the processed documents
- Evaluate different RAG configurations
- Compare performance across different models and settings

## Project Structure

```
├── pipeline/                # Pipeline related files
│   ├── rag_pipeline.py     # Main pipeline implementation
│   ├── pipeline_manager.py # Pipeline configuration
│   └── pipeline_utils.py   # Pipeline utilities
│
├── models/                  # All model implementations
│   ├── embeddings.py       # Embedding models
│   ├── llms.py            # Language models
│   ├── rerankers.py       # Reranking models
│   └── vector_stores.py   # Vector stores
│
├── ui/                     # User interface
│   ├── components.py      # UI components
│   └── sidebar.py         # Settings panel
│
├── utils/                  # Utility functions
│   ├── config.py          # Configuration
│   ├── evaluation.py      # Evaluation framework
│   └── tokenizer.py       # Token utilities
│
├── tests/                  # Test suite
├── data/                   # Data directory
├── examples/               # Example scripts
├── app.py                  # Main application
└── requirements.txt        # Dependencies
```

## Core Components

### 1. RAG Pipeline (`rag_pipeline.py`)
- Implements the core RAG functionality
- Handles document processing, embedding, retrieval, and generation
- Supports multiple embedding models, vector stores, and LLMs

### 2. Pipeline Manager (`pipeline_manager.py`)
- Manages pipeline configurations and component initialization
- Handles component registration and dependency management
- Provides interface for pipeline customization

### 3. UI Components (`ui_components.py`)
- Implements the chat interface
- Provides document upload functionality
- Displays evaluation results and metrics

### 4. Models and Stores
- **Embedding Models** (`embedding_models.py`): Various embedding model implementations
- **Vector Stores** (`vector_stores.py`): Different vector database implementations
- **LLM Models** (`llm_models.py`): Language model implementations
- **Rerankers** (`rerankers.py`): Document reranking implementations

### 5. Evaluation Framework (`evaluator.py`)
- Implements evaluation metrics and methodologies
- Supports comparison of different RAG configurations
- Provides detailed performance analysis

## Configuration

The application can be configured through:
1. Environment variables (`.env` file)
2. UI settings panel
3. Default configurations in `subject_configs.py`

Key configuration options include:
- Embedding model selection
- Vector store selection
- LLM model selection
- Chunking strategy and parameters
- Reranking model selection
- Hybrid search parameters

## Dependencies

Key dependencies include:
- Streamlit for the web interface
- LangChain for RAG pipeline components
- Various embedding models (OpenAI, Cohere, etc.)
- Vector stores (FAISS, Milvus)
- LLM providers (OpenAI, Anthropic, Google, Mistral)
- Text processing libraries (sentence-transformers, tiktoken)

## Getting Started

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables in `.env` file
4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Document Processing**
   - Upload documents through the UI
   - Configure processing parameters
   - Monitor processing status

2. **Chat Interface**
   - Interact with processed documents
   - View source contexts
   - Adjust retrieval parameters

3. **Evaluation**
   - Compare different configurations
   - View performance metrics
   - Export evaluation results

## Development

### Adding New Components

1. **New Embedding Model**
   - Implement in `embedding_models.py`
   - Register in pipeline manager
   - Add to configuration options

2. **New Vector Store**
   - Implement in `vector_stores.py`
   - Add connection handling
   - Update configuration options

3. **New LLM**
   - Implement in `llm_models.py`
   - Add API integration
   - Update configuration options

### Testing

Run tests using:
```bash
python -m pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes
4. Add tests
5. Submit pull request