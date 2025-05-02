# Configurable RAG System with Evaluation

This is a Streamlit application that allows you to build and evaluate a Retrieval-Augmented Generation (RAG) pipeline using different components and configurations. The application follows SOLID principles, making it easy to extend and modify.

## Features

- Upload text files for knowledge base creation
- Select embedding models (OpenAI, Cohere)
- Choose reranking models (Cohere)
- Pick LLM models (OpenAI GPT-3.5/4, Google Gemini)
- Configure vector store (FAISS)
- Evaluate RAG performance with multiple metrics:
  - Answer Relevance
  - Context Relevance
  - Groundedness
  - Faithfulness

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd rag-evaluation-app
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL displayed in the terminal (typically http://localhost:8501).

## Usage Instructions

1. **Upload a Knowledge Base**:
   - Click "Browse files" in the sidebar to upload a .txt file that will serve as your knowledge base.

2. **Configure Components**:
   - Select your preferred embedding model, reranker, LLM, and vector store from the dropdown menus.
   - Enter your API keys for OpenAI, Cohere, and/or Gemini as needed.
   - Set the number of documents to retrieve using the slider.

3. **Select Evaluation Metrics**:
   - Check the evaluation metrics you want to use.

4. **Enter a Query**:
   - Type your question in the "Enter your question" text area.
   - Optionally, provide a ground truth answer for evaluation.

5. **Process the Query**:
   - Click the "Process Query" button to execute the RAG pipeline.

6. **Review Results**:
   - The system will display the answer and retrieved contexts.
   - If you provided a ground truth answer, evaluation metrics will be calculated and displayed.

## Extending the Application

The application is designed with SOLID principles, making it easy to add new components:

- Add new embedding models by extending the `EmbeddingModel` class
- Add new rerankers by extending the `Reranker` class
- Add new vector stores by extending the `VectorStore` class
- Add new LLM models by extending the `LLM` class

Update the corresponding factory classes to make new components available in the UI.

## Project Structure

- `app.py`: Main Streamlit application
- `embedding_models.py`: Embedding model implementations and factory
- `rerankers.py`: Reranker implementations and factory
- `vector_stores.py`: Vector store implementations and factory
- `llm_models.py`: LLM model implementations and factory
- `evaluation.py`: RAG evaluation metrics
- `rag_pipeline.py`: RAG pipeline implementation