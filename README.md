# A Modular and Extensible Framework for Retrieval-Augmented Generation

## 1. Introduction

Retrieval-Augmented Generation (RAG) has emerged as a pivotal technique in natural language processing, enhancing the capabilities of Large Language Models (LLMs) by grounding their responses in external knowledge. This approach mitigates common LLM limitations such as factual inaccuracies (hallucinations), outdated information, and non-transparent reasoning processes. By dynamically retrieving relevant information from a specified knowledge corpus, RAG systems can produce more accurate, contextually appropriate, and trustworthy responses.

Despite the conceptual simplicity of RAG, the practical implementation of these systems presents significant engineering and research challenges. The design space of a RAG pipeline is vast, encompassing a wide array of interchangeable components, each with its own set of configurations and performance characteristics. The choice of embedding model, vector store, reranker, and the LLM itself can have a profound impact on the system's overall efficacy. Furthermore, the lack of standardized evaluation methodologies makes it difficult to systematically compare different pipeline configurations and reproduce research findings.

This project introduces a modular and extensible framework for building and evaluating RAG systems, designed to address these challenges. It provides a highly flexible platform that allows researchers and practitioners to seamlessly swap components, configure pipeline parameters, and rigorously evaluate performance using a comprehensive suite of metrics. The framework is built on a set of abstract base classes and factories, enabling the easy integration of new models and techniques.

**Key features of this framework include:**

*   **Component Modularity:** Independent, swappable modules for every stage of the RAG pipeline, including document chunking, embedding, vector storage, retrieval, reranking, and generation.
*   **Multi-Provider Support:** Support for a wide range of models from leading providers such as OpenAI, Google, Anthropic, Cohere, Mistral, and Voyage, as well as open-source models.
*   **Flexible Pipeline Configuration:** A highly configurable pipeline that allows for fine-grained control over parameters such as chunk size, retrieval top-k, and hybrid search weighting.
*   **Comprehensive Evaluation Framework:** A built-in evaluation module that supports multiple backends (including RAGAS and custom LLM-based evaluators) and a wide array of metrics to assess retrieval and generation quality.

The overarching goal of this framework is to serve as a transparent and reproducible research platform for the systematic exploration of the RAG design space. By providing a standardized environment for experimentation, we aim to accelerate the development of robust, high-performing, and well-understood RAG applications. This document details the architecture, components, and functionalities of the framework, providing a comprehensive guide for its use in both research and production settings.

## 2. Background and Motivation

The development of effective Retrieval-Augmented Generation (RAG) systems, while promising, is fraught with challenges that can hinder research and development. The primary motivation behind this framework is to address these challenges by providing a structured, flexible, and reproducible environment for building and evaluating RAG pipelines.

### The Challenges

1.  **Component Lock-in and a Vast Design Space:** A RAG pipeline is a composite of multiple components: a chunking strategy, an embedding model, a vector store, a retriever, a reranker, and a large language model. The optimal choice for each component is highly dependent on the specific use case, data domain, and performance requirements. Traditional implementations often lead to tightly-coupled systems where replacing a single component requires significant engineering effort. This "component lock-in" stifles experimentation and makes it difficult to navigate the vast design space of possible pipeline configurations.

2.  **Complex Configuration Management:** The performance of a RAG system is sensitive to a multitude of hyperparameters, such as chunk size, chunk overlap, the number of retrieved documents (top-k), and reranking thresholds. Exploring the impact of these parameters is a complex, combinatorial problem. Without a systematic way to manage and evaluate these configurations, it is challenging to optimize a pipeline effectively.

3.  **The "Evaluation Hell" of RAG:** Evaluating the quality of a RAG system is a non-trivial task. It requires assessing multiple dimensions of performance, including the relevance of retrieved contexts, the factual accuracy (faithfulness) of the generated answer, and its pertinence to the user's query. There is no single metric that captures the overall quality of a RAG system, and implementing a comprehensive evaluation suite is a significant undertaking in itself. This makes it difficult to compare different pipelines and understand the trade-offs between them.

4.  **Lack of Standardization and Reproducibility:** The absence of a standardized framework for RAG development makes it difficult to reproduce research findings and compare results across different studies. This lack of reproducibility is a major obstacle to scientific progress in the field.

### Our Solution: A Modular Framework

This framework was designed to directly address these challenges. By enforcing a modular architecture through abstract base classes and factory patterns, it decouples the components of the RAG pipeline, allowing for effortless experimentation. Researchers can easily swap out an embedding model or a vector store to assess its impact on performance, without altering the rest of the pipeline.

The framework's centralized configuration system simplifies the management of hyperparameters, while the comprehensive, multi-backend evaluation module provides the tools necessary to rigorously assess performance across a wide range of metrics. By providing a standardized platform for building and evaluating RAG systems, this project aims to foster a more systematic and reproducible approach to RAG research and development, ultimately accelerating the creation of more powerful and reliable generative AI applications.

## 3. System Architecture

The framework is designed around a highly modular and extensible architecture that promotes the principles of abstraction, encapsulation, and dependency inversion. This design allows for the seamless interchange of components and facilitates rapid experimentation with different RAG pipeline configurations.

### Core Architectural Principles

The architecture is founded on two key design patterns:

1.  **Abstract Base Classes (ABCs):** Each core component of the RAG pipeline is defined by an abstract base class (ABC) that establishes a common interface, or "contract." For instance, any embedding model integrated into the framework must inherit from the `EmbeddingModel` ABC and implement its methods, such as `embed_query()` and `embed_documents()`. This ensures that the rest of the application can interact with any embedding model in a consistent way, regardless of its underlying implementation.

2.  **The Factory Pattern:** To decouple the pipeline from the concrete implementations of these components, the framework employs the Factory design pattern. For each component type, a corresponding factory class (e.g., `EmbeddingModelFactory`, `VectorStoreFactory`) is responsible for creating instances of the concrete classes. The main pipeline code requests a component from the factory based on a configuration setting (e.g., an enum type), rather than instantiating a specific class directly.

This combination of ABCs and factories is the cornerstone of the framework's modularity. It allows developers to add new components (e.g., a new vector store) by simply creating a new class that adheres to the ABC interface and registering it with the corresponding factory, often without needing to modify the core pipeline logic.

### Architectural Flow

The diagram below illustrates the architectural flow, from configuration to execution:

```
+------------------+      +---------------------+      +----------------------+
|   UI / Config    |----->|      Factories      |----->|  Component Instances |
| (e.g., Streamlit)|      | (e.g., LLMFactory)  |      |  (e.g., OpenAILLM)   |
+------------------+      +----------+----------+      +-----------+----------+
                                     |                           |
                                     | Creates                   | Injected into
                                     |                           |
                                     v                           v
+------------------------------------+---------------------------+----------+
|                               RAG Pipeline (Orchestrator)                 |
|                                                                           |
|  [Index] -> [Retrieve] -> [Rerank] -> [Generate] -> [Evaluate]            |
+---------------------------------------------------------------------------+
```

1.  **Configuration:** The process begins with the user selecting the desired components and parameters through the user interface (or a configuration file).

2.  **Instantiation via Factories:** The main application logic takes these configuration settings and uses the respective factories to instantiate the concrete component classes. For example, if the user selects "OpenAI" as the LLM, the `LLMFactory` creates an instance of the `OpenAIGPT` class.

3.  **Pipeline Assembly:** These component instances (an embedding model, a vector store, an LLM, etc.) are then injected into the `RAGPipeline` orchestrator during its initialization.

4.  **Execution:** The `RAGPipeline` executes the end-to-end RAG workflow. Because it operates on the component abstractions (the ABCs), it can orchestrate the flow of data without being coupled to any specific implementation. It calls `embed_documents()` on whichever embedding model it was given, `search()` on whichever vector store it was given, and so on.

This architecture ensures a clean separation of concerns, making the system easy to maintain, extend, and adapt to the rapidly evolving landscape of generative AI technologies.

## 4. Core Components: A Deep Dive

The heart of the framework is its library of modular, interchangeable components. Each component type is defined by an abstract base class, ensuring a consistent interface, and is instantiated via a factory, allowing for configuration-driven pipeline assembly. This section provides a detailed look at each component category.

### 4.1. Chunking Strategies

The first step in any RAG pipeline is to process the source documents into manageable pieces, or "chunks." The chunking strategy has a significant impact on retrieval quality, as it defines the units of information that will be embedded and retrieved. An effective chunking strategy aims to create chunks that are semantically coherent and self-contained, yet small enough to be efficiently processed by embedding models.

The framework provides several chunking strategies, each with its own strengths, accessible via the `ChunkingStrategyFactory`.

*   **Paragraph-based Chunking (`ParagraphChunking`):** This strategy splits documents along paragraph boundaries (`\n\n`). It is a simple yet effective approach that often preserves the logical structure of the original document. It is well-suited for well-structured texts like articles or reports.

*   **Sliding Window Chunking (`SlidingWindowChunking`):** This method uses a fixed-size token window that slides across the document. The `chunk_size` parameter defines the size of the window, and the `chunk_overlap` parameter specifies how many tokens should be shared between adjacent chunks. This approach is useful for dense, unstructured text where topics may span multiple paragraphs.

*   **Hierarchical Chunking (`HierarchicalChunking`):** For complex documents with nested structures, hierarchical chunking provides a multi-scale view of the content. It first splits the document into smaller, paragraph-based chunks (Level 1) and then creates progressively larger summary chunks that encompass multiple smaller chunks (Level 2 and above). This allows the retrieval system to access both fine-grained details and broader context.

*   **Semantic Chunking (`SemanticChunking`):** This advanced strategy attempts to divide the text based on semantic meaning rather than fixed sizes or paragraph breaks. It analyzes the semantic similarity between adjacent sentences or paragraphs and inserts a chunk boundary where the similarity drops below a certain threshold, indicating a topic shift. This can lead to more contextually coherent chunks, improving the quality of retrieval.

### 4.2. Embedding Models

Once the documents are chunked, the next step is to convert each chunk into a numerical vector representation using an embedding model. These embeddings capture the semantic meaning of the text, allowing for similarity-based retrieval. The choice of embedding model is critical, as the quality of the embeddings directly impacts the relevance of the retrieved documents.

All embedding models in the framework inherit from the `EmbeddingModel` base class and can be instantiated via the `EmbeddingModelFactory`. The framework supports a wide range of state-of-the-art models from various providers:

*   **OpenAI:** Utilizes the powerful `text-embedding-3-large` model.
*   **Cohere:** Implements Cohere's high-performance embedding models (e.g., `embed-v4.0`).
*   **Google Gemini:** Provides access to Google's `gemini-embedding` models.
*   **Mistral:** Integrates with Mistral's `mistral-embed` model, including robust handling of API rate limits.
*   **Voyage AI:** Supports Voyage's family of embedding models, which are highly optimized for RAG tasks.
*   **Qwen (Alibaba):** Includes support for the open-source `Qwen3-Embedding` model, which can be run locally using Hugging Face Transformers.

This variety of models allows researchers to experiment with different embedding spaces and trade-offs between performance, cost, and embedding dimensionality.

### 4.3. Vector Stores

The embedding vectors are stored and indexed in a vector store to enable efficient retrieval. The vector store is a critical piece of infrastructure that allows the system to perform fast similarity searches over millions or even billions of vectors. The framework provides several options for vector storage, each implementing the `VectorStore` interface.

*   **FAISS (`FAISSVectorStore`):** An acronym for Facebook AI Similarity Search, FAISS is a highly efficient, open-source library for similarity search. The framework's FAISS implementation provides a simple, in-memory vector store that is excellent for rapid prototyping and smaller-scale experiments.

*   **Chroma (`ChromaVectorStore`):** Chroma is a popular open-source, in-memory vector store that is easy to set up and use. It provides another excellent option for local development and experimentation.

*   **Milvus (`MilvusVectorStore`):** For more scalable and production-ready deployments, the framework integrates with Milvus, a powerful and feature-rich open-source vector database. The Milvus implementation is designed with a graceful fallback mechanism: if it cannot connect to a running Milvus instance, it will automatically revert to a temporary in-memory store, ensuring that the application remains functional.

*   **Hybrid Search (`HybridVectorStore`):** Recognizing that pure semantic search can sometimes miss keyword-specific matches, the framework includes a `HybridVectorStore`. This component combines traditional sparse vector retrieval (like BM25) with dense vector retrieval (from an embedding model). By weighting and combining the results from both methods, hybrid search can often achieve superior retrieval performance, capturing both lexical and semantic relevance.

### 4.4. Rerankers

The initial retrieval step from the vector store is optimized for speed and recall, and may return a broad set of potentially relevant documents. A reranker can be added to the pipeline as a second-pass filtering stage to improve precision. The reranker takes the top-k documents from the initial retrieval and uses a more computationally intensive but powerful model to re-score them based on their relevance to the query. This ensures that the final context provided to the LLM is as relevant and noise-free as possible.

All rerankers implement the `Reranker` interface and are managed by the `RerankerFactory`.

*   **Proprietary Reranking Models (`CohereReranker`, `VoyageReranker`, `JinaReranker`):** The framework integrates with several state-of-the-art reranking models from providers like Cohere, Voyage AI, and Jina. These models are specifically trained for the reranking task and offer excellent performance.

*   **LLM-based Reranking (`LLMReranker`):** For maximum flexibility, the framework includes an `LLMReranker`. This component uses a general-purpose LLM (such as Claude 3.5 Sonnet) to perform the reranking. It does this by constructing a prompt that asks the LLM to act as a relevance judge, assessing each document against the query and returning a sorted list with relevance scores. This approach allows for complex, nuanced reranking logic to be expressed in natural language, making it a powerful tool for research and custom applications.

### 4.5. Large Language Models (LLMs)

The final component in the pipeline is the Large Language Model (LLM), which serves as the "brain" of the system. The LLM receives the user's original query along with the retrieved and reranked context, and its task is to synthesize a coherent, accurate, and helpful answer.

The framework's `StreamingLLM` abstract base class defines the interface for all generator models, ensuring that they support both standard `generate()` and `stream_generate()` methods. This allows the application to be used in both interactive, real-time chat applications (via streaming) and for offline evaluation.

The `LLMFactory` provides access to a wide range of cutting-edge models from the leading providers:

*   **OpenAI:** GPT-3.5, GPT-4, and their variants.
*   **Google:** The Gemini family of models.
*   **Anthropic:** The Claude family, including the Opus, Sonnet, and Haiku models.
*   **Mistral:** The Mistral family of high-performance models.

This extensive support allows users to easily benchmark the performance of different LLMs and select the one that best fits their needs in terms of quality, cost, and speed.

## 5. The RAG Pipeline Explained

The `RAGPipeline` class, located in `pipeline/rag_pipeline.py`, is the central orchestrator that brings all the individual components together to perform the end-to-end Retrieval-Augmented Generation task. It is designed to be a clear and sequential workflow, from initial document processing to final response generation.

The pipeline's operation can be divided into two main phases: **Indexing** and **Runtime Execution**.

### 5.1. Indexing Phase

Before the pipeline can answer questions, it must first process and index the knowledge source. This is handled by the `index_documents()` method and involves the following steps:

1.  **Document Loading:** The raw text from the source document is loaded into memory.
2.  **Chunking:** The text is passed to the configured `ChunkingStrategy` (e.g., `ParagraphChunking`), which splits it into a list of smaller text chunks.
3.  **Embedding:** The list of text chunks is then processed by the selected `EmbeddingModel` (e.g., `OpenAIEmbedding`), which converts each chunk into a semantic vector embedding.
4.  **Storage:** Finally, the chunks and their corresponding embeddings are loaded into the chosen `VectorStore` (e.g., `FAISSVectorStore`), which creates an index for efficient similarity searching.

Once the indexing phase is complete, the pipeline is ready to handle user queries.

### 5.2. Runtime Execution Phase

When a user submits a query, the `run()` or `stream_run()` method is called, which executes the following steps in real-time:

1.  **Query Embedding:** The user's query string is passed to the `EmbeddingModel` to be converted into a vector embedding.
2.  **Retrieval:** This query embedding is used to search the `VectorStore`. The store performs a similarity search and returns the `top-k` most relevant document chunks from the indexed knowledge source.
3.  **Reranking (Optional):** If a `Reranker` component has been configured in the pipeline, the retrieved chunks are passed to it. The reranker re-scores the chunks for relevance to the query and returns a new, more accurately sorted list. This step helps to refine the context and reduce noise.
4.  **Generation:** The top-ranked chunks are formatted and combined with the original query into a comprehensive prompt. This prompt is then sent to the configured `LLM`. The LLM uses the provided context to synthesize a factual and relevant answer.
5.  **Streaming:** The `stream_run()` method provides the same functionality but yields the LLM's response token-by-token as it is generated. This is crucial for providing a responsive user experience in interactive applications like the provided Streamlit UI.

This clear separation of concerns and sequential flow make the pipeline easy to understand, debug, and extend.

## 6. Comprehensive Evaluation Framework

A key contribution of this framework is its comprehensive and modular evaluation suite, designed to facilitate rigorous and reproducible assessment of RAG pipeline performance. The ability to systematically evaluate different component configurations is critical for both research and production optimization. The evaluation framework is located in `models/evaluator.py`.

### 6.1. Evaluation Backends

The framework supports multiple evaluation backends, each leveraging different techniques and models. This allows users to choose the evaluation approach that best suits their needs for accuracy, cost, and speed. All evaluators are accessible via the `EvaluatorFactory`.

*   **LLM-as-Judge Evaluators (`BuiltinEvaluator`, `LangSmithEvaluator`, `DeepEvaluator`, `CustomEvaluator`):** A significant portion of the evaluation suite is based on the "LLM-as-Judge" paradigm, where a powerful LLM is used to score the quality of the RAG pipeline's output. These evaluators use carefully crafted prompts to ask a model (like GPT-4 or Claude 3 Opus) to rate the pipeline's performance on various metrics. The framework includes several variations on this theme, using different models and prompt strategies.

*   **RAGAS Integration (`RAGASEvaluatorV2`):** The framework includes a direct integration with RAGAS, a popular open-source framework specifically designed for RAG pipeline evaluation. This provides access to a standardized, well-regarded set of metrics that are widely used in the research community, further promoting reproducibility.

### 6.2. Key Evaluation Metrics

The evaluation framework supports a wide range of metrics that cover the different aspects of RAG performance, from retrieval quality to generation quality. The core metrics include:

*   **Faithfulness:** This metric measures the factual consistency of the generated answer with respect to the provided context. A high faithfulness score indicates that the LLM is not "hallucinating" information and is grounding its answer in the retrieved documents. It is typically measured by breaking the generated answer into individual statements and checking if each statement can be verified by the context.

*   **Answer Relevancy:** This assesses how relevant the generated answer is to the user's query. It helps to ensure that the LLM is not only faithful to the context but is also directly addressing the user's question.

*   **Context Precision:** This metric evaluates the signal-to-noise ratio of the retrieved context. It answers the question: "Of the documents that were retrieved, how many are actually relevant to the query?" A high precision score means that the retriever is not introducing irrelevant or distracting information into the context.

*   **Context Recall:** This measures the ability of the retriever to find all the necessary information to answer the question. It is typically evaluated against a "ground truth" answer and answers the question: "Does the retrieved context contain all the information required to generate the correct answer?"

*   **Answer Correctness:** This metric compares the generated answer to a ground-truth (or "reference") answer, assessing its factual accuracy and completeness. This is often considered the ultimate measure of the pipeline's end-to-end performance.

By using this evaluation framework, researchers can conduct systematic A/B tests and ablation studies, comparing different chunking strategies, embedding models, or rerankers to quantitatively measure their impact on the final output quality.

## 7. Practical Usage and Extensibility

This section provides practical guidance on how to set up, run, and extend the framework.

### 7.1. Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Configure API Keys:**
    Create a file named `.env` in the root directory of the project and add your API keys. The framework will automatically load these keys.
    ```
    OPENAI_API_KEY="sk-..."
    COHERE_API_KEY="..."
    GEMINI_API_KEY="..."
    MISTRAL_API_KEY="..."
    VOYAGE_API_KEY="..."
    JINA_API_KEY="..."
    ANTHROPIC_API_KEY="..."
    LANGCHAIN_API_KEY="..." # Optional, for LangSmith integration
    ```

### 7.2. Running the Application

The framework includes an interactive Streamlit application for easy experimentation.

1.  **Run the app:**
    ```bash
    streamlit run app.py
    ```

2.  **Using the Interface:**
    The application will open in your web browser. The sidebar on the left allows you to configure every aspect of the RAG pipeline, from the chunking strategy to the LLM. You can upload a document, and the pipeline will be indexed automatically. The main interface allows you to switch between a "Chat" mode for interacting with the pipeline and an "Evaluation" mode for running systematic experiments.

### 7.3. Extending the Framework

The modular architecture makes it straightforward to add new components. Let's walk through an example of adding a new (hypothetical) "ExampleEmbedding" model.

1.  **Implement the Abstract Base Class:**
    In `models/embedding_models.py`, create a new class that inherits from `EmbeddingModel` and implements its required methods.

    ```python
    class ExampleEmbedding(EmbeddingModel):
        def __init__(self):
            # Initialization logic, e.g., API client
            pass

        def embed_query(self, query: str) -> List[float]:
            # Implementation for embedding a single query
            pass

        def embed_documents(self, documents: List[str]) -> List[List[float]]:
            # Implementation for embedding a list of documents
            pass

        @property
        def dimension(self) -> int:
            # Return the dimension of the embeddings
            return 768
    ```

2.  **Add to the Enum:**
    In `utils/enums.py`, add a new entry to the `EmbeddingModelType` enum.

    ```python
    class EmbeddingModelType(Enum):
        # ... existing enums
        EXAMPLE = "Example"
    ```

3.  **Register in the Factory:**
    In `models/embedding_models.py`, register the new class in the `EmbeddingModelFactory`.

    ```python
    class EmbeddingModelFactory:
        _model_map: Dict[EmbeddingModelType, Type[EmbeddingModel]] = {
            # ... existing models
            EmbeddingModelType.EXAMPLE: ExampleEmbedding
        }
        # ... rest of the factory code
    ```

Once these steps are completed, the "Example" embedding model will automatically appear in the Streamlit UI and can be used in any pipeline configuration. The same process applies to adding new vector stores, rerankers, LLMs, or chunking strategies.

## 8. Conclusion and Future Research

### 8.1. Conclusion

This framework provides a robust and flexible platform for the systematic construction and evaluation of Retrieval-Augmented Generation systems. By enforcing a modular architecture, it addresses the key challenges of component lock-in and configuration complexity that often hinder RAG development. The comprehensive suite of interchangeable components, combined with a powerful, multi-faceted evaluation framework, empowers researchers and practitioners to navigate the vast RAG design space with greater efficiency and rigor. Ultimately, this project aims to foster a more standardized and reproducible approach to RAG research, accelerating the development of the next generation of reliable and high-performing generative AI applications.

### 8.2. Future Research Directions

The extensibility of the framework opens up numerous avenues for future research. Some promising directions include:

*   **Advanced Routing and Agentic RAG:** The modularity of the pipeline is a perfect foundation for building more complex, agentic systems. A "meta-pipeline" or "router" could be developed to dynamically select the most appropriate components (e.g., different retrievers or LLMs) based on the nature of the incoming query.

*   **Automated Pipeline Optimization:** The framework's configurability could be leveraged to build an automated optimization layer. This could involve using techniques like Bayesian optimization or reinforcement learning to automatically search the combinatorial space of components and hyperparameters to find the optimal pipeline configuration for a specific task or dataset.

*   **Deeper Evaluation Metrics:** The evaluation suite could be expanded to include more nuanced metrics. This could include assessing the cost-performance trade-off of different components, measuring end-to-end latency, or evaluating the pipeline's ability to handle conversational context and follow-up questions.

*   **Broader Data Source Integration:** Future work could focus on expanding the range of supported data sources beyond plain text. This could include adding pre-processing modules for structured data (like CSVs or SQL databases) and more complex unstructured documents (like PDFs with tables and images or PowerPoint presentations).
