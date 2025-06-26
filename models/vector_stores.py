from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from utils.enums import VectorStoreType
import logging
import uuid

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class VectorStore(ABC):
    """Abstract base class for vector stores following Interface Segregation Principle"""
    
    @abstractmethod
    def add_documents(self, documents: List[str], embeddings: List[List[float]]) -> None:
        """Add documents and their embeddings to the vector store"""
        pass
    
    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5, **kwargs) -> List[Tuple[str, float]]:
        """Search for similar documents using the query embedding."""
        pass

class FAISSVectorStore(VectorStore):
    """FAISS vector store implementation."""

    def __init__(self, **kwargs):
        """Initialize the FAISS vector store."""
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            log.error("FAISS not installed. Please run 'pip install faiss-cpu' or 'pip install faiss-gpu'.")
            raise

        self._documents: List[str] = []
        self._index: Optional[self.faiss.Index] = None
        self._dimension: Optional[int] = None

    def add_documents(self, documents: List[str], embeddings: List[List[float]]) -> None:
        """Add documents and their embeddings to the vector store."""
        
        if not documents or not embeddings:
            return
        
        self._documents = documents
        
        # Convert embeddings to a numpy array
        embeddings_np = np.array(embeddings).astype('float32')
        self._dimension = embeddings_np.shape[1]
        
        # Create FAISS index
        self._index = self.faiss.IndexFlatIP(self._dimension)
        self._index.add(embeddings_np)
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar documents using the query embedding"""
        if self._index is None:
            return []
        
        # Convert query to numpy array
        query_np = np.array([query_embedding]).astype('float32')
        
        # Search FAISS index
        distances, indices = self._index.search(query_np, min(top_k, len(self._documents)))
        
        # Return documents with their distances
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self._documents):
                results.append((self._documents[idx], float(distances[0][i])))
        
        return results

class ChromaVectorStore(VectorStore):
    """Chroma vector store implementation."""

    def __init__(self, **kwargs):
        """Initialize the Chroma vector store."""
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            log.error("ChromaDB not installed. Please run 'pip install chromadb-client'.")
            raise

        self._client = chromadb.Client(Settings(anonymized_telemetry=False))
        self._collection_name = f"collection_{str(uuid.uuid4())[:8]}"
        self._collection = self._client.create_collection(name=self._collection_name)
        self._id_to_doc: Dict[str, str] = {}
    
    def add_documents(self, documents: List[str], embeddings: List[List[float]]) -> None:
        """Add documents and their embeddings to the vector store"""
        if not documents or not embeddings:
            return
        
        # Generate IDs for documents
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Store mapping of IDs to documents
        self._id_to_doc = {doc_id: doc for doc_id, doc in zip(ids, documents)}
        
        # Add documents to collection
        self._collection.add(
            embeddings=embeddings,
            documents=documents,
            ids=ids
        )
    
    def search(self, query_embedding: List[float], top_k: int = 5, **kwargs) -> List[Tuple[str, float]]:
        """Search for similar documents using the query embedding."""
        if not self._id_to_doc:
            return []

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, len(self._id_to_doc))
        )

        documents = results.get('documents', [[]])[0]
        distances = results.get('distances', [[]])[0]

        return list(zip(documents, map(float, distances)))

class MilvusVectorStore(VectorStore):
    """Milvus vector store implementation with fallback to in-memory."""

    def __init__(self, collection_name: str = "default_collection", force_in_memory: bool = False, **kwargs):
        """Initialize the Milvus vector store."""
        self._collection: Optional[Any] = None
        self._using_milvus = False
        self._collection_name = f"{collection_name}_{str(uuid.uuid4())[:8]}"
        self._documents: List[str] = []
        self._embeddings: Optional[np.ndarray] = None

        if not force_in_memory:
            self._connect_to_milvus(**kwargs)
        else:
            log.info("Using in-memory vector storage (forced).")

    def _connect_to_milvus(self, **kwargs):
        try:
            from pymilvus import connections, utility
            connections.connect("default", **kwargs)
            utility.get_server_version()  # Verify connection
            self._using_milvus = True
            log.info(f"Connected to Milvus server.")
        except Exception as e:
            log.warning(f"Failed to connect to Milvus server: {e}. Falling back to in-memory storage.")
            self._using_milvus = False
    
    def add_documents(self, documents: List[str], embeddings: List[List[float]]) -> None:
        """Add documents and their embeddings to the vector store."""
        if not documents or not embeddings:
            return
            
        if self._using_milvus:
            try:
                self._add_documents_milvus(documents, embeddings)
                return
            except Exception as e:
                log.error(f"Error adding documents to Milvus: {e}. Falling back to in-memory.", exc_info=True)
                self._using_milvus = False

        self._documents.extend(documents)
        new_embeddings = np.array(embeddings, dtype='float32')
        if self._embeddings is None:
            self._embeddings = new_embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, new_embeddings])
        log.info(f"Stored {len(documents)} documents in-memory.")
    
    def _add_documents_milvus(self, documents: List[str], embeddings: List[List[float]]) -> None:
        """Helper method to add documents to Milvus"""
        from pymilvus import Collection, utility

        dimension = len(embeddings[0])
        if not utility.has_collection(self._collection_name):
            fields = [
                {"name": "id", "dtype": "int64", "is_primary": True, "auto_id": True},
                {"name": "text", "dtype": "varchar", "max_length": 65535},
                {"name": "embedding", "dtype": "float_vector", "dim": dimension}
            ]
            schema = {"fields": fields, "description": "Document collection"}
            self._collection = Collection(name=self._collection_name, schema=schema)
            index_params = {"metric_type": "COSINE", "index_type": "HNSW", "params": {"M": 8, "efConstruction": 64}}
            self._collection.create_index(field_name="embedding", index_params=index_params)
            log.info(f"Created Milvus collection: {self._collection_name}")
        else:
            self._collection = Collection(self._collection_name)
        
        data = [documents, embeddings]
        self._collection.insert(data)
        self._collection.flush()
        log.info(f"Added {len(documents)} documents to Milvus.")
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar documents using the query embedding"""
        if self._using_milvus:
            try:
                return self._search_milvus(query_embedding, top_k)
            except Exception as e:
                log.error(f"Error searching Milvus: {e}. Falling back to in-memory search.")
                return self._search_in_memory(query_embedding, top_k)
        else:
            return self._search_in_memory(query_embedding, top_k)
    
    def _search_milvus(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Helper method to search in Milvus"""
        if not self._collection:
            return []
        
        # Load collection
        self._collection.load()
        
        # Search parameters
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 64}
        }
        
        # Search for similar vectors
        search_results = self._collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text"]
        )
        
        # Format results
        results = []
        for hits in search_results:
            for hit in hits:
                document = hit.entity.get("text")
                # Convert similarity to distance (0 is best)
                distance = 1.0 - hit.score
                results.append((document, float(distance)))
        
        return results
    
    def _search_in_memory(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar documents in memory when Milvus isn't available"""
        if not hasattr(self, '_embeddings') or len(self._embeddings) == 0:
            return []
        
        # Convert query to numpy array
        query_np = np.array(query_embedding).astype('float32')
        
        # Calculate cosine similarity
        dot_product = np.dot(self._embeddings, query_np)
        embedding_norms = np.linalg.norm(self._embeddings, axis=1)
        query_norm = np.linalg.norm(query_np)
        cosine_similarities = dot_product / (embedding_norms * query_norm)
        
        # Get top-k indices
        top_indices = np.argsort(-cosine_similarities)[:min(top_k, len(self._documents))]
        
        # Format results as (document, distance) where distance is 1-similarity
        results = [
            (self._documents[idx], float(1.0 - cosine_similarities[idx]))
            for idx in top_indices
        ]
        
        return results
    
    def __del__(self):
        """Clean up resources when the object is destroyed"""
        if self._using_milvus:
            try:
                from pymilvus import connections
                if self._collection:
                    self._collection.release()
                connections.disconnect("default")
                log.info("Disconnected from Milvus server")
            except:
                pass

class HybridVectorStore(VectorStore):
    """Vector store that uses hybrid search."""

    def __init__(self, alpha: float = 0.5, **kwargs):
        """Initialize hybrid vector store."""
        try:
            from pipeline.components.hybrid_search import HybridSearch
        except ImportError:
            log.error("HybridSearch component not found. Please check pipeline components.")
            raise

        self.hybrid_search = HybridSearch(alpha=alpha)
        self.documents: List[str] = []

    def add_documents(self, documents: List[str], embeddings: List[List[float]]) -> None:
        """Add documents and their embeddings to the store."""
        self.documents = documents
        self.hybrid_search.add_documents(documents, embeddings)

    def search(self, query_embedding: List[float], top_k: int = 5, **kwargs) -> List[Tuple[str, float]]:
        """Search for similar documents using hybrid search."""
        query = kwargs.get('query')
        if not query:
            raise ValueError("Hybrid search requires a 'query' text in kwargs.")
        if not self.documents:
            return []
        return self.hybrid_search.search(query, query_embedding, top_k)

class VectorStoreFactory:
    """Factory for creating vector stores."""

    _store_map: Dict[VectorStoreType, type] = {
        VectorStoreType.FAISS: FAISSVectorStore,
        VectorStoreType.CHROMA: ChromaVectorStore,
        VectorStoreType.MILVUS: MilvusVectorStore,
        VectorStoreType.HYBRID: HybridVectorStore,
    }

    @classmethod
    def create_store(cls, store_type: VectorStoreType, **kwargs) -> VectorStore:
        """Create a vector store based on the store type."""
        if store_type not in cls._store_map:
            raise ValueError(f"Unsupported vector store type: {store_type}")

        store_class = cls._store_map[store_type]
        return store_class(**kwargs)