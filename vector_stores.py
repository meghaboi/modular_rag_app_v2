from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import numpy as np
from enums import VectorStoreType
import logging
import os

class VectorStore(ABC):
    """Abstract base class for vector stores following Interface Segregation Principle"""
    
    @abstractmethod
    def add_documents(self, documents: List[str], embeddings: List[List[float]]) -> None:
        """Add documents and their embeddings to the vector store"""
        pass
    
    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar documents using the query embedding"""
        pass

class FAISSVectorStore(VectorStore):
    """FAISS vector store implementation"""
    
    def __init__(self):
        """Initialize the FAISS vector store"""
        import faiss
        
        self._documents = []
        self._index = None
        self._dimension = None
    
    def add_documents(self, documents: List[str], embeddings: List[List[float]]) -> None:
        """Add documents and their embeddings to the vector store"""
        import faiss
        
        if not documents or not embeddings:
            return
        
        self._documents = documents
        
        # Convert embeddings to numpy array
        embeddings_np = np.array(embeddings).astype('float32')
        self._dimension = embeddings_np.shape[1]
        
        # Create FAISS index
        self._index = faiss.IndexFlatL2(self._dimension)
        self._index.add(embeddings_np)
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar documents using the query embedding"""
        if self._index is None:
            return []
        
        # Convert query embedding to numpy array
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
    """Chroma vector store implementation"""
    
    def __init__(self):
        """Initialize the Chroma vector store"""
        import chromadb
        from chromadb.config import Settings
        import uuid
        
        # Create a temporary client with in-memory storage
        self._client = chromadb.Client(Settings(anonymized_telemetry=False))
        
        # Create a collection with a unique ID
        self._collection_name = f"collection_{str(uuid.uuid4())[:8]}"
        self._collection = self._client.create_collection(name=self._collection_name)
        
        # Store document mapping
        self._id_to_doc = {}
    
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
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar documents using the query embedding"""
        if not self._id_to_doc:
            return []
        
        # Query collection
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, len(self._id_to_doc))
        )
        
        # Format results
        documents = results.get('documents', [[]])[0]
        distances = results.get('distances', [[]])[0]
        
        return [(doc, float(dist)) for doc, dist in zip(documents, distances)]

class QdrantVectorStore(VectorStore):
    """Qdrant vector store implementation with configurable connection options"""
    
    def __init__(self, 
                collection_name: str = "default_collection", 
                host: str = "localhost", 
                port: int = 6333,
                use_in_memory: bool = True):
        """Initialize the Qdrant vector store
        
        Args:
            collection_name: Name of the Qdrant collection to use
            host: Hostname or IP address of the Qdrant server
            port: Port of the Qdrant server
            use_in_memory: If True, use in-memory storage instead of connecting to a server
        """
        # Import inside method to avoid requiring qdrant_client if not using this store
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        import uuid
        
        # Generate a unique collection name if one isn't provided
        self._collection_name = f"{collection_name}_{str(uuid.uuid4())[:8]}"
        
        # Initialize client with appropriate settings
        if use_in_memory:
            # Use in-memory storage - no server needed
            self._client = QdrantClient(location=":memory:")
            print("Using in-memory Qdrant storage")
        else:
            # Connect to server
            try:
                self._client = QdrantClient(host=host, port=port)
                print(f"Connected to Qdrant server at {host}:{port}")
            except Exception as e:
                print(f"Failed to connect to Qdrant server: {str(e)}")
                print("Falling back to in-memory storage")
                self._client = QdrantClient(location=":memory:")
        
        # Track if collection is created and dimension is set
        self._collection_created = False
        self._dimension = None
        
        # Store document mapping
        self._id_to_doc = {}
    
    def add_documents(self, documents: List[str], embeddings: List[List[float]]) -> None:
        """Add documents and their embeddings to the vector store"""
        from qdrant_client.models import Distance, VectorParams, PointStruct
        
        if not documents or not embeddings:
            return
        
        # Get dimension from first embedding
        if self._dimension is None:
            self._dimension = len(embeddings[0])
        
        # Create collection if it doesn't exist
        if not self._collection_created:
            try:
                self._client.recreate_collection(
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(
                        size=self._dimension,
                        distance=Distance.COSINE  # Using cosine distance, could be parameterized
                    )
                )
                self._collection_created = True
            except Exception as e:
                print(f"Error creating Qdrant collection: {str(e)}")
                raise
        
        # Generate unique IDs for documents
        ids = [i for i in range(len(documents))]
        
        # Store mapping of IDs to documents
        self._id_to_doc = {doc_id: doc for doc_id, doc in zip(ids, documents)}
        
        # Create point objects for insertion
        points = [
            PointStruct(
                id=id,
                vector=embedding,
                payload={"text": document}  # Store document in payload for retrieval
            )
            for id, document, embedding in zip(ids, documents, embeddings)
        ]
        
        # Insert points in batches to avoid memory issues with large document sets
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            try:
                self._client.upsert(
                    collection_name=self._collection_name,
                    points=batch
                )
            except Exception as e:
                print(f"Error inserting documents into Qdrant: {str(e)}")
                raise
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar documents using the query embedding
        
        Args:
            query_embedding: The embedding of the query
            top_k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        if not self._collection_created:
            return []
        
        try:
            # Search for similar vectors
            search_results = self._client.search(
                collection_name=self._collection_name,
                query_vector=query_embedding,
                limit=min(top_k, len(self._id_to_doc)),
                with_payload=True
            )
            
            # Format results as (document, score) tuples
            results = []
            for result in search_results:
                document = result.payload.get("text")
                # Convert similarity score to distance for consistency with other vector stores
                distance = 1.0 - result.score
                results.append((document, float(distance)))
            
            return results
        except Exception as e:
            print(f"Error searching Qdrant vector store: {str(e)}")
            return []

class MilvusVectorStore(VectorStore):
    """Milvus vector store implementation with fallback to in-memory when server isn't available"""
    
    def __init__(self, 
                collection_name: str = "default_collection", 
                host: str = "localhost", 
                port: int = 19530,
                force_in_memory: bool = False):
        """Initialize the Milvus vector store
        
        Args:
            collection_name: Name of the Milvus collection to use
            host: Hostname or IP address of the Milvus server
            port: Port of the Milvus server
            force_in_memory: If True, skip Milvus connection and use in-memory only
        """
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("MilvusVectorStore")
        
        # Initialize variables
        self._documents = []
        self._embeddings = []
        self._collection = None
        self._using_milvus = not force_in_memory
        self._collection_name = collection_name
        
        # Try to connect to Milvus server if not forcing in-memory
        if not force_in_memory:
            try:
                # Import Milvus
                from pymilvus import connections, Collection, utility
                import uuid
                
                # Generate a unique collection name
                self._collection_name = f"{collection_name}_{str(uuid.uuid4())[:8]}"
                
                # Connect to Milvus
                connections.connect("default", host=host, port=port)
                self.logger.info(f"Connected to Milvus server at {host}:{port}")
                self._using_milvus = True
                
                # Check if we can perform basic operations (to verify connection)
                utility.get_server_version()
                
            except Exception as e:
                self.logger.warning(f"Failed to connect to Milvus server: {str(e)}")
                self.logger.warning("Falling back to in-memory vector storage")
                self._using_milvus = False
        else:
            self.logger.info("Using in-memory vector storage (forced)")
    
    def add_documents(self, documents: List[str], embeddings: List[List[float]]) -> None:
        """Add documents and their embeddings to the vector store"""
        if not documents or not embeddings:
            return
            
        if self._using_milvus:
            try:
                self._add_documents_milvus(documents, embeddings)
            except Exception as e:
                self.logger.error(f"Error adding documents to Milvus: {str(e)}")
                self.logger.warning("Falling back to in-memory storage")
                self._using_milvus = False
                # Store documents in memory instead
                self._documents = documents
                self._embeddings = np.array(embeddings).astype('float32')
        else:
            # Store documents in memory
            self._documents = documents
            self._embeddings = np.array(embeddings).astype('float32')
            self.logger.info(f"Stored {len(documents)} documents in memory")
    
    def _add_documents_milvus(self, documents: List[str], embeddings: List[List[float]]) -> None:
        """Helper method to add documents to Milvus"""
        from pymilvus import Collection, utility
        
        # Get dimension from first embedding
        dimension = len(embeddings[0])
        
        # Check if collection exists and create it if needed
        if utility.has_collection(self._collection_name):
            self._collection = Collection(self._collection_name)
        else:
            try:
                # Try to import schema classes
                try:
                    from pymilvus import CollectionSchema, FieldSchema, DataType
                except ImportError:
                    try:
                        from pymilvus.orm import CollectionSchema, FieldSchema
                        from pymilvus.orm.types import DataType
                    except ImportError:
                        self.logger.error("Could not import Milvus schema classes")
                        raise ImportError("Milvus schema classes not found")
                
                # Define fields for the collection
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension)
                ]
                
                # Create collection schema and collection
                schema = CollectionSchema(fields)
                self._collection = Collection(name=self._collection_name, schema=schema)
                
                # Create index for vector field
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "HNSW",
                    "params": {"M": 8, "efConstruction": 64}
                }
                self._collection.create_index(field_name="embedding", index_params=index_params)
                self.logger.info(f"Created Milvus collection: {self._collection_name}")
            except Exception as e:
                self.logger.error(f"Error creating Milvus collection: {str(e)}")
                raise
        
        # Prepare data for insertion - format depends on pymilvus version
        try:
            # Try dict-based insert format first
            entities = [
                {"text": doc, "embedding": emb} 
                for doc, emb in zip(documents, embeddings)
            ]
            self._collection.insert(entities)
        except Exception as e:
            try:
                # Try list-based format as fallback
                data = [
                    documents,  # "text" field
                    embeddings   # "embedding" field
                ]
                self._collection.insert(data)
            except Exception as e2:
                self.logger.error(f"Failed both insert methods: {str(e)} | {str(e2)}")
                raise
        
        # Flush data
        self._collection.flush()
        self.logger.info(f"Added {len(documents)} documents to Milvus")
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar documents using the query embedding"""
        if self._using_milvus:
            try:
                return self._search_milvus(query_embedding, top_k)
            except Exception as e:
                self.logger.error(f"Error searching Milvus: {str(e)}")
                self.logger.warning("Falling back to in-memory search")
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
                self.logger.info("Disconnected from Milvus server")
            except:
                pass

class HybridVectorStore(VectorStore):
    """Vector store that uses hybrid search"""
    
    def __init__(self, alpha: float = 0.5):
        """
        Initialize hybrid vector store
        
        Args:
            alpha: Weight for vector search scores (1-alpha = weight for BM25)
        """
        
        from rag_pipeline import HybridSearch   

        self.hybrid_search = HybridSearch(alpha=alpha)
        self.documents = []
        self.embeddings = []
        
    def add_documents(self, documents: List[str], embeddings: List[List[float]]) -> None:
        """Add documents and their embeddings to the store"""
        self.documents = documents
        self.embeddings = embeddings
        self.hybrid_search.index_documents(documents, embeddings)
        
    def search(self, query_embedding: List[float], top_k: int = 5, query: str = None) -> List[Tuple[str, float]]:
        """
        Search for similar documents using hybrid search
        
        Args:
            query_embedding: Vector embedding of the query
            top_k: Number of results to return
            query: Text query for BM25 component (required for hybrid search)
            
        Returns:
            List of tuples with (document, score)
        """
        if query is None:
            raise ValueError("Text query is required for hybrid search")
            
        return self.hybrid_search.search(query, query_embedding, top_k)

class VectorStoreFactory:
    """Factory for creating vector stores (Factory Pattern)"""
    
    @staticmethod
    def create_store(store_name: str, **kwargs) -> 'VectorStore':
        """Create a vector store based on the store name"""
        if store_name == VectorStoreType.FAISS:
            from vector_stores import FAISSVectorStore
            return FAISSVectorStore()
        elif store_name == VectorStoreType.CHROMA:
            from vector_stores import ChromaVectorStore
            return ChromaVectorStore()
        elif store_name == VectorStoreType.MILVUS:
            from vector_stores import MilvusVectorStore
            return MilvusVectorStore()
        elif store_name == VectorStoreType.HYBRID:
            alpha = kwargs.get('alpha', 0.5)
            return HybridVectorStore(alpha=alpha)
        else:
            raise ValueError(f"Unsupported vector store: {store_name}")