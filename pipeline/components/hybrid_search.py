from typing import List, Tuple
import re
from rank_bm25 import BM25Okapi
import numpy as np

class HybridSearch:
    """Combines dense vector search with sparse keyword search (BM25)"""
    
    def __init__(self, alpha: float = 0.5):
        """
        Initialize hybrid search
        
        Args:
            alpha: Weight for vector search scores (1-alpha = weight for BM25)
        """
        self.alpha = alpha
        self.documents = []
        self.bm25 = None
        self.doc_embeddings = None
        
    def index_documents(self, documents: List[str], embeddings: List[List[float]]) -> None:
        """Index documents for both vector search and BM25"""
        self.documents = documents
        self.doc_embeddings = np.array(embeddings)
        
        # Tokenize documents for BM25
        tokenized_docs = [self._tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25"""
        # Convert to lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\w+', text)
        return tokens
    
    def search(self, query: str, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Perform hybrid search using both vector similarity and BM25
        
        Args:
            query: Text query for keyword search
            query_embedding: Vector embedding of the query
            top_k: Number of results to return
            
        Returns:
            List of tuples with (document, score)
        """
        if not self.documents or len(self.documents) == 0:
            return []
        
        # Vector search scores
        vector_scores = self._vector_search(query_embedding)
        
        # BM25 search scores
        bm25_scores = self._bm25_search(query)
        
        # Normalize scores to [0, 1] range
        vector_scores_norm = self._normalize_scores(vector_scores)
        bm25_scores_norm = self._normalize_scores(bm25_scores)
        
        # Combine scores with alpha weighting
        combined_scores = self.alpha * vector_scores_norm + (1 - self.alpha) * bm25_scores_norm
        
        # Get top k results
        top_indices = np.argsort(-combined_scores)[:top_k]
        
        results = [(self.documents[i], combined_scores[i]) for i in top_indices]
        return results
    
    def _vector_search(self, query_embedding: List[float]) -> np.ndarray:
        """Calculate vector similarity scores for all documents"""
        query_embedding = np.array(query_embedding)
        
        # Calculate cosine similarity
        # Normalize vectors for cosine similarity
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm
            
        # Calculate dot product for normalized vectors (equal to cosine similarity)
        doc_norms = np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True)
        normalized_embeddings = np.divide(self.doc_embeddings, doc_norms, 
                                         where=doc_norms != 0)
        
        similarities = np.dot(normalized_embeddings, query_embedding)
        return similarities
    
    def _bm25_search(self, query: str) -> np.ndarray:
        """Calculate BM25 scores for all documents"""
        query_tokens = self._tokenize(query)
        scores = np.array(self.bm25.get_scores(query_tokens))
        return scores
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range"""
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score == min_score:
            return np.ones_like(scores)
            
        normalized = (scores - min_score) / (max_score - min_score)
        return normalized 