from typing import List, Tuple
import re
from rank_bm25 import BM25Okapi
import numpy as np

class VectorSearcher:
    """Performs dense vector search."""
    def __init__(self, documents: List[str], embeddings: List[List[float]]):
        self.documents = documents
        self.doc_embeddings = np.array(embeddings)
        self._normalize_embeddings()

    def _normalize_embeddings(self):
        """Pre-normalize document embeddings for faster search."""
        doc_norms = np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True)
        self.normalized_embeddings = np.divide(self.doc_embeddings, doc_norms, where=doc_norms != 0)

    def search(self, query_embedding: List[float]) -> np.ndarray:
        """Calculate cosine similarity scores for a query embedding."""
        query_embedding = np.array(query_embedding)
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding /= query_norm
        return np.dot(self.normalized_embeddings, query_embedding)

class BM25Searcher:
    """Performs sparse keyword search using BM25."""
    def __init__(self, documents: List[str]):
        self.documents = documents
        tokenized_docs = [self._tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        text = text.lower()
        return re.findall(r'\w+', text)

    def search(self, query: str) -> np.ndarray:
        """Calculate BM25 scores for a query."""
        query_tokens = self._tokenize(query)
        return np.array(self.bm25.get_scores(query_tokens))

class HybridSearch:
    """Combines dense vector search with sparse keyword search (BM25)."""
    def __init__(self, documents: List[str], embeddings: List[List[float]], alpha: float = 0.5):
        self.documents = documents
        self.alpha = alpha
        self.vector_searcher = VectorSearcher(documents, embeddings)
        self.bm25_searcher = BM25Searcher(documents)

    def search(self, query: str, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Perform hybrid search and return top results."""
        if not self.documents:
            return []

        vector_scores = self.vector_searcher.search(query_embedding)
        bm25_scores = self.bm25_searcher.search(query)

        combined_scores = self._combine_scores(vector_scores, bm25_scores)

        top_indices = np.argsort(-combined_scores)[:top_k]
        return [(self.documents[i], combined_scores[i]) for i in top_indices]

    def _combine_scores(self, vec_scores: np.ndarray, bm25_scores: np.ndarray) -> np.ndarray:
        """Normalize and combine scores with alpha weighting."""
        norm_vec = self._normalize(vec_scores)
        norm_bm25 = self._normalize(bm25_scores)
        return self.alpha * norm_vec + (1 - self.alpha) * norm_bm25

    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to a [0, 1] range."""
        min_s, max_s = np.min(scores), np.max(scores)
        return np.ones_like(scores) if max_s == min_s else (scores - min_s) / (max_s - min_s)