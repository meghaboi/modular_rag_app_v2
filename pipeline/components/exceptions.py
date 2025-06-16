from typing import Optional, Dict, Any

class RAGPipelineException(Exception):
    """Base exception for all RAG pipeline-related errors."""
    def __init__(self, message: str, config: Optional[Dict[str, Any]] = None):
        self.message = message
        self.config = config
        super().__init__(self.message)

class RAGPipelineInitializationError(RAGPipelineException):
    """Raised when RAG pipeline initialization fails."""
    pass

class RAGPipelineExecutionError(RAGPipelineException):
    """Raised when RAG pipeline execution fails."""
    pass

class RAGPipelineEvaluationError(RAGPipelineException):
    """Raised when RAG pipeline evaluation fails."""
    pass