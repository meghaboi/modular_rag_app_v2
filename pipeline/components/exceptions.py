from typing import Optional, Dict, Any

class PipelineException(Exception):
    """Base exception class for pipeline-related errors"""
    def __init__(self, message: str, config: Optional[Dict[str, Any]] = None):
        self.message = message
        self.config = config
        super().__init__(self.message)

class PipelineInitializationError(PipelineException):
    """Raised when pipeline initialization fails"""
    pass

class PipelineExecutionError(PipelineException):
    """Raised when pipeline execution fails"""
    pass 

class RAGPipelineError(Exception):
    """Base exception class for RAG pipeline errors"""
    pass

class RAGPipelineInitializationError(RAGPipelineError):
    """Raised when pipeline initialization fails"""
    pass

class RAGPipelineExecutionError(RAGPipelineError):
    """Raised when pipeline execution fails"""
    pass

class RAGPipelineEvaluationError(RAGPipelineError):
    """Raised when pipeline evaluation fails"""
    pass 