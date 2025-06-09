from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from pipeline.rag_pipeline import PipelineMetrics

@dataclass
class PipelineResult:
    """Result of a pipeline execution"""
    status: str
    response: Optional[str] = None
    contexts: Optional[List[str]] = None
    metrics: Optional[PipelineMetrics] = None
    error: Optional[str] = None
    config: Optional[Dict[str, Any]] = None

    @classmethod
    def success(
        cls,
        response: str,
        contexts: List[str],
        metrics: PipelineMetrics,
        config: Dict[str, Any]
    ) -> 'PipelineResult':
        """Create a successful result"""
        return cls(
            status="success",
            response=response,
            contexts=contexts,
            metrics=metrics,
            config=config
        )

    @classmethod
    def error(
        cls,
        error: str,
        config: Optional[Dict[str, Any]] = None
    ) -> 'PipelineResult':
        """Create an error result"""
        return cls(
            status="error",
            error=error,
            config=config
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        result_dict = {
            "status": self.status
        }
        if self.response is not None:
            result_dict["response"] = self.response
        if self.contexts is not None:
            result_dict["contexts"] = self.contexts
        if self.metrics is not None:
            result_dict["metrics"] = self.metrics.to_dict()
        if self.error is not None:
            result_dict["error"] = self.error
        if self.config is not None:
            result_dict["config"] = self.config
        return result_dict 