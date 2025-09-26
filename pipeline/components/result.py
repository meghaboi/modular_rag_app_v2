from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict

from pipeline.models.metrics import PipelineMetrics

def _serialize_item(item: Any) -> Any:
    """Helper to serialize items, converting enums to values."""
    if hasattr(item, 'value'):
        return item.value
    if isinstance(item, dict):
        return {k: _serialize_item(v) for k, v in item.items()}
    if isinstance(item, list):
        return [_serialize_item(i) for i in item]
    return item

@dataclass
class PipelineResult:
    """Result of a pipeline execution."""
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
        """Create a successful result."""
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
        """Create an error result."""
        return cls(
            status="error",
            error=error,
            config=config
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the PipelineResult instance to a serializable dictionary.
        
        This method handles nested dataclasses and serializes any enum members 
        to their string values, ensuring the output is clean and ready for
        formats like JSON.
        """
        # asdict provides a deep conversion of the dataclass to a dict
        result_dict = asdict(self)
        
        # Clean the dictionary by removing keys with None values
        cleaned_dict = {k: v for k, v in result_dict.items() if v is not None}

        # Recursively serialize items to handle enums inside config or elsewhere
        return _serialize_item(cleaned_dict) 