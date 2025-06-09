from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class PipelineMetrics:
    """Metrics for pipeline execution"""
    total_time: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    llm_cost: float
    evaluation_scores: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        metrics_dict = {
            "total_time": self.total_time,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "llm_cost": self.llm_cost
        }
        if self.evaluation_scores:
            metrics_dict.update(self.evaluation_scores)
        return metrics_dict 