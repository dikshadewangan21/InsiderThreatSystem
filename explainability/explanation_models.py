"""
explainability/explanation_models.py

Data models representing explanations generated for security alerts.
"""

import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

@dataclass
class Explanation:
    risk_score: float
    risk_level: str
    confidence: float
    top_factors: List[Dict[str, Any]]  # List containing keys: factor, score, description
    feature_values: Dict[str, float]  # Mapped raw feature values
    analyst_summary: str
    recommended_action: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert the explanation object to a serializable dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 4) -> str:
        """Convert the explanation object to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
