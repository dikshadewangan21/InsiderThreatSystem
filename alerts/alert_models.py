"""
alerts/alert_models.py

Data models for alerts in the Real-Time Inference and Alert Engine.
"""

import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict

@dataclass
class Alert:
    alert_id: str
    timestamp: str  # ISO-8601 string
    user_id: str
    risk_score: float
    confidence: float
    risk_level: str
    reason: str
    suggested_action: str

    @classmethod
    def create(
        cls,
        user_id: str,
        risk_score: float,
        risk_level: str,
        reason: str,
        suggested_action: str,
        confidence: float = None,
    ) -> "Alert":
        """Factory method to create a new Alert with a unique ID and current timestamp."""
        # Standardize confidence based on risk score if not provided
        if confidence is None:
            # Scale confidence relative to risk score ranges
            confidence = min(risk_score * 100.0, 1.0)
            
        return cls(
            alert_id=f"ALERT-{uuid.uuid4().hex[:8].upper()}",
            timestamp=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            user_id=user_id,
            risk_score=float(risk_score),
            confidence=float(confidence),
            risk_level=risk_level,
            reason=reason,
            suggested_action=suggested_action,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Alert":
        return cls(
            alert_id=data["alert_id"],
            timestamp=data["timestamp"],
            user_id=data["user_id"],
            risk_score=float(data["risk_score"]),
            confidence=float(data["confidence"]),
            risk_level=data["risk_level"],
            reason=data["reason"],
            suggested_action=data["suggested_action"],
        )
