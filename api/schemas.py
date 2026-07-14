"""
api/schemas.py

Pydantic model definitions for REST API request and response bodies.
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

class EventInput(BaseModel):
    user_id: str = Field(..., description="ID of the user generating the event, e.g. CRD0624", examples=["CRD0624"])
    timestamp: float = Field(..., description="Unix epoch timestamp in seconds", examples=[1420070400.0])
    event_type: str = Field(..., description="Type of CERT activity log, e.g. logon, http, file, device", examples=["logon"])
    resource: str = Field(..., description="Resource accessed (PC ID, domain name, or file path/extension)", examples=["PC-6056"])
    target_user: str = Field("none", description="Optional target user ID involved in the transaction", examples=["none"])

class PredictRequest(BaseModel):
    event: EventInput

class AlertFactorResponse(BaseModel):
    factor: str
    importance_score: float
    raw_value: float
    description: str

class AlertNeighborResponse(BaseModel):
    neighbor_id: str
    node_type: str
    relation: str
    influence_score: float
    description: str

class AlertExplanationResponse(BaseModel):
    risk_score: float
    risk_level: str
    confidence: float
    top_factors: List[AlertFactorResponse]
    top_influential_neighbors: List[AlertNeighborResponse]
    feature_values: Dict[str, float]
    analyst_summary: str
    recommended_action: str

class PredictResponse(BaseModel):
    risk_score: float
    risk_level: str
    confidence: float
    explanation: AlertExplanationResponse
    alert_generated: bool

class BatchPredictRequest(BaseModel):
    events: List[EventInput]

class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]

class AlertResponse(BaseModel):
    alert_id: str
    timestamp: str
    user_id: str
    risk_score: float
    confidence: float
    risk_level: str
    reason: str
    suggested_action: str
