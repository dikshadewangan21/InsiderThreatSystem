"""
api/routes.py

FastAPI APIRouter definition mapping endpoints to services and schemas.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.config import APIConfig
from api.schemas import (
    EventInput,
    PredictRequest,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    AlertResponse
)
from api.dependencies import (
    get_config,
    get_risk_engine,
    get_alert_engine,
    get_explain_engine
)
from api.services import (
    predict_single_event,
    predict_batch_events,
    list_logged_alerts,
    get_alert_by_id,
    get_user_threat_history
)
from alerts.risk_engine import RiskEngine
from alerts.alert_engine import AlertEngine
from explainability.explainability_engine import ExplainabilityEngine

router = APIRouter()

@router.get("/", response_model=dict)
def read_root():
    """Return basic information about the Insider Threat Detection REST API."""
    return {
        "service": "Insider Threat Detection API",
        "version": "1.0.0",
        "documentation": "/docs"
    }

@router.get("/health", response_model=dict)
def health_check(
    risk_engine: RiskEngine = Depends(get_risk_engine)
):
    """Health check endpoint to verify ML components and lookups are ready."""
    # If the dependency resolves, the engines and their lookups are loaded.
    if risk_engine and risk_engine.tgn:
        return {
            "status": "healthy",
            "model_loaded": True,
            "lookups_loaded": len(risk_engine.uid_list) > 0
        }
    
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="ML models are not fully initialized."
    )

@router.post("/predict", response_model=PredictResponse)
def predict_threat(
    request: PredictRequest,
    risk_engine: RiskEngine = Depends(get_risk_engine),
    alert_engine: AlertEngine = Depends(get_alert_engine),
    explain_engine: ExplainabilityEngine = Depends(get_explain_engine)
):
    """Run real-time risk assessment and explainability diagnostics on a single CERT event."""
    return predict_single_event(
        request.event,
        risk_engine,
        alert_engine,
        explain_engine
    )

@router.post("/predict/batch", response_model=BatchPredictResponse)
def predict_threat_batch(
    request: BatchPredictRequest,
    risk_engine: RiskEngine = Depends(get_risk_engine),
    alert_engine: AlertEngine = Depends(get_alert_engine),
    explain_engine: ExplainabilityEngine = Depends(get_explain_engine)
):
    """Perform batch prediction and assessment on multiple CERT events sequentially."""
    return predict_batch_events(
        request.events,
        risk_engine,
        alert_engine,
        explain_engine
    )

@router.get("/alerts", response_model=List[AlertResponse])
def get_alerts(
    user_id: Optional[str] = Query(None, description="Filter alerts by User ID"),
    risk_level: Optional[str] = Query(None, description="Filter alerts by risk level (Critical, High, Medium, Low)"),
    start_time: Optional[str] = Query(None, description="Filter starting timestamp (ISO-8601 string)"),
    end_time: Optional[str] = Query(None, description="Filter ending timestamp (ISO-8601 string)"),
    config: APIConfig = Depends(get_config)
):
    """Retrieve logged alerts with optional query filters."""
    return list_logged_alerts(
        config,
        user_id=user_id,
        risk_level=risk_level,
        start_time=start_time,
        end_time=end_time
    )

@router.get("/alerts/{alert_id}", response_model=AlertResponse)
def get_alert(
    alert_id: str,
    config: APIConfig = Depends(get_config)
):
    """Fetch a single logged alert by its unique alert_id."""
    alert = get_alert_by_id(config, alert_id)
    if alert is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert with ID {alert_id} not found."
        )
    return alert

@router.get("/users/{user_id}/history", response_model=List[AlertResponse])
def get_user_history(
    user_id: str,
    config: APIConfig = Depends(get_config)
):
    """Retrieve the logged threat alert history for a specific user ID."""
    return get_user_threat_history(config, user_id)
