"""
api/services.py

Service layer for coordinating ML inference, alert evaluation, and history queries.
"""

from typing import Any, Dict, List, Optional
from alerts.alert_models import Alert
from alerts.alert_logger import load_alerts, search_alerts
from alerts.risk_engine import RiskEngine
from alerts.alert_engine import AlertEngine
from explainability.explainability_engine import ExplainabilityEngine
from api.config import APIConfig
from api.schemas import (
    EventInput,
    PredictResponse,
    BatchPredictResponse,
    AlertResponse,
    AlertExplanationResponse
)

def predict_single_event(
    event: EventInput,
    risk_engine: RiskEngine,
    alert_engine: AlertEngine,
    explain_engine: ExplainabilityEngine
) -> PredictResponse:
    """Assess a single CERT event, evaluate alerts, and run explainability diagnostics."""
    event_dict = event.model_dump()
    
    # 1. Run inference directly to get risk score & level
    risk_score = risk_engine.predict_event(event_dict)
    
    config = alert_engine.config
    if risk_score >= config.critical_threshold:
        risk_level = "Critical"
    elif risk_score >= config.high_threshold:
        risk_level = "High"
    elif risk_score >= config.medium_threshold:
        risk_level = "Medium"
    else:
        risk_level = "Low"
        
    # Scale confidence relative to decision bounds
    base_conf = 0.70 + 0.29 * (risk_score - config.high_threshold) / (1.0 - config.high_threshold)
    confidence = min(max(base_conf, 0.70), 0.99)
    
    # 2. Run through Alert Engine to log alert if High/Critical
    alert = alert_engine.process_event(event_dict)
    alert_generated = alert is not None
    
    # 3. Create transitory alert if none generated, for explainability output
    if alert is None:
        alert = Alert.create(
            user_id=event.user_id,
            risk_score=risk_score,
            risk_level=risk_level,
            reason="Transitory risk assessment",
            suggested_action="none",
            confidence=confidence
        )
        
    # 4. Generate Explanation
    explanation = explain_engine.explain_alert(alert, event_dict)
    
    return PredictResponse(
        risk_score=risk_score,
        risk_level=risk_level,
        confidence=confidence,
        explanation=AlertExplanationResponse(**explanation.to_dict()),
        alert_generated=alert_generated
    )

def predict_batch_events(
    events: List[EventInput],
    risk_engine: RiskEngine,
    alert_engine: AlertEngine,
    explain_engine: ExplainabilityEngine
) -> BatchPredictResponse:
    """Run batch prediction on multiple CERT events sequentially."""
    predictions = []
    for event in events:
        pred = predict_single_event(event, risk_engine, alert_engine, explain_engine)
        predictions.append(pred)
    return BatchPredictResponse(predictions=predictions)

def list_logged_alerts(
    config: APIConfig,
    user_id: Optional[str] = None,
    risk_level: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
) -> List[AlertResponse]:
    """Retrieve logged alerts matching given filters."""
    alerts = search_alerts(
        config.alert_config,
        user_id=user_id,
        risk_level=risk_level,
        start_time=start_time,
        end_time=end_time
    )
    return [AlertResponse(**a.to_dict()) for a in alerts]

def get_alert_by_id(config: APIConfig, alert_id: str) -> Optional[AlertResponse]:
    """Fetch a single logged alert by its unique alert_id."""
    alerts = load_alerts(config.alert_config)
    for alert in alerts:
        if alert.alert_id.upper() == alert_id.upper():
            return AlertResponse(**alert.to_dict())
    return None

def get_user_threat_history(config: APIConfig, user_id: str) -> List[AlertResponse]:
    """Load the complete alert history for a specific user ID."""
    return list_logged_alerts(config, user_id=user_id)
