"""
api/dependencies.py

Dependency injection functions for loading and retrieving singleton ML engines.
"""

from typing import Optional
from fastapi import HTTPException, status
from alerts.config import AlertConfig
from alerts.risk_engine import RiskEngine
from alerts.alert_engine import AlertEngine
from explainability.config import ExplainabilityConfig
from explainability.explainability_engine import ExplainabilityEngine
from api.config import APIConfig

_config: Optional[APIConfig] = None
_risk_engine: Optional[RiskEngine] = None
_alert_engine: Optional[AlertEngine] = None
_explain_engine: Optional[ExplainabilityEngine] = None

def init_engines(config: APIConfig) -> None:
    """Initialize the backend ML engines once during API startup."""
    global _config, _risk_engine, _alert_engine, _explain_engine
    _config = config
    
    # Initialize RiskEngine
    _risk_engine = RiskEngine(config.alert_config)
    
    # Initialize AlertEngine
    _alert_engine = AlertEngine(config.alert_config, _risk_engine)
    
    # Initialize ExplainabilityEngine
    _explain_engine = ExplainabilityEngine(config.explain_config, _risk_engine)

def get_config() -> APIConfig:
    if _config is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API configuration is not initialized."
        )
    return _config

def get_risk_engine() -> RiskEngine:
    if _risk_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Risk Engine is not initialized."
        )
    return _risk_engine

def get_alert_engine() -> AlertEngine:
    if _alert_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Alert Engine is not initialized."
        )
    return _alert_engine

def get_explain_engine() -> ExplainabilityEngine:
    if _explain_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Explainability Engine is not initialized."
        )
    return _explain_engine
