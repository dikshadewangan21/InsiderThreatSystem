"""
api/config.py

Configuration parameters for the FastAPI REST API.
"""

from dataclasses import dataclass, field
from alerts.config import AlertConfig
from explainability.config import ExplainabilityConfig

import os

@dataclass
class APIConfig:
    host: str = os.getenv("API_HOST", "127.0.0.1")
    port: int = int(os.getenv("API_PORT", 8080))
    alert_config: AlertConfig = field(default_factory=AlertConfig)
    explain_config: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
