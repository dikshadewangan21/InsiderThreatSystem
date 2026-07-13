"""
alerts module

Production-ready real-time inference and alert engine for insider threat detection.
"""

from alerts.config import AlertConfig
from alerts.alert_models import Alert
from alerts.alert_logger import log_alert, load_alerts, search_alerts
from alerts.risk_engine import RiskEngine
from alerts.alert_engine import AlertEngine
