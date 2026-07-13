"""
alerts/alert_engine.py

Alert generation and rule engine. Processes risk classifier outputs, 
assigns risk levels, and logs alerts for High and Critical risks.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
from alerts.config import AlertConfig
from alerts.alert_models import Alert
from alerts.risk_engine import RiskEngine, EVENT_TYPE_TO_RELATION
from alerts.alert_logger import log_alert

logger = logging.getLogger("insider_threat.alerts.alert_engine")

class AlertEngine:
    """Evaluates risk scores and triggers alerts for high-risk behaviors."""
    
    def __init__(self, config: AlertConfig, risk_engine: RiskEngine) -> None:
        self.config = config
        self.risk_engine = risk_engine
        
    def process_event(self, event: Dict[str, Any]) -> Optional[Alert]:
        """Assess the risk of an incoming event, classify risk level, and log alert if High/Critical."""
        user_id = str(event.get("user_id", "")).strip().upper()
        event_type = str(event.get("event_type", "")).strip().lower()
        resource = str(event.get("resource", "")).strip()
        ts = float(event.get("timestamp", 0.0))
        
        # Calculate neural risk score
        risk_score = self.risk_engine.predict_event(event)
        
        # Assign risk level based on thresholds
        if risk_score >= self.config.critical_threshold:
            risk_level = "Critical"
        elif risk_score >= self.config.high_threshold:
            risk_level = "High"
        elif risk_score >= self.config.medium_threshold:
            risk_level = "Medium"
        else:
            risk_level = "Low"
            
        logger.debug(f"Event evaluated: User={user_id}, Type={event_type}, Score={risk_score:.6f}, Level={risk_level}")
        
        # Generate alerts only for High and Critical risks
        if risk_level in ("High", "Critical"):
            reason, action = self._generate_reason_and_action(event, risk_score, risk_level)
            
            # Calibrate confidence (scale between 70% and 100% depending on risk severity)
            base_conf = 0.70 + 0.29 * (risk_score - self.config.high_threshold) / (1.0 - self.config.high_threshold)
            confidence = min(max(base_conf, 0.70), 0.99)
            
            alert = Alert.create(
                user_id=user_id,
                risk_score=risk_score,
                risk_level=risk_level,
                reason=reason,
                suggested_action=action,
                confidence=confidence
            )
            
            # Persist alert to CSV and JSON logs
            log_alert(alert, self.config)
            logger.warning(f"SECURITY ALERT TRIGGERED [{risk_level}]: User={user_id}, Score={risk_score:.6f}, Reason={reason}")
            return alert
            
        return None
        
    def _generate_reason_and_action(
        self, event: Dict[str, Any], risk_score: float, risk_level: str
    ) -> Tuple[str, str]:
        """Dynamically generate domain-specific reasoning and remediation steps based on event features."""
        user_id = str(event.get("user_id", "")).strip().upper()
        event_type = str(event.get("event_type", "")).strip().lower()
        resource = str(event.get("resource", "")).strip()
        ts = float(event.get("timestamp", 0.0))
        
        reasons = []
        actions = []
        
        # Check hour
        dti = datetime.utcfromtimestamp(ts)
        is_after_hours = dti.hour < 7 or dti.hour >= 19
        
        # Check lookup features
        user_idx = self.risk_engine.uid_to_index.get(user_id)
        if user_idx is not None:
            psych_score = float(self.risk_engine.lookups.user_psychology[user_idx])
            behav_dev = float(self.risk_engine.lookups.user_behav_dev[user_idx])
            
            if psych_score > 0.6:
                reasons.append("elevated psychological profile indicators (e.g. stress, disgruntlement)")
            if behav_dev > 0.6:
                reasons.append("substantial behavioral deviation from historical baseline")
                
        if is_after_hours:
            reasons.append(f"activity occurred during after-hours ({dti.hour:02d}:{dti.minute:02d} UTC)")
            actions.append("review logon pattern alignment with official department work hour exceptions")
            
        relation = EVENT_TYPE_TO_RELATION.get(event_type)
        if relation:
            _, dst_type = relation
            if dst_type == "FileExtension":
                ext = resource.lower().lstrip(".")
                ext_idx = self.risk_engine.ext_registry.get(ext)
                if ext_idx is not None:
                    sens = float(self.risk_engine.lookups.ext_sensitivity[ext_idx])
                    if sens > 0.7:
                        reasons.append(f"access of high-sensitivity file asset type (.{ext})")
                        actions.append(f"verify if file type (.{ext}) contents were encrypted or copied to external storage")
            elif dst_type == "WebsiteDomain":
                reasons.append(f"unusual external web traffic to domain {resource}")
                actions.append(f"verify website domain safety category and volume of data uploaded")
            elif dst_type == "PC":
                reasons.append(f"logon/logoff or device connection event on host PC {resource}")
                actions.append("verify host PC hardware connection log for unauthorized USB/removable media inserts")
                
        # Fallback reason
        if not reasons:
            reasons.append("anomalous behavior matching temporal pattern characteristics of known threat actors")
            
        # Suggested action matching risk severity
        if risk_level == "Critical":
            actions.insert(0, "IMMEDIATE ACTION REQUIRED: Temporarily suspend user account and revoke credentials")
            actions.append("initiate immediate network/PC isolation and forensic disk imaging")
        else:
            actions.insert(0, "RECOMMENDED ACTION: Flag user account for close monitoring and alert supervisor")
            actions.append("schedule interview with employee's direct manager to rule out legitimate work explanation")
            
        reason_str = "Event triggered alert due to: " + ", ".join(reasons) + f" (calibrated probability: {risk_score:.6f})."
        action_str = "; ".join(actions) + "."
        
        return reason_str, action_str
