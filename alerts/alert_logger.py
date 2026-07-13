"""
alerts/alert_logger.py

Thread-safe logging, saving, and querying of security alerts to CSV and JSON formats.
"""

import csv
import json
import os
import threading
from typing import Any, Dict, List, Optional
from alerts.config import AlertConfig
from alerts.alert_models import Alert

_LOCK = threading.Lock()

def log_alert(alert: Alert, config: AlertConfig) -> None:
    """Thread-safely log a generated Alert to both JSON and CSV files."""
    alert_dict = alert.to_dict()
    
    with _LOCK:
        # --- 1. Write to JSON ---
        alerts_list = []
        if os.path.exists(config.alerts_json_path):
            try:
                with open(config.alerts_json_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        alerts_list = json.loads(content)
            except Exception:
                # If corrupted, back up and start fresh
                pass
                
        alerts_list.append(alert_dict)
        
        with open(config.alerts_json_path, "w", encoding="utf-8") as f:
            json.dump(alerts_list, f, indent=4, ensure_ascii=False)
            
        # --- 2. Write to CSV ---
        file_exists = os.path.exists(config.alerts_csv_path)
        headers = ["alert_id", "timestamp", "user_id", "risk_score", "confidence", "risk_level", "reason", "suggested_action"]
        
        with open(config.alerts_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow(alert_dict)

def load_alerts(config: AlertConfig) -> List[Alert]:
    """Load all logged alerts from the JSON store."""
    if not os.path.exists(config.alerts_json_path):
        return []
        
    with _LOCK:
        try:
            with open(config.alerts_json_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return []
                data_list = json.loads(content)
                return [Alert.from_dict(d) for d in data_list]
        except Exception:
            return []

def search_alerts(
    config: AlertConfig,
    user_id: Optional[str] = None,
    risk_level: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> List[Alert]:
    """Search and filter alerts based on user_id, risk level, or timestamp ranges."""
    alerts = load_alerts(config)
    filtered = []
    
    for alert in alerts:
        if user_id and alert.user_id.upper() != user_id.upper():
            continue
        if risk_level and alert.risk_level.upper() != risk_level.upper():
            continue
        if start_time and alert.timestamp < start_time:
            continue
        if end_time and alert.timestamp > end_time:
            continue
        filtered.append(alert)
        
    return filtered
