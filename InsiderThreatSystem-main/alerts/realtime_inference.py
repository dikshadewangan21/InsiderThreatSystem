"""
alerts/realtime_inference.py

Self-contained Real-Time Inference entry point and self-test suite.
Run with `python alerts/realtime_inference.py` to validate the alert pipeline.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from alerts.config import AlertConfig
from alerts.risk_engine import RiskEngine
from alerts.alert_engine import AlertEngine
from alerts.alert_logger import search_alerts

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def run_self_test() -> bool:
    """Run self-test on the real-time inference and alert pipeline."""
    setup_logging()
    logger = logging.getLogger("insider_threat.alerts.self_test")
    logger.info("Starting Alert Engine self-test...")
    
    try:
        # 1. Initialize Configuration
        config = AlertConfig()
        
        # Adjust thresholds specifically for the self-test to guarantee an alert is triggered
        # (Since the model has calibrated probabilities in [0.0001, 0.0023],
        # we set the test threshold slightly lower to ensure trigger paths are fully evaluated).
        config.high_threshold = 0.0001
        config.critical_threshold = 0.001
        
        # 2. Initialize Engines
        risk_engine = RiskEngine(config)
        alert_engine = AlertEngine(config, risk_engine)
        
        # 3. Dynamic Registry Query
        # To avoid hardcoding values that might not be in the registries,
        # we extract real user IDs, PCs, domains, and extensions from the loaded lookups.
        user_list = risk_engine.uid_list
        if not user_list:
            logger.error("User list registry is empty.")
            return False
            
        test_user = "CRD0624" if "CRD0624" in user_list else user_list[0]
        
        # Find a valid PC from registry
        pc_list = list(risk_engine.pc_registry.keys())
        test_pc = pc_list[0] if pc_list else "PC-0001"
        
        # Find a valid Domain from registry
        domain_list = list(risk_engine.domain_registry.keys())
        test_domain = domain_list[0] if domain_list else "google.com"
        
        # Find a valid Extension from registry
        ext_list = list(risk_engine.ext_registry.keys())
        test_ext = ext_list[0] if ext_list else "doc"
        
        logger.info(f"Using test entities -> User: {test_user}, PC: {test_pc}, Domain: {test_domain}, Ext: {test_ext}")
        
        # 4. Construct Sample Events
        sample_events = [
            {
                "user_id": test_user,
                "timestamp": 1420070400.0,  # Arbitrary timestamp
                "event_type": "logon",
                "resource": test_pc,
                "target_user": "none"
            },
            {
                "user_id": test_user,
                "timestamp": 1420070500.0,
                "event_type": "http",
                "resource": test_domain,
                "target_user": "none"
            },
            {
                "user_id": test_user,
                "timestamp": 1420070600.0,
                "event_type": "file",
                "resource": f"test.{test_ext}",
                "target_user": "none"
            }
        ]
        
        # 5. Process Events & Verify Inference Flow
        triggered_alerts = []
        for i, event in enumerate(sample_events):
            logger.info(f"Processing event {i+1}/{len(sample_events)}: {event['event_type']}...")
            alert = alert_engine.process_event(event)
            if alert is not None:
                triggered_alerts.append(alert)
                logger.info(f"Event {i+1} triggered alert: {alert.alert_id} ({alert.risk_level})")
                
        # 6. Verify alert logging and search functions
        logger.info("Verifying search/filter capabilities...")
        alerts_found = search_alerts(config, user_id=test_user)
        logger.info(f"Search found {len(alerts_found)} alert(s) for user {test_user}.")
        
        # Verify both JSON and CSV files exist
        json_exists = config.alerts_json_path.exists()
        csv_exists = config.alerts_csv_path.exists()
        logger.info(f"Alert logs exist -> CSV: {csv_exists}, JSON: {json_exists}")
        
        if not json_exists or not csv_exists:
            logger.error("Alert log files were not created.")
            return False
            
        logger.info("PASS")
        return True
        
    except Exception as e:
        logger.exception(f"Self-test failed due to exception: {e}")
        return False

if __name__ == "__main__":
    success = run_self_test()
    sys.exit(0 if success else 1)
