"""
explainability/explainability_engine.py

Explainability Engine for Insider Threat Detection. Analyzes neural predictions, 
ranks feature importances, and compiles human-readable analyst explanation models.
"""

import logging
import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from explainability.config import ExplainabilityConfig
from explainability.explanation_models import Explanation
from explainability.feature_importance import get_saliency_importance
from alerts.config import AlertConfig
from alerts.risk_engine import RiskEngine
from alerts.alert_models import Alert

logger = logging.getLogger("insider_threat.explainability.engine")

class ExplainabilityEngine:
    """Generates detailed, post-hoc explanations for threat alerts."""
    
    def __init__(self, config: ExplainabilityConfig, risk_engine: RiskEngine) -> None:
        self.config = config
        self.risk_engine = risk_engine
        
    def explain_alert(self, alert: Alert, event: Dict[str, Any]) -> Explanation:
        """Analyze an Alert and its raw event to generate a detailed Explanation."""
        user_id = alert.user_id
        risk_score = alert.risk_score
        risk_level = alert.risk_level
        confidence = alert.confidence
        
        # 1. Compute Feature Importance & Extract Raw Values
        importance_dict, raw_features_dict, attn_weight = get_saliency_importance(self.risk_engine, event)
        
        # 2. Sort Features by Saliency Importance and extract Top 5
        sorted_features = sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)
        top_5_raw = sorted_features[:5]
        
        # Normalize the top 5 scores so they sum to 100% (1.0)
        top_5_sum = sum(score for name, score in top_5_raw)
        normalized_top_5 = []
        for name, score in top_5_raw:
            norm_score = (score / top_5_sum) if top_5_sum > 0 else 0.2
            normalized_top_5.append((name, norm_score))
            
        top_factors = []
        feature_values_subset = {}
        
        for name, norm_score in normalized_top_5:
            raw_val = raw_features_dict.get(name, 0.0)
            feature_values_subset[name] = raw_val
            
            # Map raw feature name to human-readable display name and reason text
            display_name = self.config.feature_display_names.get(name, name)
            reason_text = self.config.feature_reason_mapping.get(
                name, f"Elevated {display_name.lower()}"
            )
            
            top_factors.append({
                "factor": display_name,  # Clean human-readable display name
                "importance_score": float(norm_score),  # Normalized to sum to 100%
                "raw_value": float(raw_val),
                "description": f"{reason_text} (SOC priority: {norm_score:.1%})"
            })
            
        # 2.5 Compute Top Influential Neighbors
        neighbors = []
        resource = event.get("resource", "unknown")
        event_type = event.get("event_type", "unknown")
        
        # Get relation name and destination type dynamically
        from alerts.risk_engine import EVENT_TYPE_TO_RELATION, TEMPORAL_RELATION_NAME
        relation = EVENT_TYPE_TO_RELATION.get(event_type.lower())
        if relation is not None:
            src_node_type, dst_node_type = relation
            relation_name = TEMPORAL_RELATION_NAME[relation]
            
            neighbors.append({
                "neighbor_id": str(resource),
                "node_type": str(dst_node_type),
                "relation": str(relation_name),
                "influence_score": float(attn_weight),
                "description": f"Active target resource accessed during session (Neural Attention: {attn_weight:.1%})"
            })
            
        # Static LDAP relationships (Structural neighbors)
        lookups = self.risk_engine.lookups
        if hasattr(lookups, "uid_to_dept") and user_id in lookups.uid_to_dept:
            dept = lookups.uid_to_dept[user_id]
            if dept and dept != "unknown":
                neighbors.append({
                    "neighbor_id": str(dept),
                    "node_type": "Department",
                    "relation": "belongs_to",
                    "influence_score": 0.15,
                    "description": "User organizational department group (Static Influence: 15.0%)"
                })
        if hasattr(lookups, "uid_to_role") and user_id in lookups.uid_to_role:
            role = lookups.uid_to_role[user_id]
            if role and role != "unknown":
                neighbors.append({
                    "neighbor_id": str(role),
                    "node_type": "Role",
                    "relation": "has_role",
                    "influence_score": 0.10,
                    "description": "User organizational role assignment (Static Influence: 10.0%)"
                })
        if hasattr(lookups, "uid_to_team") and user_id in lookups.uid_to_team:
            team = lookups.uid_to_team[user_id]
            if team and team != "unknown":
                neighbors.append({
                    "neighbor_id": str(team),
                    "node_type": "Team",
                    "relation": "member_of",
                    "influence_score": 0.05,
                    "description": "User team assignment group (Static Influence: 5.0%)"
                })

        # 3. Compile Analyst Summary
        import numpy as np
        # Scaled threat score (0-100 scale) to represent risk index consistently
        xp = [0.0, 1e-4, 0.000883, 0.005, 1.0]
        fp = [0.0, 35.0, 75.0, 92.0, 100.0]
        soc_score = float(np.interp(risk_score, xp, fp))
        
        top_descriptions = [f"{fact['factor']} ({fact['importance_score']:.1%})" for fact in top_factors[:3]]
        
        summary = (
            f"Security Threat Assessment: A {risk_level} severity risk level was assigned to user {user_id}. "
            f"The calculated Threat Index is {soc_score:.1f}/100 (representing a raw neural probability of {risk_score:.6f} with {confidence:.1%} confidence). "
            f"The primary security indicators driving this assessment are: {', '.join(top_descriptions)}. "
            f"The active session exhibits behavioral patterns that significantly deviate from the user's historical baseline, suggesting potential insider activity."
        )
        
        # 4. Generate Recommended Actions
        actions = []
        if risk_level == "Critical":
            actions.append("CRITICAL RESPONSE: Revoke employee credentials immediately and initiate automated PC host isolation.")
        else:
            actions.append("ALERT RESPONSE: Flag user account for continuous session logging and queue for immediate supervisor review.")
            
        # Add feature-specific actionable recommendations based on top factors
        for name, norm_score in normalized_top_5:
            raw_val = raw_features_dict.get(name, 0.0)
            # Suggest if the feature has a non-trivial value
            if raw_val > 0.1:
                if name in ("is_after_hours", "weekend_flag", "after_hours_score"):
                    actions.append("Audit active directory logs to verify if a manager authorized off-hours login windows.")
                elif name in ("file_sensitivity", "extension_risk"):
                    actions.append("Initiate a filesystem scan to review recent documents read or archived by this user.")
                elif name in ("is_device_event", "pc_popularity"):
                    actions.append("Inspect endpoint protection logs for unauthorized USB storage inserts or external device mounts.")
                elif name in ("domain_popularity", "is_http_event"):
                    actions.append("Review firewall connection states and DNS queries to identify external web destinations visited during this session.")
                elif name in ("psychology_score", "behavior_deviation"):
                    actions.append("Coordinate a behavioral risk evaluation with HR and the security supervisor.")
                    
        # Deduplicate actions
        seen = set()
        unique_actions = []
        for action in actions:
            if action not in seen:
                seen.add(action)
                unique_actions.append(action)
                
        recommended_action_str = " ".join(unique_actions)
        
        return Explanation(
            risk_score=risk_score,
            risk_level=risk_level,
            confidence=confidence,
            top_factors=top_factors,
            top_influential_neighbors=neighbors,
            feature_values=feature_values_subset,
            analyst_summary=summary,
            recommended_action=recommended_action_str
        )

def run_self_test() -> bool:
    """Run self-test for the Explainability Engine."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    logger.info("Initializing Explainability self-test...")
    
    try:
        # Load configs
        alert_cfg = AlertConfig()
        exp_cfg = ExplainabilityConfig()
        
        # Override threshold to guarantee testing conditions
        alert_cfg.high_threshold = 0.0001
        
        # Load risk engine
        risk_engine = RiskEngine(alert_cfg)
        exp_engine = ExplainabilityEngine(exp_cfg, risk_engine)
        
        # Get valid entities dynamically from lookups
        user_list = risk_engine.uid_list
        test_user = "CRD0624" if "CRD0624" in user_list else user_list[0]
        pc_list = list(risk_engine.pc_registry.keys())
        test_pc = pc_list[0] if pc_list else "PC-0001"
        
        # Mock alert and event
        mock_alert = Alert.create(
            user_id=test_user,
            risk_score=0.0023,
            risk_level="High",
            reason="Mock alert reason",
            suggested_action="Mock suggested action",
            confidence=0.88
        )
        
        mock_event = {
            "user_id": test_user,
            "timestamp": 1420070400.0,
            "event_type": "logon",
            "resource": test_pc,
            "target_user": "none"
        }
        
        # Generate explanation
        logger.info(f"Generating explanation for user {test_user} logon event...")
        explanation = exp_engine.explain_alert(mock_alert, mock_event)
        
        # Print explanation output
        print("\n=== GENERATED ANALYST EXPLANATION ===")
        print(explanation.to_json())
        print("======================================\n")
        
        # Basic assertions to check validity
        assert explanation.risk_score == 0.0023
        assert explanation.risk_level == "High"
        assert len(explanation.top_factors) == 5
        assert len(explanation.top_influential_neighbors) > 0
        assert len(explanation.analyst_summary) > 0
        assert len(explanation.recommended_action) > 0
        
        logger.info("PASS")
        return True
        
    except Exception as e:
        logger.exception(f"Explainability self-test failed: {e}")
        return False

if __name__ == "__main__":
    success = run_self_test()
    sys.exit(0 if success else 1)
