"""
explainability/config.py

Configuration parameters and feature-to-reason mappings for the Explainability Engine.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

@dataclass
class ExplainabilityConfig:
    project_root: Path = Path(__file__).resolve().parent.parent
    checkpoint_dir: Path = project_root / "checkpoints"
    artifacts_path: Path = project_root / "graph/output/preprocessing_artifacts.pkl"
    psychology_csv: Path = project_root / "data/processed/psychology_features.csv"
    fused_csv: Path = project_root / "data/processed/fused_features.csv"
    
    # Feature category thresholds above which a feature is flagged as anomalous
    anomaly_threshold: float = 0.5
    
    # Map of raw feature column names to human-readable reasons
    feature_reason_mapping: Dict[str, str] = field(default_factory=lambda: {
        "is_after_hours": "After-hours logon/activity",
        "weekend_flag": "Weekend logon/activity",
        "login_frequency": "Abnormal login frequency",
        "activity_frequency": "Abnormal overall activity frequency",
        "file_sensitivity": "Sensitive file access spike",
        "extension_risk": "High-risk file extension interaction",
        "is_device_event": "USB device activity",
        "domain_popularity": "Unusual/unpopular web domain access",
        "is_http_event": "Unusual external web traffic activity",
        "psychology_score": "Psychological stress/anomalous indicators increase",
        "behavior_deviation": "Behavioral deviation from baseline",
        "user_behavior_score": "Significant deviation in user behavior history",
        "after_hours_score": "Historical preference for after-hours activity",
        "time_since_last_event": "Abnormal time gap since last event (rapid bursts)",
        "session_duration": "Abnormally long session duration",
        "has_target_user": "Interaction targeting another user's resources"
    })

    # Human-readable title-case display names for all 40 features
    feature_display_names: Dict[str, str] = field(default_factory=lambda: {
        "hour_of_day": "Hour of day",
        "minute_of_hour": "Minute of hour",
        "weekday": "Day of week",
        "weekend_flag": "Weekend logon",
        "month": "Month of year",
        "quarter": "Quarter of year",
        "is_after_hours": "After-hours activity",
        "is_working_hours": "Working-hours activity",
        "time_since_last_event": "Time gap since last activity",
        "session_duration": "User session duration",
        "time_gap_prev_action": "Category activity gap",
        "user_behavior_score": "User historical deviation rating",
        "anomaly_score": "User baseline anomaly score",
        "activity_frequency": "User event frequency rate",
        "rolling_action_count": "Rolling action count",
        "login_frequency": "User login frequency rate",
        "device_frequency": "User device connection rate",
        "website_frequency": "User web browsing rate",
        "psychology_score": "Psychological stress score",
        "behavior_deviation": "User behavior deviation from baseline",
        "after_hours_score": "Historical after-hours activity score",
        "dept_encoded": "Department profile encoding",
        "role_encoded": "Job role profile encoding",
        "team_encoded": "Team assignment profile",
        "bu_encoded": "Business unit profile",
        "has_manager": "Supervisor assignment status",
        "file_sensitivity": "File sensitivity score",
        "domain_popularity": "Web domain popularity rating",
        "pc_popularity": "PC workstation popularity rating",
        "extension_risk": "File extension threat rating",
        "source_degree": "User interaction degree",
        "destination_degree": "Asset connection density",
        "historical_interaction_count": "Historical connection frequency",
        "is_logon_event": "Logon event indicator",
        "is_device_event": "USB/device connection event",
        "is_http_event": "External HTTP request event",
        "rel_accesses_pc": "PC workstation access",
        "rel_visits_web": "External domain visit",
        "rel_touches_file": "File interaction",
        "has_target_user": "Target user indicator"
    })

