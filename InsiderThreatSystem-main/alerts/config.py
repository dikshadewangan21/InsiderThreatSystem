"""
alerts/config.py

Configuration parameters for the Real-Time Inference and Alert Engine.
"""

import os
from dataclasses import dataclass
from pathlib import Path

@dataclass
class AlertConfig:
    # --- Paths ---
    project_root: Path = Path(__file__).resolve().parent.parent
    checkpoint_dir: Path = project_root / "checkpoints"
    best_model_path: Path = checkpoint_dir / "best_model.pt"
    last_model_path: Path = checkpoint_dir / "last_model.pt"
    
    graph_path: Path = project_root / "graph/output/node_graph_skeleton.pt"
    artifacts_path: Path = project_root / "graph/output/preprocessing_artifacts.pkl"
    psychology_csv: Path = project_root / "data/processed/psychology_features.csv"
    fused_csv: Path = project_root / "data/processed/fused_features.csv"
    
    alerts_csv_path: Path = project_root / "alerts/alerts.csv"
    alerts_json_path: Path = project_root / "alerts/alerts.json"
    
    # --- Inference settings ---
    device: str = "cpu"  # cpu is preferred for single-event real-time inference
    
    # --- Calibration & Alert Thresholds ---
    # Low / Medium / High / Critical levels
    # Risk probabilities mapping:
    # - Low: < 0.0001
    # - Medium: >= 0.0001
    # - High: >= 0.000883 (the validated decision threshold)
    # - Critical: >= 0.005 (indicates highly calibrated anomalous score)
    low_threshold: float = 1e-6
    medium_threshold: float = 1e-4
    high_threshold: float = 0.000883
    critical_threshold: float = 0.005
    
    # Calibrated logit shift parameter (from pos_weight)
    pos_weight: float = 20.0
    calibrate_weighted_logits: bool = True
    
    def __post_init__(self) -> None:
        # Create checkpoints and alert dirs if missing
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.alerts_csv_path.parent.mkdir(parents=True, exist_ok=True)
