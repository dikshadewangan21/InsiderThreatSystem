#!/usr/bin/env python3
"""
evaluate.py

Production-grade model evaluation script for the Insider Threat Detection pipeline.
Loads the trained model checkpoints and evaluates performance on the test split.
"""

import argparse
import sys
import os
import json
import logging
from typing import Sequence, Optional
from pathlib import Path

# Ensure project root is in the path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from training.config import TrainingConfig
from train import InsiderThreatTrainer, _configure_logging, _require_production_modules

LOGGER = logging.getLogger("insider_threat.evaluation")

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the trained Insider Threat Detection model on the test split."
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a JSON config previously written by TrainingConfig.save().",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Override the model checkpoint path to load.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Override the evaluation device ('cpu', 'cuda', etc.).",
    )
    parser.add_argument(
        "--output-report", type=str, default="reports/evaluation_report.json",
        help="Path to save the evaluation JSON report.",
    )
    return parser.parse_args(argv)

def main(argv: Optional[Sequence[str]] = None) -> None:
    _configure_logging()
    args = parse_args(argv)

    config = TrainingConfig.load(args.config) if args.config else TrainingConfig()
    
    # Force epochs to 0 to prevent any training execution
    config.epochs = 0
    
    if args.device is not None:
        config.device = args.device
        LOGGER.info("Config override | device = %s", args.device)
        
    if args.checkpoint is not None:
        config.best_model_path = Path(args.checkpoint)
        LOGGER.info("Config override | best_model_path = %s", args.checkpoint)

    _require_production_modules()
    
    LOGGER.info("Initializing evaluator and loading datasets...")
    trainer = InsiderThreatTrainer(config)
    
    LOGGER.info("Evaluating saved model checkpoint on test split...")
    test_metrics = trainer.evaluate_test_set()
    
    # Save the JSON report
    output_path = Path(config.project_root) / args.output_report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=4)
        
    LOGGER.info("=========================================")
    LOGGER.info("EVALUATION COMPLETED SUCCESSFULLY")
    LOGGER.info("Report saved to: %s", output_path)
    LOGGER.info("Node-Level classification metrics: %s", test_metrics)
    LOGGER.info("=========================================")

if __name__ == "__main__":
    main()
