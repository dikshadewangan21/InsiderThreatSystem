"""
training/config.py

Central configuration object for the Insider Threat Detection training
pipeline. All tunable values live here so that ``training/train.py`` never
contains hardcoded paths or magic numbers.

The configuration can be constructed from defaults, from an argparse
Namespace, or from a JSON/YAML file on disk via ``TrainingConfig.load``.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass


@dataclass
class TrainingConfig:
    """Configuration for the TGN -> GAT -> MLP training pipeline.

    Attributes:
        graph_path: Path to the serialized production graph produced by
            ``graph/build_event_graph.py``.
        edge_shard_dir: Directory containing streamed edge-feature shards
            named ``edge_features_XXXXXX.pt`` (produced by
            ``graph/edge_features.py``).
        labels_dir: Directory expected to contain exactly one of
            ``labels.pt``, ``labels.csv`` or ``labels.pkl`` with node-level
            risk labels.
        checkpoint_dir: Directory where ``best_model.pt``, ``last_model.pt``,
            ``optimizer.pt``, ``scheduler.pt`` and ``training_state.pt`` are
            written.
        log_dir: Directory for TensorBoard event files.
        batch_size: Number of nodes/events per forward pass within a shard.
        epochs: Maximum number of training epochs.
        learning_rate: Initial learning rate for the optimizer.
        weight_decay: L2 weight decay coefficient.
        optimizer: One of {"adam", "adamw", "sgd"}.
        scheduler: One of {"cosine", "step", "plateau"}.
        loss: One of {"bce", "bce_logits", "focal"}.
        focal_alpha: Alpha term used only when ``loss == "focal"``.
        focal_gamma: Gamma term used only when ``loss == "focal"``.
        max_pos_weight: Upper bound for positive-class weighting. CERT r4.2
            has only two positive users, so the raw neg/pos ratio can force
            every prediction positive if used directly.
        calibrate_weighted_logits: If true, subtract log(pos_weight) from
            logits before converting weighted-BCE outputs to probabilities.
        threshold_min: Lower bound for selected classification thresholds.
        threshold_max_pred_positive_rate: Maximum validation positive
            prediction rate allowed during threshold search.
        threshold_min_recall: Preferred minimum recall during constrained
            threshold search.
        threshold_min_validation_positives: Below this many validation
            positives, F1 threshold tuning is considered statistically
            unstable and falls back to rate-constrained thresholding.
        patience: Number of epochs with no F1 improvement before early
            stopping triggers.
        min_delta: Minimum F1 improvement to reset the early-stopping
            counter.
        grad_clip_norm: Max norm used by ``clip_grad_norm_``.
        device: One of {"auto", "cpu", "cuda", "mps"}.
        num_workers: Reserved for future DataLoader-based shard loading.
        seed: Global random seed for reproducibility.
        train_split: Fraction of labeled nodes used for training.
        val_split: Fraction of labeled nodes used for validation.
        test_split: Fraction of labeled nodes used for held-out testing.
        step_size: Step size (in epochs) for StepLR.
        step_gamma: Multiplicative decay factor for StepLR.
        plateau_factor: Decay factor for ReduceLROnPlateau.
        plateau_patience: Patience (in epochs) for ReduceLROnPlateau.
        tgn_embedding_dim: Node embedding dimension produced by the TGN.
        gat_embedding_dim: Contextual embedding dimension produced by the
            GAT.
        log_every_n_shards: How often (in shards) to emit a progress log
            line within an epoch.
    """

    # --- Data locations -------------------------------------------------
    # Paths relative to training/ directory (will be resolved in __post_init__)
    graph_path: str = "graph/output/node_graph_skeleton.pt"
    edge_shard_dir: str = "graph/output/edge_feature_shards"
    labels_dir: str = "data/labels"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs/insider_threat_tgn_gat_mlp"

    # --- Optimization -----------------------------------------------------
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    loss: str = "bce_logits"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    max_pos_weight: float = 20.0
    calibrate_weighted_logits: bool = True
    threshold_min: float = 0.005001
    threshold_max_pred_positive_rate: float = 0.01
    threshold_min_recall: float = 0.80
    threshold_min_validation_positives: int = 3

    # --- Regularization / stability --------------------------------------
    patience: int = 10
    min_delta: float = 1e-4
    grad_clip_norm: float = 5.0

    # --- Runtime ----------------------------------------------------------
    device: str = "auto"
    num_workers: int = 4
    seed: int = 42

    # --- Data splits --------------------------------------------------
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15

    # --- Scheduler-specific knobs ------------------------------------------
    step_size: int = 20
    step_gamma: float = 0.5
    plateau_factor: float = 0.5
    plateau_patience: int = 5

    # --- Model dimensionality (must match production model configs) -------
    tgn_embedding_dim: int = 128
    gat_embedding_dim: int = 128

    # --- Logging ------------------------------------------------------
    log_every_n_shards: int = 10

    def __post_init__(self) -> None:
        """Resolve all paths relative to the project root (parent of training/)."""
        # Get the project root (parent of the training/ directory)
        training_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(training_dir)
        
        # Resolve all paths relative to project root
        if not os.path.isabs(self.graph_path):
            self.graph_path = os.path.join(project_root, self.graph_path)
        if not os.path.isabs(self.edge_shard_dir):
            self.edge_shard_dir = os.path.join(project_root, self.edge_shard_dir)
        if not os.path.isabs(self.labels_dir):
            self.labels_dir = os.path.join(project_root, self.labels_dir)
        if not os.path.isabs(self.checkpoint_dir):
            self.checkpoint_dir = os.path.join(project_root, self.checkpoint_dir)
        if not os.path.isabs(self.log_dir):
            self.log_dir = os.path.join(project_root, self.log_dir)

    def validate(self) -> None:
        """Raise ``ValueError`` if the configuration is internally inconsistent."""
        valid_optimizers = {"adam", "adamw", "sgd"}
        valid_schedulers = {"cosine", "step", "plateau"}
        valid_losses = {"bce", "bce_logits", "focal"}
        valid_devices = {"auto", "cpu", "cuda", "mps"}

        if self.optimizer.lower() not in valid_optimizers:
            raise ValueError(
                f"Unsupported optimizer '{self.optimizer}'. "
                f"Expected one of {sorted(valid_optimizers)}."
            )
        if self.scheduler.lower() not in valid_schedulers:
            raise ValueError(
                f"Unsupported scheduler '{self.scheduler}'. "
                f"Expected one of {sorted(valid_schedulers)}."
            )
        if self.loss.lower() not in valid_losses:
            raise ValueError(
                f"Unsupported loss '{self.loss}'. Expected one of {sorted(valid_losses)}."
            )
        if self.device.lower() not in valid_devices:
            raise ValueError(
                f"Unsupported device '{self.device}'. Expected one of {sorted(valid_devices)}."
            )
        splits_sum = self.train_split + self.val_split + self.test_split
        if abs(splits_sum - 1.0) > 1e-6:
            raise ValueError(
                "train_split + val_split + test_split must sum to 1.0, "
                f"got {splits_sum:.6f}."
            )
        if self.epochs < 0:
            raise ValueError("epochs must be a non-negative integer.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if self.patience <= 0:
            raise ValueError("patience must be a positive integer.")
        if self.max_pos_weight < 1.0:
            raise ValueError("max_pos_weight must be >= 1.0.")
        if not 0.0 < self.threshold_min < 1.0:
            raise ValueError("threshold_min must be in (0, 1).")
        if not 0.0 < self.threshold_max_pred_positive_rate <= 1.0:
            raise ValueError("threshold_max_pred_positive_rate must be in (0, 1].")
        if not 0.0 <= self.threshold_min_recall <= 1.0:
            raise ValueError("threshold_min_recall must be in [0, 1].")
        if self.threshold_min_validation_positives < 1:
            raise ValueError("threshold_min_validation_positives must be >= 1.")

    def to_dict(self) -> dict:
        """Return a plain-dict representation for logging/serialization."""
        return asdict(self)

    def save(self, path: str) -> None:
        """Persist the configuration as JSON for reproducibility."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        """Load a configuration previously written by :meth:`save`."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(**payload)
