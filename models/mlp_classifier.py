"""
MLP Risk Classifier — Production Module
=========================================

Rewritten from scratch. Consumes ONLY the real node embeddings produced
by the already-completed Graph Attention Network stage
(models/gat_model.py), which itself consumes the already-completed
Temporal Graph Network stage (models/tgn_model.py). This module never
modifies, copies-across-device, or regenerates upstream outputs — it
validates and classifies them.

    GAT node embeddings, shape [num_nodes, embedding_dim]
          |
        Linear
          |
     BatchNorm1d
          |
         ReLU
          |
       Dropout
          |
        Linear
          |
     BatchNorm1d
          |
         ReLU
          |
       Dropout
          |
        Linear
          |
       Sigmoid (inference only)
          |
    Risk Probability, shape [num_nodes, 1]  ∈ [0, 1]
    0 = Normal, 1 = Insider Threat

The network exposes raw logits via `forward()` / `MLPRiskClassifier.__call__`
for training with `BCEWithLogitsLoss` (numerically stable), and a
separate `predict_proba()` that applies sigmoid for inference only.

Run directly from the project root:

    python models/mlp_classifier.py

No sys.path manipulation, no PYTHONPATH required: when launched
directly, Python automatically prepends this script's own directory
(models/) to sys.path, so the sibling `tgn_model.py` / `gat_model.py`
imports below resolve without any manual path handling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tgn_model import TGNConfig, build_tgn_from_root
from .gat_model import GATConfig, ProductionGAT, ShardLoader

# ======================================================================
# Logging
# ======================================================================

logger = logging.getLogger("mlp_classifier")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    )
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


# ======================================================================
# Config
# ======================================================================

@dataclass
class MLPClassifierConfig:
    """Hyperparameters for the MLP risk classifier. Input dimension is
    deliberately NOT part of this config — it is always inferred from
    the real GAT output embeddings at construction time."""

    hidden_dim_1: int = 256
    hidden_dim_2: int = 128
    dropout: float = 0.3
    device: Optional[str] = None  # None -> auto-detect CUDA / CPU

    def resolved_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self) -> None:
        if self.hidden_dim_1 <= 0 or self.hidden_dim_2 <= 0:
            raise ValueError(
                f"hidden dimensions must be positive integers, got "
                f"hidden_dim_1={self.hidden_dim_1}, hidden_dim_2={self.hidden_dim_2}"
            )
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


# ======================================================================
# Loss
# ======================================================================

class LossType(str, Enum):
    BCE = "bce"                          # expects probabilities (post-sigmoid)
    BCE_WITH_LOGITS = "bce_with_logits"  # expects raw logits (numerically stable, preferred)


def _validate_binary_labels(labels: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(labels):
        raise TypeError(f"labels must be a torch.Tensor, got {type(labels)}")
    if torch.isnan(labels.float()).any():
        raise ValueError("labels contain NaN values")
    flat = labels.reshape(-1)
    unique_vals = torch.unique(flat)
    invalid = unique_vals[(unique_vals != 0) & (unique_vals != 1)]
    if invalid.numel() > 0:
        raise ValueError(
            f"Labels must be binary (0=Normal, 1=Insider Threat); "
            f"found invalid values: {invalid.tolist()}"
        )
    return flat


def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_type: LossType = LossType.BCE_WITH_LOGITS,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Binary classification loss for the risk classifier.

    `logits` must be the raw, pre-sigmoid output of `MLPRiskClassifier.forward()`.
    For `LossType.BCE_WITH_LOGITS` (the numerically stable, recommended
    default) logits are consumed directly. For `LossType.BCE`, logits are
    passed through sigmoid first to obtain probabilities.
    """
    if not torch.is_tensor(logits):
        raise TypeError(f"logits must be a torch.Tensor, got {type(logits)}")
    if torch.isnan(logits).any():
        raise ValueError("logits contain NaN values")
    if torch.isinf(logits).any():
        raise ValueError("logits contain Inf values")

    _validate_binary_labels(labels)
    labels = labels.to(device=logits.device, dtype=logits.dtype)
    if labels.dim() == 1:
        labels = labels.view(-1, 1)
    if labels.shape != logits.shape:
        raise ValueError(
            f"Label shape {tuple(labels.shape)} does not match logits shape {tuple(logits.shape)}"
        )
    if labels.device != logits.device:
        raise RuntimeError(
            f"Device mismatch: labels are on {labels.device} but logits are on {logits.device}"
        )

    if loss_type == LossType.BCE_WITH_LOGITS:
        return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
    if loss_type == LossType.BCE:
        probabilities = torch.sigmoid(logits)
        return F.binary_cross_entropy(probabilities, labels)
    raise ValueError(f"Unknown loss_type: {loss_type!r}")


# ======================================================================
# MLP Risk Classifier
# ======================================================================

class MLPRiskClassifier(nn.Module):
    """Binary insider-threat risk classifier operating on GAT node
    embeddings. Input dimension is fixed at construction time from the
    real embedding width — never hardcoded.

    `forward()` returns raw logits (for training with BCEWithLogitsLoss).
    `predict_proba()` returns sigmoid-activated probabilities and is
    intended for inference only.
    """

    def __init__(self, in_dim: int, config: Optional[MLPClassifierConfig] = None) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError(f"in_dim must be a positive integer, got {in_dim}")

        self.config = config or MLPClassifierConfig()
        self.device = self.config.resolved_device()
        self.in_dim = in_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, self.config.hidden_dim_1),
            nn.BatchNorm1d(self.config.hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim_1, self.config.hidden_dim_2),
            nn.BatchNorm1d(self.config.hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim_2, 1),
        )

        self.to(self.device)
        logger.info(
            "Initialized MLPRiskClassifier | in_dim=%d hidden_dim_1=%d hidden_dim_2=%d dropout=%.2f device=%s",
            in_dim, self.config.hidden_dim_1, self.config.hidden_dim_2, self.config.dropout, self.device,
        )

    def _validate_input(self, x: torch.Tensor) -> None:
        if not torch.is_tensor(x):
            raise TypeError(f"Input must be a torch.Tensor, got {type(x)}")
        if x.dim() != 2:
            raise ValueError(
                f"Expected a 2D tensor [num_nodes, embedding_dim], got shape {tuple(x.shape)}"
            )
        if x.size(-1) != self.in_dim:
            raise ValueError(
                f"Input embedding dimension {x.size(-1)} does not match classifier "
                f"in_dim {self.in_dim}. This classifier was built for a specific GAT "
                f"output width and cannot silently reinterpret a different dimension."
            )
        if x.device != self.device:
            raise RuntimeError(
                f"Device mismatch: input embeddings are on {x.device} but the classifier "
                f"is on {self.device}. This classifier never copies tensors across devices "
                f"implicitly (to avoid duplicating GAT output tensors) — move the embeddings "
                f"to {self.device} at the source (e.g. by matching GATConfig.device) before "
                f"calling forward()."
            )
        if torch.isnan(x).any():
            raise ValueError("Input embeddings contain NaN values")
        if torch.isinf(x).any():
            raise ValueError("Input embeddings contain Inf values")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits, shape [num_nodes, 1]. Use for training
        with `compute_loss(..., loss_type=LossType.BCE_WITH_LOGITS)`."""
        self._validate_input(x)
        logits = self.net(x)

        if torch.isnan(logits).any():
            raise RuntimeError("NaN detected in classifier logits")
        if torch.isinf(logits).any():
            raise RuntimeError("Inf detected in classifier logits")

        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Inference-only: returns sigmoid-activated risk probabilities,
        shape [num_nodes, 1], values in [0, 1]."""
        was_training = self.training
        self.eval()
        try:
            logits = self.forward(x)
            probabilities = torch.sigmoid(logits)
        finally:
            self.train(was_training)

        if torch.isnan(probabilities).any():
            raise RuntimeError("NaN detected in predicted probabilities")
        if torch.isinf(probabilities).any():
            raise RuntimeError("Inf detected in predicted probabilities")
        if (probabilities < 0.0).any() or (probabilities > 1.0).any():
            raise RuntimeError("Predicted probabilities fell outside the valid [0, 1] range")

        return probabilities

    @torch.no_grad()
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Inference-only: returns binary risk labels, shape [num_nodes, 1].
        0 = Normal, 1 = Insider Threat."""
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        probabilities = self.predict_proba(x)
        return (probabilities >= threshold).long()


def build_mlp_classifier_from_embeddings(
    embeddings: torch.Tensor, config: Optional[MLPClassifierConfig] = None
) -> MLPRiskClassifier:
    """Factory: infers `in_dim` directly from real GAT output embeddings
    (never hardcoded) and constructs a ready-to-use classifier."""
    if not torch.is_tensor(embeddings):
        raise TypeError(f"embeddings must be a torch.Tensor, got {type(embeddings)}")
    if embeddings.dim() != 2:
        raise ValueError(
            f"embeddings must be a 2D tensor [num_nodes, embedding_dim] to infer the "
            f"classifier input dimension; got shape {tuple(embeddings.shape)}"
        )
    in_dim = int(embeddings.size(-1))
    return MLPRiskClassifier(in_dim=in_dim, config=config)


# ======================================================================
# Metrics
# ======================================================================

@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float


def accuracy_score(labels: torch.Tensor, predictions: torch.Tensor) -> float:
    labels = _validate_binary_labels(labels)
    predictions = predictions.reshape(-1)
    if labels.shape != predictions.shape:
        raise ValueError(
            f"labels shape {tuple(labels.shape)} and predictions shape "
            f"{tuple(predictions.shape)} must match"
        )
    correct = (labels == predictions).float().sum()
    return float(correct / labels.numel())


def precision_score(labels: torch.Tensor, predictions: torch.Tensor) -> float:
    labels = _validate_binary_labels(labels)
    predictions = predictions.reshape(-1)
    true_positive = ((predictions == 1) & (labels == 1)).sum().float()
    false_positive = ((predictions == 1) & (labels == 0)).sum().float()
    denom = true_positive + false_positive
    return float(true_positive / denom) if denom > 0 else 0.0


def recall_score(labels: torch.Tensor, predictions: torch.Tensor) -> float:
    labels = _validate_binary_labels(labels)
    predictions = predictions.reshape(-1)
    true_positive = ((predictions == 1) & (labels == 1)).sum().float()
    false_negative = ((predictions == 0) & (labels == 1)).sum().float()
    denom = true_positive + false_negative
    return float(true_positive / denom) if denom > 0 else 0.0


def f1_score(labels: torch.Tensor, predictions: torch.Tensor) -> float:
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    denom = precision + recall
    return float(2.0 * precision * recall / denom) if denom > 0 else 0.0


def _rank_data(scores: torch.Tensor) -> torch.Tensor:
    """Assigns average ranks (1-indexed, ties averaged) to `scores`,
    vectorized via sort + unique-group boundaries."""
    sorted_scores, sorted_idx = torch.sort(scores)
    n = scores.numel()
    unique_vals, inverse, counts = torch.unique(sorted_scores, return_inverse=True, return_counts=True)
    cum = torch.cumsum(counts, dim=0).to(torch.float32)
    start = cum - counts.to(torch.float32) + 1.0
    avg_rank_per_group = (start + cum) / 2.0
    ranks_sorted = avg_rank_per_group[inverse]
    ranks = torch.empty(n, dtype=torch.float32, device=scores.device)
    ranks[sorted_idx] = ranks_sorted
    return ranks


def roc_auc_score(labels: torch.Tensor, scores: torch.Tensor) -> float:
    """Mann-Whitney U / Wilcoxon rank-sum formulation of ROC-AUC,
    computed in closed form from ranked scores (equivalent to
    integrating the ROC curve, without an explicit threshold sweep)."""
    labels = _validate_binary_labels(labels).float()
    scores = scores.reshape(-1).float()
    if labels.shape != scores.shape:
        raise ValueError(
            f"labels shape {tuple(labels.shape)} and scores shape {tuple(scores.shape)} must match"
        )
    if torch.isnan(scores).any():
        raise ValueError("scores contain NaN values")
    if torch.isinf(scores).any():
        raise ValueError("scores contain Inf values")

    n_pos = labels.sum()
    n_neg = labels.numel() - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            "ROC-AUC is undefined when labels contain only one class; "
            f"got n_pos={int(n_pos)}, n_neg={int(n_neg)}"
        )

    ranks = _rank_data(scores)
    sum_ranks_pos = ranks[labels == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def compute_classification_metrics(
    labels: torch.Tensor, probabilities: torch.Tensor, threshold: float = 0.5
) -> ClassificationMetrics:
    """Convenience aggregator producing all standard binary-classification
    metrics from ground-truth labels and predicted probabilities."""
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    labels = _validate_binary_labels(labels)
    probabilities = probabilities.reshape(-1)
    if torch.isnan(probabilities).any():
        raise ValueError("probabilities contain NaN values")
    if torch.isinf(probabilities).any():
        raise ValueError("probabilities contain Inf values")

    predictions = (probabilities >= threshold).long()

    return ClassificationMetrics(
        accuracy=accuracy_score(labels, predictions),
        precision=precision_score(labels, predictions),
        recall=recall_score(labels, predictions),
        f1_score=f1_score(labels, predictions),
        roc_auc=roc_auc_score(labels, probabilities),
    )


# ======================================================================
# Production self-test
# ======================================================================

def _self_test() -> None:
    """Loads one real production shard, runs the real TGN, feeds its
    output into the real GAT, feeds that output into the real MLP risk
    classifier, and validates the result end to end. No fabricated
    tensors, no synthetic graphs, no synthetic embeddings."""
    project_root = Path(__file__).resolve().parent.parent
    graph_output_root = project_root / "graph" / "output"

    print("=" * 70)
    print()
    print("MLP CLASSIFIER PRODUCTION TEST")
    print()
    print("Loading production artifacts...")

    tgn_config = TGNConfig()
    tgn_model, shard_map = build_tgn_from_root(graph_output_root, tgn_config)
    tgn_model.eval()

    loader = ShardLoader(tgn_model, graph_output_root)
    first_relation = next(iter(shard_map))
    first_shard_path = shard_map[first_relation][0]

    print("Running TGN...")
    with torch.no_grad():
        graph = loader.load_graph_for_shard(first_shard_path)

    print("Running GAT...")
    gat_config = GATConfig(
        in_dim=tgn_config.embedding_dim,
        hidden_dim=tgn_config.embedding_dim,
        out_dim=tgn_config.embedding_dim,
        heads=4,
        dropout=0.2,
        device=tgn_config.device,
    )
    gat_model = ProductionGAT(gat_config)
    gat_model.eval()
    with torch.no_grad():
        node_embeddings, _attention = gat_model(
            graph["node_embeddings"], graph["edge_index"], graph["edge_weight"]
        )

    print("Running MLP...")
    mlp_config = MLPClassifierConfig(device=tgn_config.device)
    classifier = build_mlp_classifier_from_embeddings(node_embeddings, mlp_config)
    classifier.eval()

    with torch.no_grad():
        probabilities = classifier.predict_proba(node_embeddings)
        logits = classifier.forward(node_embeddings)

    num_nodes = node_embeddings.size(0)

    print()
    print("Node embeddings:")
    print(tuple(node_embeddings.shape))
    print()
    print("Output probabilities:")
    print(tuple(probabilities.shape))
    print()

    min_prob = float(probabilities.min())
    max_prob = float(probabilities.max())
    mean_prob = float(probabilities.mean())

    print("Min probability:")
    print(f"{min_prob:.6f}")
    print()
    print("Max probability:")
    print(f"{max_prob:.6f}")
    print()
    print("Mean probability:")
    print(f"{mean_prob:.6f}")
    print()

    no_nan = not bool(torch.isnan(probabilities).any())
    no_inf = not bool(torch.isinf(probabilities).any())
    probs_in_range = bool(((probabilities >= 0.0) & (probabilities <= 1.0)).all())
    shape_ok = probabilities.shape == (num_nodes, 1)
    logits_shape_ok = logits.shape == (num_nodes, 1)
    device_ok = probabilities.device.type == classifier.device.type

    assert no_nan, "NaN detected in output probabilities"
    assert no_inf, "Inf detected in output probabilities"
    assert probs_in_range, "Probabilities fell outside the [0, 1] range"
    assert shape_ok, f"Output shape mismatch: {tuple(probabilities.shape)} != {(num_nodes, 1)}"
    assert logits_shape_ok, f"Logits shape mismatch: {tuple(logits.shape)} != {(num_nodes, 1)}"
    assert device_ok, "Device mismatch between classifier output and classifier model"

    print(f"No NaN:")
    print("PASS" if no_nan else "FAIL")
    print()
    print(f"No Inf:")
    print("PASS" if no_inf else "FAIL")
    print()
    print(f"Output Shape:")
    print("PASS" if shape_ok else "FAIL")
    print()
    print("Device:")
    print(classifier.device)
    print()
    print("=" * 70)
    print()
    print("PASS")


if __name__ == "__main__":
    _self_test()

# Backward compatibility
MLPClassifier = MLPRiskClassifier