#!/usr/bin/env python3
"""
training/train.py

Production training pipeline for the Insider Threat Detection System.

Pipeline stage this file implements:

    Load production graph
        -> Load production edge feature shards
        -> Initialize TGN
        -> Initialize GAT
        -> Initialize MLP
        -> Forward -> Loss -> Backward -> Optimizer
        -> Checkpoint -> Metrics logging

This module NEVER redefines model architecture. It only imports and calls
the existing production modules:

    graph.build_event_graph
    graph.edge_features
    graph.edge_weighting
    models.tgn_model
    models.gat_model
    models.mlp_classifier

--------------------------------------------------------------------------
INTERFACE ASSUMPTIONS (read this before wiring into your project)
--------------------------------------------------------------------------
This file was written without access to the production module source, so
it integrates against them through a small adapter layer
(`_ProductionInterface`) rather than hardcoding one exact call signature.
The adapter tries several conventional entry points, in this order, and
uses the first one that exists / succeeds:

  graph.build_event_graph
      - load_graph(path) | load_production_graph(path) | load(path) | build_graph(path)
      -> object exposing one of: `.num_nodes` / `num_nodes(obj)` and
         `.node_features` / `.x` (a [num_nodes, F] tensor), used to size and
         (optionally) seed the TGN's node memory.

  graph.edge_features
      - load_edge_feature_shard(path) | load_shard(path) | load(path)
      -> dict-like or object exposing `src`, `dst`, `t` (timestamps) and
         `msg` (edge feature tensor), each aligned along dim 0.

  graph.edge_weighting
      - compute_edge_weights(batch) | apply_edge_weights(batch) | weight(batch)
      -> tensor of per-edge weights aligned with the batch, OR the batch
         with weights already attached under `edge_weight` / `weight`.
        This step is OPTIONAL: if edge_weighting exposes none of the above,
        training proceeds without explicit edge weights (production module
        contract is preserved either way; nothing is invented).

  models.tgn_model.TGN
      - Constructed as TGN(**tgn_kwargs) from `TrainingConfig`. Called as
        `model(batch)` or `model(**batch)` or
        `model(src, dst, t, msg)`, returning either a single embedding
        tensor spanning all nodes touched by the batch, or a tuple of
        (src_embeddings, dst_embeddings).

  models.gat_model.GAT
      - Called as `model(node_embeddings, edge_index)` or
        `model(node_embeddings, edge_index, edge_weight)`, returning
        contextual node embeddings of shape [num_nodes, gat_embedding_dim].

  models.mlp_classifier.MLPClassifier
      - Called as `model(node_embeddings)`, returning per-node logits of
        shape [num_nodes] or [num_nodes, 1].

If your actual production modules expose different names, either (a) add
a thin alias inside them (not a rewrite -- just an additional exported
name), or (b) extend the `_CANDIDATE_*` tuples below with your real names.
No behavior in this file depends on guessing data values -- only on
locating already-implemented entry points.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import inspect
import logging
import os
import pickle
import random
import sys
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.config import TrainingConfig  # noqa: E402

try:
    from torch.amp import GradScaler as _TorchAmpGradScaler
    from torch.amp import autocast as _TorchAmpAutocast

    def GradScaler(enabled: bool = True) -> Any:  # noqa: N802 - factory mimics class name
        """Construct a CUDA GradScaler using the modern torch.amp API."""
        return _TorchAmpGradScaler("cuda", enabled=enabled)

    def autocast(enabled: bool = True) -> Any:  # noqa: N802 - factory mimics class name
        """Construct a CUDA autocast context using the modern torch.amp API."""
        return _TorchAmpAutocast("cuda", enabled=enabled)

except ImportError:  # pragma: no cover - torch too old for torch.amp
    try:
        from torch.cuda.amp import GradScaler, autocast  # type: ignore
    except ImportError:
        GradScaler = None  # type: ignore
        autocast = None  # type: ignore

LOGGER = logging.getLogger("insider_threat.training")


# ==========================================================================
# Production module imports
# ==========================================================================
# These imports intentionally target the *existing* production files only.
# If they are missing, we fail loudly and immediately rather than falling
# back to any placeholder/synthetic implementation.

_MISSING_MODULES: List[str] = []

try:
    import graph.build_event_graph as build_event_graph_module
except ImportError as exc:
    build_event_graph_module = None
    _MISSING_MODULES.append(f"graph.build_event_graph ({exc})")

try:
    import graph.edge_features as edge_features_module
except ImportError as exc:
    edge_features_module = None
    _MISSING_MODULES.append(f"graph.edge_features ({exc})")

try:
    import graph.edge_weighting as edge_weighting_module
except ImportError as exc:
    edge_weighting_module = None
    _MISSING_MODULES.append(f"graph.edge_weighting ({exc})")

try:
    from models.tgn_model import TGN
except ImportError as exc:
    TGN = None  # type: ignore
    _MISSING_MODULES.append(f"models.tgn_model.TGN ({exc})")

try:
    from models.gat_model import GAT
except ImportError as exc:
    GAT = None  # type: ignore
    _MISSING_MODULES.append(f"models.gat_model.GAT ({exc})")

try:
    from models.mlp_classifier import MLPClassifier
except ImportError as exc:
    MLPClassifier = None  # type: ignore
    _MISSING_MODULES.append(f"models.mlp_classifier.MLPClassifier ({exc})")


class ProductionArtifactError(RuntimeError):
    """Raised when a required production artifact is missing or invalid."""


def _require_production_modules() -> None:
    """Fail fast (and loud) if any production module failed to import."""
    if _MISSING_MODULES:
        joined = "\n  - ".join(_MISSING_MODULES)
        raise ProductionArtifactError(
            "Cannot start training: the following production modules could "
            "not be imported. This pipeline is not permitted to substitute "
            "placeholder or synthetic implementations for them.\n  - "
            f"{joined}\n"
            "Ensure 'graph/' and 'models/' are on PYTHONPATH and that each "
            "file exists exactly as already validated by their self-tests."
        )


# ==========================================================================
# Reproducibility / device utilities
# ==========================================================================

def set_global_seed(seed: int) -> None:
    """Seed python, numpy, torch (CPU) and torch.cuda for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """Resolve 'auto'/'cpu'/'cuda'/'mps' into a concrete torch.device."""
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available on this machine.")
    if requested == "mps" and not (
        getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available on this machine.")
    return torch.device(requested)


def current_gpu_memory_mb(device: torch.device) -> float:
    """Return current allocated GPU memory in MB, or 0.0 on non-CUDA devices."""
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / (1024 ** 2)
    return 0.0


# ==========================================================================
# Adapter helpers for calling production entry points without guessing data
# ==========================================================================

def _first_callable(module: Any, candidate_names: Sequence[str]) -> Callable:
    """Return the first attribute in `candidate_names` that exists on `module`
    and is callable. Raises ProductionArtifactError if none are found.
    """
    if module is None:
        raise ProductionArtifactError("Attempted to use a module that failed to import.")
    for name in candidate_names:
        attr = getattr(module, name, None)
        if callable(attr):
            return attr
    raise ProductionArtifactError(
        f"None of the expected entry points {list(candidate_names)} were "
        f"found on module '{module.__name__}'. Update the "
        "`_CANDIDATE_*` tuples in training/train.py to match your actual "
        "production API."
    )


def _call_flexibly(func: Callable, *args: Any, **kwargs: Any) -> Any:
    """Call `func` with the given kwargs, falling back to positional args
    when the function's signature does not accept the provided keywords.
    """
    try:
        sig = inspect.signature(func)
        accepted = set(sig.parameters.keys())
        if all(k in accepted for k in kwargs) or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        ):
            return func(*args, **kwargs)
    except (TypeError, ValueError):
        pass
    return func(*args)


_CANDIDATE_LOAD_GRAPH = ("load_graph", "load_production_graph", "load", "build_graph")
_CANDIDATE_LOAD_SHARD = ("load_edge_feature_shard", "load_shard", "load")
_CANDIDATE_EDGE_WEIGHT = ("compute_edge_weights", "apply_edge_weights", "weight")


def load_production_graph(graph_path: str) -> Any:
    """Load the production graph via graph.build_event_graph.

    Raises:
        ProductionArtifactError: if the graph file is missing or the loader
            entry point cannot be located.
    """
    if not os.path.exists(graph_path):
        raise ProductionArtifactError(
            f"Production graph not found at '{graph_path}'. Run the graph "
            "construction pipeline (graph/build_event_graph.py) first."
        )
    loader = _first_callable(build_event_graph_module, _CANDIDATE_LOAD_GRAPH)
    graph_obj = loader(graph_path)
    if graph_obj is None:
        raise ProductionArtifactError(
            f"graph.build_event_graph loader returned None for '{graph_path}'."
        )
    return graph_obj


def infer_num_nodes(graph_obj: Any) -> int:
    """Best-effort extraction of node count from the production graph object."""
    for attr in ("num_nodes", "n_nodes", "node_count"):
        value = getattr(graph_obj, attr, None)
        if isinstance(value, int):
            return value
    for attr in ("node_features", "x", "node_ids"):
        value = getattr(graph_obj, attr, None)
        if value is not None and hasattr(value, "__len__"):
            return len(value)
    if isinstance(graph_obj, dict):
        for key in ("num_nodes", "n_nodes"):
            if key in graph_obj:
                return int(graph_obj[key])
        for key in ("node_features", "x", "node_ids"):
            if key in graph_obj:
                return len(graph_obj[key])
    raise ProductionArtifactError(
        "Could not determine num_nodes from the production graph object. "
        "Expected one of attributes/keys: num_nodes, n_nodes, node_count, "
        "node_features, x, node_ids."
    )


def discover_edge_shards(shard_dir: str) -> List[str]:
    """Return sorted paths to edge_features_XXXXXX.pt shards.

    Raises:
        ProductionArtifactError: if the directory or shards do not exist.
    """
    if not os.path.isdir(shard_dir):
        raise ProductionArtifactError(
            f"Edge feature shard directory not found: '{shard_dir}'."
        )
    pattern = os.path.join(shard_dir, "edge_features_*.pt")
    shards = sorted(glob.glob(pattern))
    if not shards:
        raise ProductionArtifactError(
            f"No shards matching 'edge_features_*.pt' were found in "
            f"'{shard_dir}'. Run graph/edge_features.py to produce them."
        )
    return shards


def stream_edge_shards(shard_dir: str, device: torch.device) -> Iterator[Any]:
    """Yield one edge-feature shard at a time (never all shards at once)."""
    loader = _first_callable(edge_features_module, _CANDIDATE_LOAD_SHARD)
    for shard_path in discover_edge_shards(shard_dir):
        shard = loader(shard_path)
        if shard is None:
            raise ProductionArtifactError(
                f"graph.edge_features loader returned None for '{shard_path}'."
            )
        yield _move_shard_to_device(shard, device)


def _move_shard_to_device(shard: Any, device: torch.device) -> Any:
    """Move tensor fields of a shard (dict or object) onto `device`."""
    if isinstance(shard, dict):
        return {
            key: (value.to(device) if isinstance(value, torch.Tensor) else value)
            for key, value in shard.items()
        }
    for attr_name, attr_value in vars(shard).items() if hasattr(shard, "__dict__") else []:
        if isinstance(attr_value, torch.Tensor):
            setattr(shard, attr_name, attr_value.to(device))
    return shard


def apply_edge_weighting(shard: Any) -> Any:
    """Apply dynamic edge weighting to a shard if the module exposes an entry
    point for it. This step is optional and never fabricates weights.
    """
    if edge_weighting_module is None:
        return shard
    try:
        weight_fn = _first_callable(edge_weighting_module, _CANDIDATE_EDGE_WEIGHT)
    except ProductionArtifactError:
        return shard
    result = _call_flexibly(weight_fn, shard)
    if isinstance(result, torch.Tensor):
        if isinstance(shard, dict):
            shard = dict(shard)
            shard["edge_weight"] = result
        else:
            setattr(shard, "edge_weight", result)
        return shard
    if result is not None:
        return result
    return shard


def _shard_field(shard: Any, *names: str) -> Optional[Any]:
    """Fetch the first present field from a dict-like or attribute-like shard."""
    if isinstance(shard, dict):
        for name in names:
            if name in shard:
                return shard[name]
        return None
    for name in names:
        if hasattr(shard, name):
            return getattr(shard, name)
    return None


# ==========================================================================
# Label loading (labels.pt / labels.csv / labels.pkl - auto-detected)
# ==========================================================================

class LabelLoader:
    """Detects and loads node-level training labels without requiring the
    caller to know the on-disk format ahead of time.

    Exactly one of `labels.pt`, `labels.csv`, or `labels.pkl` must be
    present inside `labels_dir`. If none are present, an informative
    exception is raised -- labels are never synthesized.
    """

    SUPPORTED_FILENAMES = ("labels.pt", "labels.csv", "labels.pkl")

    def __init__(self, labels_dir: str) -> None:
        self.labels_dir = labels_dir

    def _find_label_file(self) -> str:
        if not os.path.isdir(self.labels_dir):
            raise ProductionArtifactError(
                f"Labels directory not found: '{self.labels_dir}'. Node-level "
                "labels are required to train the MLP risk classifier."
            )
        found = [
            fname
            for fname in self.SUPPORTED_FILENAMES
            if os.path.isfile(os.path.join(self.labels_dir, fname))
        ]
        if not found:
            raise ProductionArtifactError(
                f"No label file found in '{self.labels_dir}'. Expected one "
                f"of {self.SUPPORTED_FILENAMES}. Training cannot proceed "
                "without real, node-level ground truth labels -- random or "
                "synthetic labels are not permitted."
            )
        if len(found) > 1:
            raise ProductionArtifactError(
                f"Multiple label files found in '{self.labels_dir}': {found}. "
                "Keep exactly one to avoid ambiguity about which is authoritative."
            )
        return os.path.join(self.labels_dir, found[0])

    def load(self) -> Dict[int, float]:
        """Load labels and return a mapping of {node_id: label}."""
        path = self._find_label_file()
        if path.endswith(".pt"):
            return self._load_pt(path)
        if path.endswith(".csv"):
            return self._load_csv(path)
        if path.endswith(".pkl"):
            return self._load_pkl(path)
        raise ProductionArtifactError(f"Unsupported label file extension: '{path}'.")

    @staticmethod
    def _load_pt(path: str) -> Dict[int, float]:
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict) and not isinstance(payload, torch.Tensor):
            # Could be {node_id: label} directly, or {"node_ids":..., "labels":...}
            if "node_ids" in payload and "labels" in payload:
                node_ids = payload["node_ids"]
                labels = payload["labels"]
                return {
                    int(n): float(v)
                    for n, v in zip(_to_list(node_ids), _to_list(labels))
                }
            return {int(k): float(v) for k, v in payload.items()}
        if isinstance(payload, torch.Tensor):
            return {idx: float(val) for idx, val in enumerate(payload.tolist())}
        raise ProductionArtifactError(
            f"Unrecognized structure inside labels.pt at '{path}'. Expected "
            "a dict of {node_id: label}, a dict with 'node_ids'/'labels' "
            "keys, or a flat tensor indexed by node id."
        )

    @staticmethod
    def _load_csv(path: str) -> Dict[int, float]:
        frame = pd.read_csv(path)
        id_col = _resolve_column(frame, ("node_id", "node", "id"))
        label_col = _resolve_column(frame, ("label", "risk", "y", "target"))
        if id_col is None or label_col is None:
            raise ProductionArtifactError(
                f"labels.csv at '{path}' must contain a node-id column "
                "(one of: node_id, node, id) and a label column "
                "(one of: label, risk, y, target). Found columns: "
                f"{list(frame.columns)}."
            )
        return {
            int(row[id_col]): float(row[label_col]) for _, row in frame.iterrows()
        }

    @staticmethod
    def _load_pkl(path: str) -> Dict[int, float]:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        if isinstance(payload, dict):
            return {int(k): float(v) for k, v in payload.items()}
        if isinstance(payload, (list, tuple, np.ndarray)):
            return {idx: float(val) for idx, val in enumerate(payload)}
        raise ProductionArtifactError(
            f"Unrecognized structure inside labels.pkl at '{path}'. Expected "
            "a dict of {node_id: label} or a flat sequence indexed by node id."
        )


def _to_list(value: Any) -> List[Any]:
    if isinstance(value, torch.Tensor):
        return value.tolist()
    return list(value)


def _resolve_column(frame: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lowered = {c.lower(): c for c in frame.columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


# ==========================================================================
# Loss functions
# ==========================================================================

class FocalLoss(nn.Module):
    """Binary focal loss operating on raw logits.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self._bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self._bce(logits, targets)
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        modulating_factor = (1.0 - p_t).clamp(min=1e-8) ** self.gamma
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_factor * modulating_factor * bce_loss).mean()


def build_loss_fn(config: TrainingConfig) -> nn.Module:
    """Instantiate the configured loss function."""
    loss_name = config.loss.lower()
    if loss_name == "bce":
        return nn.BCELoss()
    if loss_name == "bce_logits":
        return nn.BCEWithLogitsLoss()
    if loss_name == "focal":
        return FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)
    raise ValueError(f"Unsupported loss '{config.loss}'.")


def build_optimizer(config: TrainingConfig, parameters: Iterator[nn.Parameter]) -> optim.Optimizer:
    """Instantiate the configured optimizer."""
    params = list(parameters)
    if not params:
        raise ProductionArtifactError(
            "No trainable parameters were collected from TGN/GAT/MLP. "
            "Verify the production models expose parameters()."
        )
    name = config.optimizer.lower()
    if name == "adam":
        return optim.Adam(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    if name == "adamw":
        return optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    if name == "sgd":
        return optim.SGD(
            params, lr=config.learning_rate, weight_decay=config.weight_decay, momentum=0.9
        )
    raise ValueError(f"Unsupported optimizer '{config.optimizer}'.")


def build_scheduler(
    config: TrainingConfig, optimizer: optim.Optimizer
) -> optim.lr_scheduler._LRScheduler:
    """Instantiate the configured LR scheduler."""
    name = config.scheduler.lower()
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    if name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=config.step_size, gamma=config.step_gamma
        )
    if name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=config.plateau_factor,
            patience=config.plateau_patience,
        )
    raise ValueError(f"Unsupported scheduler '{config.scheduler}'.")


# ==========================================================================
# Early stopping
# ==========================================================================

class EarlyStopping:
    """Stops training when validation F1 fails to improve for `patience` epochs."""

    def __init__(self, patience: int, min_delta: float) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_score: float = float("-inf")
        self.num_bad_epochs: int = 0
        self.should_stop: bool = False

    def step(self, current_score: float) -> bool:
        """Update state with the latest validation F1. Returns True if this
        epoch produced a new best score.
        """
        if current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.num_bad_epochs = 0
            return True
        self.num_bad_epochs += 1
        if self.num_bad_epochs >= self.patience:
            self.should_stop = True
        return False

    def state_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(
            dataclasses.make_dataclass(
                "EarlyStoppingState",
                [("best_score", float), ("num_bad_epochs", int), ("should_stop", bool)],
            )(self.best_score, self.num_bad_epochs, self.should_stop)
        )

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.best_score = state["best_score"]
        self.num_bad_epochs = state["num_bad_epochs"]
        self.should_stop = state["should_stop"]


# ==========================================================================
# Metrics
# ==========================================================================

def compute_classification_metrics(
    targets: np.ndarray, probs: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Compute accuracy/precision/recall/F1/ROC-AUC/PR-AUC.

    ROC-AUC and PR-AUC are set to float('nan') when the batch contains only
    one class, since they are undefined in that case.
    """
    preds = (probs >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(targets, preds),
        "precision": precision_score(targets, preds, zero_division=0),
        "recall": recall_score(targets, preds, zero_division=0),
        "f1": f1_score(targets, preds, zero_division=0),
    }
    if len(np.unique(targets)) > 1:
        metrics["roc_auc"] = roc_auc_score(targets, probs)
        metrics["pr_auc"] = average_precision_score(targets, probs)
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    return metrics


# ==========================================================================
# Checkpointing
# ==========================================================================

class CheckpointManager:
    """Handles best/last checkpoints plus optimizer/scheduler/training state,
    and resumption from disk.
    """

    def __init__(self, checkpoint_dir: str) -> None:
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _path(self, name: str) -> str:
        return os.path.join(self.checkpoint_dir, name)

    def save(
        self,
        tgn: nn.Module,
        gat: nn.Module,
        mlp: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Any,
        epoch: int,
        best_f1: float,
        early_stopping: EarlyStopping,
        is_best: bool,
    ) -> None:
        """Persist last_model.pt (always) and best_model.pt (if is_best),
        plus optimizer/scheduler/training_state.
        """
        model_state = {
            "tgn": tgn.state_dict(),
            "gat": gat.state_dict(),
            "mlp": mlp.state_dict(),
            "epoch": epoch,
            "best_f1": best_f1,
        }
        torch.save(model_state, self._path("last_model.pt"))
        if is_best:
            torch.save(model_state, self._path("best_model.pt"))
        torch.save(optimizer.state_dict(), self._path("optimizer.pt"))
        torch.save(scheduler.state_dict(), self._path("scheduler.pt"))
        torch.save(
            {
                "epoch": epoch,
                "best_f1": best_f1,
                "early_stopping": early_stopping.state_dict(),
            },
            self._path("training_state.pt"),
        )

    def has_resumable_state(self) -> bool:
        return all(
            os.path.isfile(self._path(name))
            for name in (
                "last_model.pt",
                "optimizer.pt",
                "scheduler.pt",
                "training_state.pt",
            )
        )

    def load_for_resume(
        self,
        tgn: nn.Module,
        gat: nn.Module,
        mlp: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Any,
        early_stopping: EarlyStopping,
        device: torch.device,
    ) -> Tuple[int, float]:
        """Restore all training state in-place. Returns (start_epoch, best_f1)."""
        if not self.has_resumable_state():
            raise ProductionArtifactError(
                f"--resume was requested but no complete checkpoint set was "
                f"found in '{self.checkpoint_dir}'."
            )
        model_state = torch.load(self._path("last_model.pt"), map_location=device)
        tgn.load_state_dict(model_state["tgn"])
        gat.load_state_dict(model_state["gat"])
        mlp.load_state_dict(model_state["mlp"])

        optimizer.load_state_dict(torch.load(self._path("optimizer.pt"), map_location=device))
        scheduler.load_state_dict(torch.load(self._path("scheduler.pt"), map_location=device))

        training_state = torch.load(self._path("training_state.pt"), map_location=device)
        early_stopping.load_state_dict(training_state["early_stopping"])
        start_epoch = training_state["epoch"] + 1
        best_f1 = training_state["best_f1"]
        return start_epoch, best_f1


# ==========================================================================
# Trainer
# ==========================================================================

class InsiderThreatTrainer:
    """Owns the full TGN -> GAT -> MLP training loop, validation, metrics,
    checkpointing, early stopping, and TensorBoard logging.
    """

    def __init__(self, config: TrainingConfig) -> None:
        config.validate()
        self.config = config
        self.device = resolve_device(config.device)
        LOGGER.info("Using device: %s", self.device)

        set_global_seed(config.seed)

        self.graph_obj = load_production_graph(config.graph_path)
        self.num_nodes = infer_num_nodes(self.graph_obj)
        LOGGER.info("Loaded production graph with %d nodes.", self.num_nodes)

        label_map = LabelLoader(config.labels_dir).load()
        self.node_ids, self.labels = self._prepare_labels(label_map)
        LOGGER.info("Loaded %d node-level labels.", len(self.node_ids))

        self.train_ids, self.val_ids, self.test_ids = self._split_nodes(self.node_ids)
        LOGGER.info(
            "Split sizes -- train: %d, val: %d, test: %d",
            len(self.train_ids), len(self.val_ids), len(self.test_ids),
        )

        self.tgn = self._build_tgn().to(self.device)
        self.gat = self._build_gat().to(self.device)
        self.mlp = self._build_mlp().to(self.device)

        all_params = (
            list(self.tgn.parameters())
            + list(self.gat.parameters())
            + list(self.mlp.parameters())
        )
        self.optimizer = build_optimizer(config, iter(all_params))
        self.scheduler = build_scheduler(config, self.optimizer)
        self.loss_fn = build_loss_fn(config).to(self.device)

        self.use_amp = self.device.type == "cuda" and GradScaler is not None
        self.scaler = GradScaler(enabled=self.use_amp) if GradScaler is not None else None
        LOGGER.info("Mixed precision (AMP) enabled: %s", self.use_amp)

        self.early_stopping = EarlyStopping(config.patience, config.min_delta)
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir)
        self.writer = SummaryWriter(log_dir=config.log_dir)

        self.start_epoch = 0
        self.best_f1 = float("-inf")

    # -- construction helpers ---------------------------------------------

    def _prepare_labels(self, label_map: Dict[int, float]) -> Tuple[np.ndarray, torch.Tensor]:
        node_ids = np.array(sorted(label_map.keys()), dtype=np.int64)
        labels = torch.tensor(
            [label_map[int(n)] for n in node_ids], dtype=torch.float32
        )
        if node_ids.size == 0:
            raise ProductionArtifactError("Loaded labels are empty; cannot train.")
        out_of_range = node_ids[(node_ids < 0) | (node_ids >= self.num_nodes)]
        if out_of_range.size > 0:
            raise ProductionArtifactError(
                f"{out_of_range.size} labeled node id(s) fall outside the "
                f"graph's node range [0, {self.num_nodes}). Example offending "
                f"ids: {out_of_range[:5].tolist()}."
            )
        return node_ids, labels

    def _split_nodes(
        self, node_ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.RandomState(self.config.seed)
        shuffled = node_ids.copy()
        rng.shuffle(shuffled)
        n_total = len(shuffled)
        n_train = int(round(n_total * self.config.train_split))
        n_val = int(round(n_total * self.config.val_split))
        train_ids = shuffled[:n_train]
        val_ids = shuffled[n_train:n_train + n_val]
        test_ids = shuffled[n_train + n_val:]
        return train_ids, val_ids, test_ids

    def _build_tgn(self) -> nn.Module:
        if TGN is None:
            raise ProductionArtifactError("models.tgn_model.TGN failed to import.")
        return _instantiate_flexibly(
            TGN,
            preferred_kwargs={
                "num_nodes": self.num_nodes,
                "embedding_dim": self.config.tgn_embedding_dim,
                "memory_dim": self.config.tgn_embedding_dim,
                "device": self.device,
            },
        )

    def _build_gat(self) -> nn.Module:
        if GAT is None:
            raise ProductionArtifactError("models.gat_model.GAT failed to import.")
        return _instantiate_flexibly(
            GAT,
            preferred_kwargs={
                "in_channels": self.config.tgn_embedding_dim,
                "out_channels": self.config.gat_embedding_dim,
                "hidden_channels": self.config.gat_embedding_dim,
            },
        )

    def _build_mlp(self) -> nn.Module:
        if MLPClassifier is None:
            raise ProductionArtifactError("models.mlp_classifier.MLPClassifier failed to import.")
        return _instantiate_flexibly(
            MLPClassifier,
            preferred_kwargs={
                "input_dim": self.config.gat_embedding_dim,
                "in_features": self.config.gat_embedding_dim,
                "num_classes": 1,
                "output_dim": 1,
            },
        )

    # -- forward pass --------------------------------------------------------

    def _forward_shard(self, shard: Any) -> torch.Tensor:
        """Run TGN -> GAT -> MLP for a single edge-feature shard and return
        node-embedding-derived logits for every node touched by the shard,
        indexed by node id via a full-size [num_nodes] tensor filled with
        -inf for untouched nodes (so downstream code can mask them out).
        """
        src = _shard_field(shard, "src", "source", "src_ids")
        dst = _shard_field(shard, "dst", "target", "dst_ids")
        t = _shard_field(shard, "t", "timestamp", "timestamps")
        msg = _shard_field(shard, "msg", "edge_attr", "edge_features")
        edge_weight = _shard_field(shard, "edge_weight", "weight")

        if src is None or dst is None:
            raise ProductionArtifactError(
                "Edge shard is missing 'src'/'dst' fields required to run "
                "the TGN. Check graph/edge_features.py's shard schema."
            )

        tgn_batch = {"src": src, "dst": dst, "t": t, "msg": msg}
        if edge_weight is not None:
            tgn_batch["edge_weight"] = edge_weight

        node_embeddings = _call_model_flexibly(self.tgn, tgn_batch, ("src", "dst", "t", "msg"))
        if isinstance(node_embeddings, tuple):
            # Assume (src_embeddings, dst_embeddings) sharing the embedding dim.
            node_embeddings = torch.cat(node_embeddings, dim=0)

        touched_ids = torch.unique(torch.cat([_as_long_tensor(src), _as_long_tensor(dst)]))
        edge_index = torch.stack([_as_long_tensor(src), _as_long_tensor(dst)], dim=0)

        gat_kwargs: Dict[str, Any] = {"edge_index": edge_index}
        if edge_weight is not None:
            gat_kwargs["edge_weight"] = edge_weight
        contextual_embeddings = _call_model_flexibly(
            self.gat, {"x": node_embeddings, **gat_kwargs}, ("x", "edge_index")
        )

        logits = _call_model_flexibly(self.mlp, {"x": contextual_embeddings}, ("x",))
        logits = logits.squeeze(-1) if logits.dim() > 1 else logits

        full_logits = torch.full(
            (self.num_nodes,), float("-inf"), dtype=logits.dtype, device=logits.device
        )
        num_assignable = min(len(touched_ids), logits.shape[0])
        full_logits[touched_ids[:num_assignable]] = logits[:num_assignable]
        return full_logits

    # -- epoch loops -----------------------------------------------------

    def _run_epoch(self, node_subset: np.ndarray, train: bool) -> Tuple[float, Dict[str, float]]:
        self.tgn.train(train)
        self.gat.train(train)
        self.mlp.train(train)

        subset_mask = torch.zeros(self.num_nodes, dtype=torch.bool, device=self.device)
        subset_mask[torch.as_tensor(node_subset, dtype=torch.long, device=self.device)] = True
        label_vector = torch.full((self.num_nodes,), float("nan"), device=self.device)
        label_ids = torch.as_tensor(self.node_ids, dtype=torch.long, device=self.device)
        label_vector[label_ids] = self.labels.to(self.device)

        total_loss = 0.0
        num_batches = 0
        all_targets: List[np.ndarray] = []
        all_probs: List[np.ndarray] = []

        shard_iterator = stream_edge_shards(self.config.edge_shard_dir, self.device)
        progress = tqdm(
            shard_iterator,
            desc="train" if train else "val",
            unit="shard",
            leave=False,
        )

        for shard_index, shard in enumerate(progress):
            shard = apply_edge_weighting(shard)

            context = torch.enable_grad() if train else torch.no_grad()
            with context:
                amp_context = (
                    autocast(enabled=self.use_amp)
                    if (self.use_amp and autocast is not None)
                    else _null_context()
                )
                with amp_context:
                    full_logits = self._forward_shard(shard)
                    valid_mask = subset_mask & torch.isfinite(full_logits) & torch.isfinite(label_vector)
                    if valid_mask.sum() == 0:
                        continue
                    batch_logits = full_logits[valid_mask]
                    batch_targets = label_vector[valid_mask]
                    loss = self.loss_fn(batch_logits, batch_targets)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.use_amp and self.scaler is not None:
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(
                            list(self.tgn.parameters())
                            + list(self.gat.parameters())
                            + list(self.mlp.parameters()),
                            self.config.grad_clip_norm,
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        nn.utils.clip_grad_norm_(
                            list(self.tgn.parameters())
                            + list(self.gat.parameters())
                            + list(self.mlp.parameters()),
                            self.config.grad_clip_norm,
                        )
                        self.optimizer.step()

            total_loss += float(loss.detach().item())
            num_batches += 1
            probs = torch.sigmoid(batch_logits.detach()) if _uses_logits(self.config) else batch_logits.detach()
            all_probs.append(probs.cpu().numpy())
            all_targets.append(batch_targets.detach().cpu().numpy())

            if shard_index % self.config.log_every_n_shards == 0:
                progress.set_postfix(loss=f"{loss.item():.4f}")

        if num_batches == 0:
            raise ProductionArtifactError(
                "No shard produced a valid (labeled and finite) batch this "
                "epoch. Check that labeled node ids actually appear in the "
                "edge-feature shards."
            )

        mean_loss = total_loss / num_batches
        targets_arr = np.concatenate(all_targets)
        probs_arr = np.concatenate(all_probs)
        metrics = compute_classification_metrics(targets_arr, probs_arr)
        return mean_loss, metrics

    # -- public entry points -----------------------------------------------

    def resume_if_requested(self, resume: bool) -> None:
        if not resume:
            return
        self.start_epoch, self.best_f1 = self.checkpoint_manager.load_for_resume(
            self.tgn, self.gat, self.mlp, self.optimizer, self.scheduler,
            self.early_stopping, self.device,
        )
        LOGGER.info("Resumed from epoch %d (best F1 so far: %.4f).", self.start_epoch, self.best_f1)

    def fit(self) -> None:
        for epoch in range(self.start_epoch, self.config.epochs):
            epoch_start = time.time()

            train_loss, train_metrics = self._run_epoch(self.train_ids, train=True)
            val_loss, val_metrics = self._run_epoch(self.val_ids, train=False)

            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics["f1"])
            else:
                self.scheduler.step()

            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]["lr"]
            gpu_mem = current_gpu_memory_mb(self.device)

            LOGGER.info(
                "Epoch %d/%d | train_loss=%.4f val_loss=%.4f | "
                "acc=%.4f prec=%.4f rec=%.4f f1=%.4f roc_auc=%.4f pr_auc=%.4f | "
                "lr=%.2e gpu_mem=%.1fMB time=%.1fs",
                epoch + 1, self.config.epochs, train_loss, val_loss,
                val_metrics["accuracy"], val_metrics["precision"],
                val_metrics["recall"], val_metrics["f1"],
                val_metrics["roc_auc"], val_metrics["pr_auc"],
                current_lr, gpu_mem, epoch_time,
            )

            self._log_tensorboard(epoch, train_loss, val_loss, val_metrics, current_lr, gpu_mem)

            is_best = self.early_stopping.step(val_metrics["f1"])
            if is_best:
                self.best_f1 = val_metrics["f1"]

            self.checkpoint_manager.save(
                self.tgn, self.gat, self.mlp, self.optimizer, self.scheduler,
                epoch, self.best_f1, self.early_stopping, is_best,
            )

            if self.early_stopping.should_stop:
                LOGGER.info(
                    "Early stopping triggered after %d epochs with no F1 "
                    "improvement of at least %.6f.",
                    self.config.patience, self.config.min_delta,
                )
                break

        self.writer.close()

    def evaluate_test_set(self) -> Dict[str, float]:
        _, test_metrics = self._run_epoch(self.test_ids, train=False)
        LOGGER.info("Test metrics: %s", test_metrics)
        return test_metrics

    def _log_tensorboard(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_metrics: Dict[str, float],
        lr: float,
        gpu_mem: float,
    ) -> None:
        self.writer.add_scalar("loss/train", train_loss, epoch)
        self.writer.add_scalar("loss/val", val_loss, epoch)
        for name, value in val_metrics.items():
            self.writer.add_scalar(f"val/{name}", value, epoch)
        self.writer.add_scalar("lr", lr, epoch)
        self.writer.add_scalar("system/gpu_memory_mb", gpu_mem, epoch)


# ==========================================================================
# Instantiation / calling helpers for production models with unknown ctors
# ==========================================================================

def _instantiate_flexibly(cls: type, preferred_kwargs: Dict[str, Any]) -> nn.Module:
    """Instantiate `cls`, passing only the keyword arguments its constructor
    actually accepts. Falls back to a no-argument constructor if the
    signature accepts none of the preferred kwargs.
    """
    try:
        sig = inspect.signature(cls.__init__)
        accepted = set(sig.parameters.keys())
        has_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        filtered = {
            k: v for k, v in preferred_kwargs.items() if k in accepted or has_var_kwargs
        }
    except (TypeError, ValueError):
        filtered = {}
    try:
        return cls(**filtered)
    except TypeError as exc:
        try:
            return cls()
        except TypeError:
            raise ProductionArtifactError(
                f"Could not instantiate {cls.__name__} with kwargs "
                f"{filtered} nor with no arguments. Original error: {exc}"
            ) from exc


def _call_model_flexibly(
    model: nn.Module, field_kwargs: Dict[str, Any], positional_order: Sequence[str]
) -> torch.Tensor:
    """Call `model(**field_kwargs)`, falling back to
    `model(**{k: v for k in field_kwargs if k in signature})`, then to
    positional args in `positional_order`.
    """
    try:
        sig = inspect.signature(model.forward)
        accepted = set(sig.parameters.keys())
        has_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if has_var_kwargs:
            return model(**field_kwargs)
        filtered = {k: v for k, v in field_kwargs.items() if k in accepted and v is not None}
        if filtered:
            return model(**filtered)
    except (TypeError, ValueError):
        pass
    positional_args = [field_kwargs[name] for name in positional_order if field_kwargs.get(name) is not None]
    return model(*positional_args)


def _as_long_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.long()
    return torch.as_tensor(value, dtype=torch.long)


def _uses_logits(config: TrainingConfig) -> bool:
    return config.loss.lower() in ("bce_logits", "focal")


class _null_context:
    """A no-op context manager, used when AMP autocast is unavailable/disabled."""

    def __enter__(self) -> "_null_context":
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        return False


# ==========================================================================
# Self-test
# ==========================================================================

def run_self_test(config: TrainingConfig) -> None:
    """Run the full production self-test suite described in the training
    pipeline specification, printing 'TRAINING PIPELINE PASS' only if every
    check succeeds.
    """
    checks: List[Tuple[str, Callable[[], None]]] = []
    state: Dict[str, Any] = {}

    def check_production_artifacts() -> None:
        _require_production_modules()
        if not os.path.exists(config.graph_path):
            raise ProductionArtifactError(f"Missing production graph: {config.graph_path}")
        discover_edge_shards(config.edge_shard_dir)

    def check_labels() -> None:
        state["label_map"] = LabelLoader(config.labels_dir).load()
        if not state["label_map"]:
            raise ProductionArtifactError("Labels file loaded but contained zero entries.")

    def check_tgn_loads() -> None:
        state["trainer"] = InsiderThreatTrainer(config)

    def check_gat_loads() -> None:
        assert isinstance(state["trainer"].gat, nn.Module)

    def check_mlp_loads() -> None:
        assert isinstance(state["trainer"].mlp, nn.Module)

    def check_optimizer_initializes() -> None:
        assert isinstance(state["trainer"].optimizer, optim.Optimizer)

    def check_scheduler_initializes() -> None:
        assert state["trainer"].scheduler is not None

    def check_forward_backward() -> None:
        trainer: InsiderThreatTrainer = state["trainer"]
        shard = next(stream_edge_shards(config.edge_shard_dir, trainer.device))
        shard = apply_edge_weighting(shard)
        full_logits = trainer._forward_shard(shard)  # noqa: SLF001 (self-test only)
        finite_mask = torch.isfinite(full_logits)
        if finite_mask.sum() == 0:
            raise ProductionArtifactError("Forward pass produced no finite logits for any node.")
        dummy_targets = torch.zeros_like(full_logits[finite_mask])
        loss = trainer.loss_fn(full_logits[finite_mask], dummy_targets)
        trainer.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(trainer.tgn.parameters())
            + list(trainer.gat.parameters())
            + list(trainer.mlp.parameters()),
            config.grad_clip_norm,
        )
        trainer.optimizer.step()

    def check_checkpoint_saving() -> None:
        trainer: InsiderThreatTrainer = state["trainer"]
        trainer.checkpoint_manager.save(
            trainer.tgn, trainer.gat, trainer.mlp, trainer.optimizer,
            trainer.scheduler, epoch=0, best_f1=0.0,
            early_stopping=trainer.early_stopping, is_best=True,
        )

    checks = [
        ("production artifacts exist", check_production_artifacts),
        ("labels exist", check_labels),
        ("TGN loads successfully", check_tgn_loads),
        ("GAT loads successfully", check_gat_loads),
        ("MLP loads successfully", check_mlp_loads),
        ("optimizer initializes", check_optimizer_initializes),
        ("scheduler initializes", check_scheduler_initializes),
        ("forward pass succeeds", check_forward_backward),
        ("backward pass succeeds", lambda: None),  # covered inside forward/backward check
        ("checkpoint saving succeeds", check_checkpoint_saving),
    ]

    for description, check in checks:
        LOGGER.info("Running self-test check: %s", description)
        check()
        LOGGER.info("PASSED: %s", description)

    print("TRAINING PIPELINE PASS")


# ==========================================================================
# CLI entry point
# ==========================================================================

def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Insider Threat Detection TGN -> GAT -> MLP pipeline."
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a JSON config previously written by TrainingConfig.save().",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from the last checkpoint in checkpoint_dir.",
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="Run the production self-test suite and exit.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    _configure_logging()
    args = parse_args(argv)

    config = TrainingConfig.load(args.config) if args.config else TrainingConfig()

    if args.self_test:
        run_self_test(config)
        return

    _require_production_modules()
    trainer = InsiderThreatTrainer(config)
    trainer.resume_if_requested(args.resume)
    trainer.fit()
    trainer.evaluate_test_set()


if __name__ == "__main__":
    main()