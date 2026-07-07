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
import csv
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
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - optional runtime dependency
    SummaryWriter = None  # type: ignore

_THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = (
    _THIS_FILE_DIR
    if os.path.isdir(os.path.join(_THIS_FILE_DIR, "training"))
    else os.path.dirname(_THIS_FILE_DIR)
)
sys.path.insert(0, _PROJECT_ROOT)

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
_CANDIDATE_EDGE_WEIGHT = ("weight", "compute_edge_weights", "apply_edge_weights")


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

    Searches recursively in shard_dir to find shards in subdirectories
    (created by edge_features.py per relation).

    Raises:
        ProductionArtifactError: if the directory or shards do not exist.
    """
    if not os.path.isdir(shard_dir):
        raise ProductionArtifactError(
            f"Edge feature shard directory not found: '{shard_dir}'."
        )
    # Search recursively for edge_features_*.pt files
    pattern = os.path.join(shard_dir, "**", "edge_features_*.pt")
    shards = sorted(glob.glob(pattern, recursive=True))
    # Debug logging
    if not shards:
        LOGGER.error("discover_edge_shards - pattern: %s", pattern)
        LOGGER.error("discover_edge_shards - dir exists: %s", os.path.isdir(shard_dir))
        LOGGER.error("discover_edge_shards - glob result: %s", shards)
        raise ProductionArtifactError(
            f"No shards matching 'edge_features_*.pt' were found in "
            f"'{shard_dir}'. Run graph/edge_features.py to produce them."
        )
    LOGGER.info("discover_edge_shards - found %d shards", len(shards))
    return shards


_NODE_OFFSETS: Optional[Dict[str, int]] = None

def _get_node_offsets() -> Dict[str, int]:
    global _NODE_OFFSETS
    if _NODE_OFFSETS is not None:
        return _NODE_OFFSETS
    
    # Resolve skeleton path relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    skeleton_path = os.path.join(project_root, "graph", "output", "node_graph_skeleton.pt")
    if not os.path.exists(skeleton_path):
        skeleton_path = "graph/output/node_graph_skeleton.pt"
        
    try:
        skeleton = torch.load(skeleton_path, map_location="cpu", weights_only=False)
        offsets = {}
        offset = 0
        for node_type in skeleton.node_types:
            offsets[node_type] = offset
            offset += skeleton[node_type].num_nodes
        _NODE_OFFSETS = offsets
        LOGGER.info("Computed cumulative node type offsets from skeleton: %s", _NODE_OFFSETS)
    except Exception as e:
        LOGGER.warning("Could not compute node type offsets: %s. Defaulting to zero offsets.", e)
        _NODE_OFFSETS = {}
    return _NODE_OFFSETS


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
    """Move tensor fields of a shard (dict or object) onto `device`, and translate
    local node indices in `edge_index` to global unique node IDs.
    """
    offsets = _get_node_offsets()
    
    src_type = _shard_field(shard, "src_node_type", "src_type") or "User"
    dst_type = _shard_field(shard, "dst_node_type", "dst_type")
    
    src_offset = offsets.get(src_type, 0)
    dst_offset = offsets.get(dst_type, 0) if dst_type is not None else 0

    if isinstance(shard, dict):
        shard = dict(shard)
        for key, value in shard.items():
            if isinstance(value, torch.Tensor):
                shard[key] = value.to(device)
                
        if "edge_index" in shard and isinstance(shard["edge_index"], torch.Tensor):
            edge_index = shard["edge_index"].clone()
            edge_index[0] += src_offset
            edge_index[1] += dst_offset
            shard["edge_index"] = edge_index
            
        return shard

    if hasattr(shard, "__dict__"):
        for attr_name, attr_value in list(vars(shard).items()):
            if isinstance(attr_value, torch.Tensor):
                setattr(shard, attr_name, attr_value.to(device))
                
        edge_index = getattr(shard, "edge_index", None)
        if isinstance(edge_index, torch.Tensor):
            edge_index = edge_index.clone()
            edge_index[0] += src_offset
            edge_index[1] += dst_offset
            setattr(shard, "edge_index", edge_index)
            
    return shard


def apply_edge_weighting(shard: Any, weighter: Optional[nn.Module] = None) -> Any:
    """Apply dynamic edge weighting to a shard if the module exposes an entry
    point for it. This step is optional and never fabricates weights.
    """
    if weighter is not None:
        features = _shard_field(shard, "features")
        if features is not None:
            edge_weight = weighter(features)
            # Ensure it is squeezed to shape [num_edges] if it's [num_edges, 1]
            if edge_weight.dim() > 1 and edge_weight.size(-1) == 1:
                edge_weight = edge_weight.squeeze(-1)
            if isinstance(shard, dict):
                shard = dict(shard)
                shard["edge_weight"] = edge_weight
            else:
                setattr(shard, "edge_weight", edge_weight)
        return shard

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
        self.alpha = min(max(float(alpha), 0.05), 0.95)
        self.gamma = gamma
        self._bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self._bce(logits, targets)
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        modulating_factor = (1.0 - p_t).clamp(min=1e-8) ** self.gamma
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_factor * modulating_factor * bce_loss).mean()


def build_loss_fn(config: TrainingConfig, pos_weight: Optional[torch.Tensor] = None) -> nn.Module:
    """Instantiate the configured loss function."""
    loss_name = config.loss.lower()
    if loss_name == "bce":
        return nn.BCELoss()
    if loss_name == "bce_logits":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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

def _rate_capped_threshold(
    probs: np.ndarray, max_pred_positive_rate: float, min_threshold: float
) -> float:
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    if probs.size == 0:
        return 0.5
    if float(probs.max() - probs.min()) < 1e-6:
        return max(0.5, min_threshold)
    quantile = 1.0 - min(max_pred_positive_rate, 1.0)
    return float(max(min_threshold, np.quantile(probs, quantile)))


def find_best_threshold(
    targets: np.ndarray,
    probs: np.ndarray,
    min_threshold: float = 0.05,
    max_pred_positive_rate: float = 0.20,
    min_recall: float = 0.80,
    min_positive_count: int = 3,
) -> float:
    """Find a constrained threshold without overfitting to one positive.

    Plain F1 maximization on CERT r4.2 validation can select thresholds near
    zero because the validation split has only one positive user. This search
    prefers high recall, but rejects thresholds that classify almost every
    validation node as positive.
    """
    if probs.size == 0 or targets.size == 0:
        return 0.5
    targets = np.asarray(targets, dtype=np.float32).reshape(-1)
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    if np.unique(targets.astype(int)).size < 2:
        return 0.5
    n_pos = int((targets == 1).sum())
    if n_pos < min_positive_count:
        threshold = _rate_capped_threshold(probs, max_pred_positive_rate, min_threshold)
        LOGGER.warning(
            "Validation has only %d positive sample(s); using rate-capped "
            "threshold %.4f instead of unconstrained F1 tuning.",
            n_pos, threshold,
        )
        return threshold

    unique_probs = np.unique(probs[np.isfinite(probs)])
    thresholds = np.unique(np.concatenate([
        np.linspace(min_threshold, 0.99, 95, dtype=np.float32),
        unique_probs,
    ]))
    thresholds = thresholds[(thresholds >= min_threshold) & (thresholds <= 0.99)]
    best_thresh = _rate_capped_threshold(probs, max_pred_positive_rate, min_threshold)
    best_score = (-1.0, -1.0, -1.0)
    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        pred_rate = float(preds.mean())
        if pred_rate > max_pred_positive_rate:
            continue
        scores = _binary_counts_and_scores(targets, preds)
        recall_ok = 1.0 if scores["recall"] >= min_recall else 0.0
        score = (recall_ok, scores["f1"], scores["precision"])
        if score > best_score:
            best_score = score
            best_thresh = float(thresh)
    return float(best_thresh)


def _binary_counts_and_scores(targets: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    targets = targets.astype(int).reshape(-1)
    preds = preds.astype(int).reshape(-1)
    tn = int(((targets == 0) & (preds == 0)).sum())
    fp = int(((targets == 0) & (preds == 1)).sum())
    fn = int(((targets == 1) & (preds == 0)).sum())
    tp = int(((targets == 1) & (preds == 1)).sum())
    total = max(int(targets.size), 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def _roc_auc_np(targets: np.ndarray, probs: np.ndarray) -> float:
    targets = targets.astype(int).reshape(-1)
    probs = probs.reshape(-1)
    pos = targets == 1
    neg = targets == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0
    order = np.argsort(probs, kind="mergesort")
    sorted_scores = probs[order]
    ranks = np.empty_like(sorted_scores, dtype=np.float64)
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = (start + end + 1) / 2.0
        start = end
    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    sum_pos_ranks = float(original_ranks[pos].sum())
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _average_precision_np(targets: np.ndarray, probs: np.ndarray) -> float:
    targets = targets.astype(int).reshape(-1)
    probs = probs.reshape(-1)
    n_pos = int((targets == 1).sum())
    if n_pos == 0:
        return 0.0
    order = np.argsort(-probs, kind="mergesort")
    sorted_targets = targets[order]
    tp_cumsum = np.cumsum(sorted_targets == 1)
    precision_at_k = tp_cumsum / (np.arange(sorted_targets.size) + 1)
    return float((precision_at_k * (sorted_targets == 1)).sum() / n_pos)


def compute_classification_metrics(
    targets: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
    logits: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute accuracy/precision/recall/F1/ROC-AUC/PR-AUC, confusion matrix,
    and probability bounds, handling single-class cases gracefully.
    """
    targets = np.asarray(targets, dtype=np.float32).reshape(-1)
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    if targets.size != probs.size:
        raise ProductionArtifactError(
            f"Metric input mismatch: {targets.size} targets for {probs.size} probabilities."
        )
    if targets.size == 0:
        raise ProductionArtifactError("Metric input is empty; no labeled samples were evaluated.")
    if not np.isfinite(probs).all():
        raise ProductionArtifactError("Predicted probabilities contain NaN or Inf.")
    if not np.isin(targets, [0.0, 1.0]).all():
        raise ProductionArtifactError("Metric targets contain labels outside {0, 1}.")

    preds = (probs >= threshold).astype(int)
    metrics = _binary_counts_and_scores(targets, preds)
    metrics.update({
        "prob_min": float(probs.min()) if probs.size > 0 else 0.0,
        "prob_max": float(probs.max()) if probs.size > 0 else 0.0,
        "prob_mean": float(probs.mean()) if probs.size > 0 else 0.0,
        "prob_std": float(probs.std()) if probs.size > 0 else 0.0,
        "prob_saturated_low": float((probs <= 1e-4).mean()) if probs.size > 0 else 0.0,
        "prob_saturated_high": float((probs >= 1.0 - 1e-4).mean()) if probs.size > 0 else 0.0,
        "pred_neg": float((preds == 0).sum()),
        "pred_pos": float((preds == 1).sum()),
    })
    if logits is not None:
        logits = np.asarray(logits, dtype=np.float32).reshape(-1)
        finite_logits = logits[np.isfinite(logits)]
        if finite_logits.size > 0:
            metrics.update({
                "logit_min": float(finite_logits.min()),
                "logit_max": float(finite_logits.max()),
                "logit_mean": float(finite_logits.mean()),
                "logit_std": float(finite_logits.std()),
                "logit_saturated_abs_gt_10": float((np.abs(finite_logits) > 10.0).mean()),
            })
    if metrics["pred_pos"] == 0 or metrics["pred_neg"] == 0:
        LOGGER.warning(
            "Prediction collapse at threshold %.4f: pred_neg=%d pred_pos=%d prob_std=%.6e.",
            threshold, int(metrics["pred_neg"]), int(metrics["pred_pos"]), metrics["prob_std"],
        )

    if len(np.unique(targets)) > 1:
        metrics["roc_auc"] = float(_roc_auc_np(targets, probs))
        metrics["pr_auc"] = float(_average_precision_np(targets, probs))
    else:
        metrics["roc_auc"] = 0.0
        metrics["pr_auc"] = 0.0

    LOGGER.info(
        "Diagnostics | Probabilities: min=%.6f, max=%.6f, mean=%.6f | "
        "Pred histogram: neg=%d pos=%d | Confusion Matrix: TN=%d, FP=%d, FN=%d, TP=%d",
        metrics["prob_min"], metrics["prob_max"], metrics["prob_mean"],
        int(metrics["pred_neg"]), int(metrics["pred_pos"]),
        int(metrics["tn"]), int(metrics["fp"]), int(metrics["fn"]), int(metrics["tp"])
    )
    for name, value in metrics.items():
        if isinstance(value, float) and not np.isfinite(value):
            raise ProductionArtifactError(f"Metric {name} became non-finite: {value}")
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
        edge_weighter: Optional[nn.Module] = None,
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
        if edge_weighter is not None:
            model_state["edge_weighter"] = edge_weighter.state_dict()
            
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
        edge_weighter: Optional[nn.Module] = None,
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
        if edge_weighter is not None and "edge_weighter" in model_state:
            edge_weighter.load_state_dict(model_state["edge_weighter"])
            LOGGER.info("Resumed learnable DynamicEdgeWeighting weights from checkpoint.")

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
        self._audit_labels(label_map)
        LOGGER.info("Loaded %d node-level labels.", len(self.node_ids))

        self.train_ids, self.val_ids, self.test_ids = self._split_nodes(self.node_ids)
        self._audit_split("train", self.train_ids)
        self._audit_split("validation", self.val_ids)
        self._audit_split("test", self.test_ids, require_both_classes=False)
        LOGGER.info(
            "Split sizes -- train: %d, val: %d, test: %d",
            len(self.train_ids), len(self.val_ids), len(self.test_ids),
        )

        self.tgn = self._build_tgn().to(self.device)
        self.gat = self._build_gat().to(self.device)
        self.mlp = self._build_mlp().to(self.device)

        # Initialize learnable DynamicEdgeWeighting model once
        self.edge_weighter = None
        if edge_weighting_module is not None:
            try:
                from graph.edge_weighting import DynamicEdgeWeighting
                self.edge_weighter = DynamicEdgeWeighting(in_features=40).to(self.device)
                LOGGER.info("Initialized learnable DynamicEdgeWeighting model once in trainer.")
            except Exception as e:
                LOGGER.warning("Could not initialize learnable DynamicEdgeWeighting model: %s", e)

        # Calculate pos_weight for class imbalance
        train_labels = self.labels[torch.as_tensor(self.train_ids, dtype=torch.long)]
        num_neg = (train_labels == 0).sum().item()
        num_pos = (train_labels == 1).sum().item()
        if num_pos > 0:
            raw_pos_weight_val = num_neg / num_pos
            pos_weight_val = min(float(raw_pos_weight_val), float(config.max_pos_weight))
            self.pos_weight = torch.tensor([pos_weight_val], device=self.device)
            self.raw_pos_weight = float(raw_pos_weight_val)
            LOGGER.info(
                "Stratified train labels: normal=%d, threat=%d. Raw pos_weight=%.4f, "
                "capped pos_weight=%.4f.",
                num_neg, num_pos, raw_pos_weight_val, pos_weight_val,
            )
            self.class_ratio = float(raw_pos_weight_val)
        else:
            self.pos_weight = None
            self.raw_pos_weight = float("inf")
            self.class_ratio = float("inf")
            LOGGER.warning("No positive samples in the train split!")

        # Dynamic validation of class balance in splits
        val_labels = self.labels[torch.as_tensor(self.val_ids, dtype=torch.long)]
        val_num_pos = (val_labels == 1).sum().item()
        val_num_neg = (val_labels == 0).sum().item()
        # Assertions on label class counts
        assert num_pos > 0 and num_neg > 0, f"Train split must have both classes, got pos={num_pos}, neg={num_neg}"
        assert val_num_pos > 0 and val_num_neg > 0, f"Val split must have both classes, got pos={val_num_pos}, neg={val_num_neg}"

        all_params = (
            list(self.tgn.parameters())
            + list(self.gat.parameters())
            + list(self.mlp.parameters())
        )
        if self.edge_weighter is not None:
            all_params += list(self.edge_weighter.parameters())

        self.optimizer = build_optimizer(config, iter(all_params))
        self.scheduler = build_scheduler(config, self.optimizer)
        if config.loss.lower() == "focal":
            LOGGER.info(
                "Using focal loss with alpha=%.4f gamma=%.4f. Pos_weight is not "
                "converted into focal alpha to avoid suppressing negative samples.",
                config.focal_alpha, config.focal_gamma,
            )
        self.loss_fn = build_loss_fn(config, pos_weight=self.pos_weight).to(self.device)
        self.probability_logit_shift = 0.0
        if (
            config.calibrate_weighted_logits
            and config.loss.lower() == "bce_logits"
            and self.pos_weight is not None
            and float(self.pos_weight.item()) > 1.0
        ):
            self.probability_logit_shift = float(np.log(float(self.pos_weight.item())))
            LOGGER.info(
                "Calibrating weighted-BCE probabilities with logit shift %.6f.",
                self.probability_logit_shift,
            )

        self.use_amp = self.device.type == "cuda" and GradScaler is not None
        self.scaler = GradScaler(enabled=self.use_amp) if GradScaler is not None else None
        LOGGER.info("Mixed precision (AMP) enabled: %s", self.use_amp)

        self.early_stopping = EarlyStopping(config.patience, config.min_delta)
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir)
        self.writer = SummaryWriter(log_dir=config.log_dir) if SummaryWriter is not None else None
        if self.writer is None:
            LOGGER.warning("TensorBoard is not installed; continuing with CSV and console metrics.")
        os.makedirs(config.log_dir, exist_ok=True)
        self.csv_log_path = os.path.join(config.log_dir, "metrics.csv")
        self._init_csv_log()

        self.start_epoch = 0
        self.best_f1 = float("-inf")
        self.best_threshold = 0.5

    def _logits_to_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        calibrated_logits = logits
        if self.probability_logit_shift:
            calibrated_logits = calibrated_logits - self.probability_logit_shift
        return torch.sigmoid(calibrated_logits).clamp(1e-6, 1.0 - 1e-6)

    # -- construction helpers ---------------------------------------------

    def _prepare_labels(self, label_map: Dict[int, float]) -> Tuple[np.ndarray, torch.Tensor]:
        node_ids = np.array(sorted(label_map.keys()), dtype=np.int64)
        if node_ids.size == 0:
            raise ProductionArtifactError("Loaded labels are empty; cannot train.")
        if len(set(node_ids.tolist())) != len(node_ids):
            raise ProductionArtifactError("Duplicate node ids detected in label map.")
        out_of_range = node_ids[(node_ids < 0) | (node_ids >= self.num_nodes)]
        if out_of_range.size > 0:
            raise ProductionArtifactError(
                f"{out_of_range.size} labeled node id(s) fall outside the "
                f"graph's node range [0, {self.num_nodes}). Example offending "
                f"ids: {out_of_range[:5].tolist()}."
            )
        values = np.array([label_map[int(n)] for n in node_ids], dtype=np.float32)
        if np.isnan(values).any():
            raise ProductionArtifactError("Missing/NaN labels detected.")
        invalid = values[~np.isin(values, [0.0, 1.0])]
        if invalid.size > 0:
            raise ProductionArtifactError(
                f"Invalid binary labels detected. Expected only 0/1, examples: {invalid[:5].tolist()}."
            )
        unique, counts = np.unique(values.astype(int), return_counts=True)
        class_counts = {int(k): int(v) for k, v in zip(unique, counts)}
        if len(class_counts) < 2:
            raise ProductionArtifactError(
                f"Only one class exists in labels: {class_counts}. Training is invalid."
            )
        full_labels = torch.full((self.num_nodes,), float("nan"), dtype=torch.float32)
        full_labels[torch.as_tensor(node_ids, dtype=torch.long)] = torch.as_tensor(values)
        return node_ids, full_labels

    def _audit_labels(self, label_map: Dict[int, float]) -> None:
        values = np.array([label_map[int(n)] for n in self.node_ids], dtype=np.float32)
        neg = int((values == 0).sum())
        pos = int((values == 1).sum())
        total = int(values.size)
        duplicates = total - len(set(int(n) for n in self.node_ids.tolist()))
        missing = int(np.isnan(values).sum())
        invalid = int((~np.isin(values, [0.0, 1.0])).sum())
        minority = min(pos, neg) / total * 100.0
        majority = max(pos, neg) / total * 100.0
        ratio = (max(pos, neg) / max(min(pos, neg), 1)) if min(pos, neg) else float("inf")
        LOGGER.info(
            "DATA AUDIT | labels total=%d normal=%d threat=%d missing=%d invalid=%d duplicates=%d "
            "minority=%.4f%% majority=%.4f%% class_ratio=%.2f:1",
            total, neg, pos, missing, invalid, duplicates, minority, majority, ratio,
        )
        positive_ids = self.node_ids[values == 1].tolist()
        LOGGER.info("DATA AUDIT | positive node ids=%s", positive_ids)

    def _audit_split(self, name: str, node_ids: np.ndarray, require_both_classes: bool = True) -> None:
        ids = torch.as_tensor(node_ids, dtype=torch.long)
        split_labels = self.labels[ids]
        neg = int((split_labels == 0).sum().item())
        pos = int((split_labels == 1).sum().item())
        missing = int(torch.isnan(split_labels).sum().item())
        LOGGER.info(
            "DATA AUDIT | %s split total=%d normal=%d threat=%d missing=%d",
            name, len(node_ids), neg, pos, missing,
        )
        if missing:
            raise ProductionArtifactError(f"{name} split contains {missing} missing labels.")
        if require_both_classes and (neg == 0 or pos == 0):
            raise ProductionArtifactError(
                f"{name} split must contain both classes, got normal={neg}, threat={pos}."
            )
        if not require_both_classes and (neg == 0 or pos == 0):
            LOGGER.warning(
                "%s split has a single class (normal=%d, threat=%d). ROC/PR will be reported as 0.0, not NaN.",
                name, neg, pos,
            )

    def _split_nodes(
        self, node_ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Perform class-stratified splitting
        labels_np = self.labels[torch.as_tensor(node_ids, dtype=torch.long)].cpu().numpy()
        pos_mask = (labels_np == 1)
        pos_ids = node_ids[pos_mask]
        neg_ids = node_ids[~pos_mask]

        rng = np.random.RandomState(self.config.seed)
        shuffled_pos = pos_ids.copy()
        rng.shuffle(shuffled_pos)
        shuffled_neg = neg_ids.copy()
        rng.shuffle(shuffled_neg)

        # Distribute positives. Since we have exactly 2 positive samples in CMU CERT r4.2:
        # Put 1 in train, 1 in val, 0 in test.
        if len(shuffled_pos) >= 2:
            train_pos = shuffled_pos[:1]
            val_pos = shuffled_pos[1:2]
            test_pos = shuffled_pos[2:]
        else:
            n_train_pos = int(round(len(shuffled_pos) * self.config.train_split))
            n_val_pos = int(round(len(shuffled_pos) * self.config.val_split))
            train_pos = shuffled_pos[:n_train_pos]
            val_pos = shuffled_pos[n_train_pos:n_train_pos + n_val_pos]
            test_pos = shuffled_pos[n_train_pos + n_val_pos:]

        n_neg = len(shuffled_neg)
        n_train_neg = int(round(n_neg * self.config.train_split))
        n_val_neg = int(round(n_neg * self.config.val_split))

        train_neg = shuffled_neg[:n_train_neg]
        val_neg = shuffled_neg[n_train_neg:n_train_neg + n_val_neg]
        test_neg = shuffled_neg[n_train_neg + n_val_neg:]

        train_ids = np.concatenate([train_pos, train_neg])
        val_ids = np.concatenate([val_pos, val_neg])
        test_ids = np.concatenate([test_pos, test_neg])

        # Shuffle again so they are mixed
        rng.shuffle(train_ids)
        rng.shuffle(val_ids)
        rng.shuffle(test_ids)

        return train_ids, val_ids, test_ids

    def _build_tgn(self) -> nn.Module:
        if TGN is None:
            raise ProductionArtifactError("models.tgn_model.TGN failed to import.")
        return _instantiate_flexibly(
            TGN,
            preferred_kwargs={
                "num_nodes": self.num_nodes,
                "edge_feature_dim": 56,  # 40 scalar features + 16 temporal encoding
                "embedding_dim": self.config.tgn_embedding_dim,
                "memory_dim": self.config.tgn_embedding_dim,
                "device": self.device,
            },
        )

    def _build_gat(self) -> nn.Module:
        if GAT is None:
            raise ProductionArtifactError("models.gat_model.GAT failed to import.")
        
        from models.gat_model import GATConfig
        gat_config = GATConfig(
            in_dim=self.config.tgn_embedding_dim,
            hidden_dim=self.config.gat_embedding_dim,
            out_dim=self.config.gat_embedding_dim,
            heads=4,
            dropout=0.2,
        )
        return GAT(gat_config)

    def _build_mlp(self) -> nn.Module:
        if MLPClassifier is None:
            raise ProductionArtifactError("models.mlp_classifier.MLPClassifier failed to import.")
        
        from models.mlp_classifier import MLPClassifierConfig
        mlp_config = MLPClassifierConfig()
        return MLPClassifier(in_dim=self.config.gat_embedding_dim, config=mlp_config)

    # -- forward pass --------------------------------------------------------

    def _forward_shard(self, shard: Any) -> torch.Tensor:
        """Run TGN -> GAT -> MLP for a single edge-feature shard and return
        node-embedding-derived logits for every node touched by the shard,
        indexed by node id via a full-size [num_nodes] tensor filled with
        -inf for untouched nodes (so downstream code can mask them out).
        """
        edge_index = _shard_field(shard, "edge_index")
        features = _shard_field(shard, "features")
        temporal_encoding = _shard_field(shard, "temporal_encoding")
        edge_time = _shard_field(shard, "edge_time")
        edge_weight = _shard_field(shard, "edge_weight", "weight")

        if edge_index is None:
            src = _shard_field(shard, "src", "source", "src_ids")
            dst = _shard_field(shard, "dst", "target", "dst_ids")
            if src is None or dst is None:
                raise ProductionArtifactError(
                    "Edge shard is missing the production edge-index fields required to run "
                    "the TGN. Check graph/edge_features.py's shard schema."
                )
            edge_index = torch.stack([_as_long_tensor(src), _as_long_tensor(dst)], dim=0)

        if features is None or temporal_encoding is None or edge_time is None:
            raise ProductionArtifactError(
                "Edge shard is missing feature tensors required to run the TGN."
            )

        tgn_shard: Dict[str, Any] = {
            "edge_index": _as_long_tensor(edge_index),
            "features": features,
            "temporal_encoding": temporal_encoding,
            "edge_time": edge_time,
        }
        if edge_weight is not None:
            tgn_shard["edge_weight"] = edge_weight

        if hasattr(self.tgn, "process_shard"):
            tgn_result = self.tgn.process_shard(tgn_shard)
            node_embeddings = tgn_result["embeddings"]
            touched_ids = tgn_result["updated_node_ids"]
        else:
            tgn_batch = {"src": edge_index[0], "dst": edge_index[1], "t": edge_time, "msg": features}
            if edge_weight is not None:
                tgn_batch["edge_weight"] = edge_weight
            node_embeddings = _call_model_flexibly(self.tgn, tgn_batch, ("src", "dst", "t", "msg"))
            if isinstance(node_embeddings, tuple):
                node_embeddings = torch.cat(node_embeddings, dim=0)
            touched_ids = torch.unique(torch.cat([edge_index[0], edge_index[1]]))

        # Stability check: verify that embeddings are not constant
        if node_embeddings.numel() > 0:
            var_val = node_embeddings.var().item()
            if hasattr(self, "_epoch_embedding_variances"):
                self._epoch_embedding_variances.append(float(var_val))
            assert var_val > 1e-6 or node_embeddings.size(0) <= 1, f"Node embeddings collapsed to constant! var={var_val:.6e}"
        if hasattr(self.tgn, "memory") and hasattr(self.tgn.memory, "memory"):
            memory_tensor = self.tgn.memory.memory.detach()
            if memory_tensor.numel() > 1 and hasattr(self, "_epoch_memory_variances"):
                self._epoch_memory_variances.append(float(memory_tensor.float().var().item()))

        local_edge_index = torch.stack(
            [
                torch.searchsorted(touched_ids, edge_index[0]),
                torch.searchsorted(touched_ids, edge_index[1]),
            ],
            dim=0,
        )

        gat_kwargs: Dict[str, Any] = {"x": node_embeddings, "edge_index": local_edge_index}
        if edge_weight is not None:
            gat_kwargs["edge_weight"] = edge_weight
        contextual_result = _call_model_flexibly(
            self.gat, gat_kwargs, ("x", "edge_index")
        )
        if isinstance(contextual_result, tuple):
            contextual_embeddings = contextual_result[0]
            if len(contextual_result) > 1 and torch.is_tensor(contextual_result[1]):
                attention = contextual_result[1].detach().float().clamp_min(1e-12)
                entropy = -(attention * attention.log()).sum(dim=0).mean().item() if attention.dim() > 1 else -(attention * attention.log()).mean().item()
                if hasattr(self, "_epoch_attention_entropies"):
                    self._epoch_attention_entropies.append(float(entropy))
        else:
            contextual_embeddings = contextual_result

        logits = _call_model_flexibly(self.mlp, {"x": contextual_embeddings}, ("x",))
        logits = logits.squeeze(-1) if logits.dim() > 1 else logits
        if logits.numel() > 1 and hasattr(self, "_epoch_logit_variances"):
            self._epoch_logit_variances.append(float(logits.detach().float().var().item()))

        # Stability check: logits must not be NaN/inf
        assert not torch.isnan(logits).any(), "MLP logits contain NaN values"
        assert not torch.isinf(logits).any(), "MLP logits contain Inf values"

        full_logits = torch.full(
            (self.num_nodes,), float("-inf"), dtype=logits.dtype, device=logits.device
        )
        full_logits[touched_ids] = logits
        return full_logits

    # -- epoch loops -----------------------------------------------------

    def _run_epoch(self, node_subset: np.ndarray, train: bool) -> Tuple[float, Dict[str, float], Tuple[np.ndarray, np.ndarray]]:
        self.tgn.train(train)
        self.gat.train(train)
        self.mlp.train(train)
        if self.edge_weighter is not None:
            self.edge_weighter.train(train)

        # Reset TGN memory at the beginning of each epoch pass
        if hasattr(self.tgn, "memory") and hasattr(self.tgn.memory, "reset"):
            self.tgn.memory.reset()
            LOGGER.info("Reset TGN persistent node memory buffer.")

        subset_mask = torch.zeros(self.num_nodes, dtype=torch.bool, device=self.device)
        subset_mask[torch.as_tensor(node_subset, dtype=torch.long, device=self.device)] = True
        label_vector = self.labels.to(self.device)

        total_loss = 0.0
        num_batches = 0
        all_targets: List[np.ndarray] = []
        all_probs: List[np.ndarray] = []
        all_logits: List[np.ndarray] = []
        total_grad_norm = 0.0
        total_mlp_grad_norm = 0.0
        grad_steps = 0
        self._epoch_embedding_variances: List[float] = []
        self._epoch_memory_variances: List[float] = []
        self._epoch_attention_entropies: List[float] = []
        self._epoch_logit_variances: List[float] = []

        node_prob_sum = torch.zeros((self.num_nodes,), device=self.device)
        node_logit_sum = torch.zeros((self.num_nodes,), device=self.device)
        node_prob_count = torch.zeros((self.num_nodes,), device=self.device)
        node_targets = torch.full((self.num_nodes,), -1.0, device=self.device)

        shard_iterator = stream_edge_shards(self.config.edge_shard_dir, self.device)
        progress = tqdm(
            shard_iterator,
            desc="train" if train else "val",
            unit="shard",
            leave=False,
        )

        params_to_clip = (
            list(self.tgn.parameters())
            + list(self.gat.parameters())
            + list(self.mlp.parameters())
        )
        if self.edge_weighter is not None:
            params_to_clip += list(self.edge_weighter.parameters())

        for shard_index, shard in enumerate(progress):
            shard = apply_edge_weighting(shard, self.edge_weighter)

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
                        
                        # Calculate and verify gradient norm before clipping
                        grad_norm = 0.0
                        mlp_grad_norm = 0.0
                        for p in params_to_clip:
                            if p.grad is not None:
                                grad_norm += p.grad.data.norm(2).item() ** 2
                        for p in self.mlp.parameters():
                            if p.grad is not None:
                                mlp_grad_norm += p.grad.data.norm(2).item() ** 2
                        grad_norm = grad_norm ** 0.5
                        mlp_grad_norm = mlp_grad_norm ** 0.5
                        total_grad_norm += grad_norm
                        total_mlp_grad_norm += mlp_grad_norm
                        grad_steps += 1
                        assert grad_norm > 1e-8, f"Gradients collapsed to zero unexpectedly! grad_norm={grad_norm:.6e}"
                        assert mlp_grad_norm > 1e-10, f"MLP gradients collapsed to zero unexpectedly! mlp_grad_norm={mlp_grad_norm:.6e}"

                        nn.utils.clip_grad_norm_(
                            params_to_clip,
                            self.config.grad_clip_norm,
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()

                        # Calculate and verify gradient norm before clipping
                        grad_norm = 0.0
                        mlp_grad_norm = 0.0
                        for p in params_to_clip:
                            if p.grad is not None:
                                grad_norm += p.grad.data.norm(2).item() ** 2
                        for p in self.mlp.parameters():
                            if p.grad is not None:
                                mlp_grad_norm += p.grad.data.norm(2).item() ** 2
                        grad_norm = grad_norm ** 0.5
                        mlp_grad_norm = mlp_grad_norm ** 0.5
                        total_grad_norm += grad_norm
                        total_mlp_grad_norm += mlp_grad_norm
                        grad_steps += 1
                        assert grad_norm > 1e-8, f"Gradients collapsed to zero unexpectedly! grad_norm={grad_norm:.6e}"
                        assert mlp_grad_norm > 1e-10, f"MLP gradients collapsed to zero unexpectedly! mlp_grad_norm={mlp_grad_norm:.6e}"

                        nn.utils.clip_grad_norm_(
                            params_to_clip,
                            self.config.grad_clip_norm,
                        )
                        self.optimizer.step()

            total_loss += float(loss.detach().item())
            num_batches += 1
            
            # Extract active indices & update node-level predictions
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]
            logits_for_metrics = batch_logits.detach()
            probs = self._logits_to_probabilities(logits_for_metrics) if _uses_logits(self.config) else logits_for_metrics
            
            node_prob_sum[valid_indices] += probs
            node_logit_sum[valid_indices] += logits_for_metrics
            node_prob_count[valid_indices] += 1.0
            node_targets[valid_indices] = label_vector[valid_indices]

            all_probs.append(probs.cpu().numpy())
            all_logits.append(logits_for_metrics.cpu().numpy())
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
        logits_arr = np.concatenate(all_logits)
        metrics = compute_classification_metrics(
            targets_arr, probs_arr, threshold=self.best_threshold, logits=logits_arr
        )
        metrics["grad_norm_mean"] = float(total_grad_norm / max(grad_steps, 1)) if train else 0.0
        metrics["mlp_grad_norm_mean"] = float(total_mlp_grad_norm / max(grad_steps, 1)) if train else 0.0
        metrics["embedding_var_mean"] = float(np.mean(self._epoch_embedding_variances)) if self._epoch_embedding_variances else 0.0
        metrics["memory_var_mean"] = float(np.mean(self._epoch_memory_variances)) if self._epoch_memory_variances else 0.0
        metrics["attention_entropy_mean"] = float(np.mean(self._epoch_attention_entropies)) if self._epoch_attention_entropies else 0.0
        metrics["logit_var_mean"] = float(np.mean(self._epoch_logit_variances)) if self._epoch_logit_variances else 0.0
        LOGGER.info(
            "%s model health | grad_norm=%.6f mlp_grad_norm=%.6f embedding_var=%.6e memory_var=%.6e "
            "attention_entropy=%.6f logit_var=%.6e",
            "train" if train else "eval",
            metrics["grad_norm_mean"], metrics["mlp_grad_norm_mean"], metrics["embedding_var_mean"],
            metrics["memory_var_mean"], metrics["attention_entropy_mean"],
            metrics["logit_var_mean"],
        )
        
        # Gather node-level metrics
        eval_mask = (node_targets != -1) & (node_prob_count > 0)
        node_targets_np = node_targets[eval_mask].cpu().numpy()
        node_probs_np = (node_prob_sum[eval_mask] / node_prob_count[eval_mask]).cpu().numpy()

        return mean_loss, metrics, (node_targets_np, node_probs_np)

    # -- public entry points -----------------------------------------------

    def resume_if_requested(self, resume: bool) -> None:
        if not resume:
            return
        self.start_epoch, self.best_f1 = self.checkpoint_manager.load_for_resume(
            self.tgn, self.gat, self.mlp, self.optimizer, self.scheduler,
            self.early_stopping, self.device, self.edge_weighter
        )
        LOGGER.info("Resumed from epoch %d (best F1 so far: %.4f).", self.start_epoch, self.best_f1)

    def fit(self) -> None:
        for epoch in range(self.start_epoch, self.config.epochs):
            epoch_start = time.time()

            params_before = self._parameter_vector()
            train_loss, train_metrics, _ = self._run_epoch(self.train_ids, train=True)
            params_after = self._parameter_vector()
            param_delta = self._parameter_l2_delta(params_before, params_after)
            val_loss, val_metrics, (val_targets, val_probs) = self._run_epoch(self.val_ids, train=False)

            # Optimize classification threshold on validation set
            self.best_threshold = find_best_threshold(
                val_targets,
                val_probs,
                min_threshold=self.config.threshold_min,
                max_pred_positive_rate=self.config.threshold_max_pred_positive_rate,
                min_recall=self.config.threshold_min_recall,
                min_positive_count=self.config.threshold_min_validation_positives,
            )
            LOGGER.info("Optimized classification threshold for epoch %d: %.4f", epoch + 1, self.best_threshold)

            # Compute node-level validation metrics using the optimized threshold
            node_val_metrics = compute_classification_metrics(val_targets, val_probs, threshold=self.best_threshold)
            LOGGER.info("Validation set Node-Level metrics: %s", node_val_metrics)
            classifier_collapsed = self._classifier_is_collapsed(node_val_metrics)
            if classifier_collapsed:
                LOGGER.warning(
                    "Classifier collapse detected on validation: pred_neg=%d pred_pos=%d "
                    "prob_std=%.6e saturated_high=%.4f saturated_low=%.4f. This epoch "
                    "will not be eligible for best-checkpoint selection.",
                    int(node_val_metrics.get("pred_neg", 0.0)),
                    int(node_val_metrics.get("pred_pos", 0.0)),
                    node_val_metrics.get("prob_std", 0.0),
                    node_val_metrics.get("prob_saturated_high", 0.0),
                    node_val_metrics.get("prob_saturated_low", 0.0),
                )

            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(node_val_metrics["f1"])
            else:
                self.scheduler.step()

            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]["lr"]
            gpu_mem = current_gpu_memory_mb(self.device)

            LOGGER.info(
                "Epoch %d/%d | train_loss=%.4f val_loss=%.4f | "
                "acc=%.4f prec=%.4f rec=%.4f f1=%.4f roc_auc=%.4f pr_auc=%.4f | "
                "grad_norm=%.6f param_delta=%.6e lr=%.2e gpu_mem=%.1fMB time=%.1fs",
                epoch + 1, self.config.epochs, train_loss, val_loss,
                node_val_metrics["accuracy"], node_val_metrics["precision"],
                node_val_metrics["recall"], node_val_metrics["f1"],
                node_val_metrics["roc_auc"], node_val_metrics["pr_auc"],
                train_metrics.get("grad_norm_mean", 0.0), param_delta,
                current_lr, gpu_mem, epoch_time,
            )
            if param_delta <= 1e-12:
                raise ProductionArtifactError(
                    "Model parameters did not change during the training epoch. "
                    "Check gradient flow, optimizer parameter groups, and frozen modules."
                )

            self._log_tensorboard(epoch, train_loss, val_loss, node_val_metrics, current_lr, gpu_mem)
            self._log_csv(epoch, train_loss, val_loss, train_metrics, node_val_metrics, current_lr, param_delta)

            selection_f1 = node_val_metrics["f1"] if not classifier_collapsed else float("-inf")
            is_best = self.early_stopping.step(selection_f1)
            if is_best:
                self.best_f1 = node_val_metrics["f1"]

            self.checkpoint_manager.save(
                self.tgn, self.gat, self.mlp, self.optimizer, self.scheduler,
                epoch, self.best_f1, self.early_stopping, is_best, self.edge_weighter
            )

            if self.early_stopping.should_stop:
                LOGGER.info(
                    "Early stopping triggered after %d epochs with no F1 "
                    "improvement of at least %.6f.",
                    self.config.patience, self.config.min_delta,
                )
                break

        if self.writer is not None:
            self.writer.close()

    def evaluate_test_set(self) -> Dict[str, float]:
        _, test_metrics, (test_targets, test_probs) = self._run_epoch(self.test_ids, train=False)
        node_test_metrics = compute_classification_metrics(test_targets, test_probs, threshold=self.best_threshold)
        LOGGER.info("Test set Interaction-Level metrics: %s", test_metrics)
        LOGGER.info("Test set Node-Level metrics (Authoritative): %s", node_test_metrics)
        return node_test_metrics

    def _log_tensorboard(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_metrics: Dict[str, float],
        lr: float,
        gpu_mem: float,
    ) -> None:
        if self.writer is None:
            return
        self.writer.add_scalar("loss/train", train_loss, epoch)
        self.writer.add_scalar("loss/val", val_loss, epoch)
        for name, value in val_metrics.items():
            self.writer.add_scalar(f"val/{name}", value, epoch)
        self.writer.add_scalar("lr", lr, epoch)
        self.writer.add_scalar("system/gpu_memory_mb", gpu_mem, epoch)

    def _parameter_l2_norm(self) -> float:
        total = 0.0
        for module in (self.tgn, self.gat, self.mlp, self.edge_weighter):
            if module is None:
                continue
            for param in module.parameters():
                total += float(param.detach().float().norm(2).item() ** 2)
        return total ** 0.5

    def _parameter_vector(self) -> torch.Tensor:
        vectors: List[torch.Tensor] = []
        for module in (self.tgn, self.gat, self.mlp, self.edge_weighter):
            if module is None:
                continue
            for param in module.parameters():
                if param.requires_grad:
                    vectors.append(param.detach().float().reshape(-1).cpu())
        if not vectors:
            return torch.empty(0)
        return torch.cat(vectors)

    @staticmethod
    def _parameter_l2_delta(before: torch.Tensor, after: torch.Tensor) -> float:
        if before.numel() != after.numel():
            raise ProductionArtifactError(
                "Parameter vector size changed during training; cannot verify optimizer update."
            )
        if before.numel() == 0:
            return 0.0
        return float((after - before).norm(2).item())

    @staticmethod
    def _classifier_is_collapsed(metrics: Dict[str, float]) -> bool:
        pred_neg = int(metrics.get("pred_neg", 0.0))
        pred_pos = int(metrics.get("pred_pos", 0.0))
        prob_std = float(metrics.get("prob_std", 0.0))
        saturated_high = float(metrics.get("prob_saturated_high", 0.0))
        saturated_low = float(metrics.get("prob_saturated_low", 0.0))
        return (
            pred_neg == 0
            or pred_pos == 0
            or prob_std <= 1e-6
            or saturated_high >= 0.95
            or saturated_low >= 0.95
        )

    def _init_csv_log(self) -> None:
        header = [
            "epoch", "train_loss", "val_loss", "val_accuracy", "val_precision",
            "val_recall", "val_f1", "val_roc_auc", "val_pr_auc",
            "prob_min", "prob_max", "prob_mean", "pred_neg", "pred_pos",
            "prob_std", "prob_saturated_low", "prob_saturated_high",
            "logit_min", "logit_max", "logit_mean", "logit_std",
            "logit_saturated_abs_gt_10", "grad_norm_mean",
            "mlp_grad_norm_mean", "param_delta", "learning_rate",
        ]
        if os.path.exists(self.csv_log_path):
            with open(self.csv_log_path, "r", newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                existing_header = next(reader, [])
                has_rows = next(reader, None) is not None
            if existing_header == header:
                return
            if has_rows:
                backup_path = f"{self.csv_log_path}.bak_{int(time.time())}"
                os.replace(self.csv_log_path, backup_path)
                LOGGER.warning(
                    "Existing metrics CSV header is incompatible; moved old log to %s.",
                    backup_path,
                )
        with open(self.csv_log_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)

    def _log_csv(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        lr: float,
        param_delta: float,
    ) -> None:
        with open(self.csv_log_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                epoch + 1,
                train_loss,
                val_loss,
                val_metrics.get("accuracy", 0.0),
                val_metrics.get("precision", 0.0),
                val_metrics.get("recall", 0.0),
                val_metrics.get("f1", 0.0),
                val_metrics.get("roc_auc", 0.0),
                val_metrics.get("pr_auc", 0.0),
                val_metrics.get("prob_min", 0.0),
                val_metrics.get("prob_max", 0.0),
                val_metrics.get("prob_mean", 0.0),
                val_metrics.get("pred_neg", 0.0),
                val_metrics.get("pred_pos", 0.0),
                val_metrics.get("prob_std", 0.0),
                val_metrics.get("prob_saturated_low", 0.0),
                val_metrics.get("prob_saturated_high", 0.0),
                val_metrics.get("logit_min", 0.0),
                val_metrics.get("logit_max", 0.0),
                val_metrics.get("logit_mean", 0.0),
                val_metrics.get("logit_std", 0.0),
                val_metrics.get("logit_saturated_abs_gt_10", 0.0),
                train_metrics.get("grad_norm_mean", 0.0),
                train_metrics.get("mlp_grad_norm_mean", 0.0),
                param_delta,
                lr,
            ])


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
        params_to_clip = (
            list(trainer.tgn.parameters())
            + list(trainer.gat.parameters())
            + list(trainer.mlp.parameters())
        )
        if trainer.edge_weighter is not None:
            params_to_clip += list(trainer.edge_weighter.parameters())

        nn.utils.clip_grad_norm_(
            params_to_clip,
            config.grad_clip_norm,
        )
        trainer.optimizer.step()

    def check_checkpoint_saving() -> None:
        trainer: InsiderThreatTrainer = state["trainer"]
        trainer.checkpoint_manager.save(
            trainer.tgn, trainer.gat, trainer.mlp, trainer.optimizer,
            trainer.scheduler, epoch=0, best_f1=0.0,
            early_stopping=trainer.early_stopping, is_best=True,
            edge_weighter=trainer.edge_weighter,
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
    # CLI Overrides for TrainingConfig
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override the maximum number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=None,
        help="Override the initial learning rate.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Override the training device ('cpu', 'cuda', etc.).",
    )
    parser.add_argument(
        "--loss", type=str, default=None,
        help="Override the loss function ('bce_logits', 'focal').",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    _configure_logging()
    args = parse_args(argv)

    config = TrainingConfig.load(args.config) if args.config else TrainingConfig()

    # Apply overrides from argparse to config
    for key, value in vars(args).items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)
            LOGGER.info("Config override | %s = %s", key, value)

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
