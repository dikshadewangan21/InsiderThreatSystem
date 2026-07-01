"""
graph/edge_weighting.py

Phase 7 - Dynamic Edge Weighting for the CERT Insider Threat Heterogeneous
Temporal Graph.

This module reads the Phase 6 outputs (heterogeneous_graph.pt carrying
`edge_attr` of shape (N, 6) per dynamic relation, plus
edge_feature_metadata.pkl describing the feature order) and computes a
learnable scalar `edge_weight` per edge using:

    Weight = softmax(alpha, beta, gamma, delta) dotted with
             [SensitivityScore, PsychologyScore, CommunicationRisk, BehaviorDeviation]

The coefficients alpha, beta, gamma, delta are learnable `nn.Parameter`
values (NOT constants), initialized equally, and passed through a softmax
so they always sum to 1 and the resulting weight stays in [0, 1] (since each
input feature is itself normalized to [0, 1] in Phase 6).

`edge_attr` is preserved untouched; `edge_weight` is added as a new,
separate tensor per edge type.

Outputs:
    - graph[edge_type].edge_weight  (shape: (num_edges,))
    - graph/heterogeneous_graph.pt  (overwritten, now also carrying edge_weight)

SERIALIZATION COMPATIBILITY NOTE:
    As of graph/edge_features.py's fix for the cross-module pickle
    AttributeError, edge_feature_metadata.pkl now stores a plain nested
    dict (produced via dataclasses.asdict()) rather than an
    EdgeFeatureMetadata instance. load_metadata() below simply returns
    whatever pickle.load() yields, and resolve_feature_index() reads the
    `feature_order` entry from it. resolve_feature_index() transparently
    supports both the current dict-based format and any legacy
    object-based .pkl file that might still be present from a prior run,
    so no other part of this module's IO or weighting logic needs to
    change.

Author: Senior AI Researcher / PyG Insider Threat Detection Pipeline
"""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from torch_geometric.data import HeteroData
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "torch_geometric is required for this script. "
        "Install it with `pip install torch-geometric`."
    ) from exc


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

GRAPH_PATH = Path("graph/heterogeneous_graph.pt")
METADATA_PATH = Path("graph/edge_feature_metadata.pkl")

EDGE_ATTR_DIM = 6
EPS = 1e-12

DYNAMIC_RELATIONS: List[Tuple[str, str, str]] = [
    ("user", "uses", "pc"),
    ("user", "touches", "file_extension"),
    ("user", "visits", "website_domain"),
    ("user", "works_with", "user"),
]

# Index positions within edge_attr (must match graph/edge_features.py).
FEATURE_INDEX = {
    "Timestamp": 0,
    "SensitivityScore": 1,
    "PsychologyScore": 2,
    "CommunicationRisk": 3,
    "BehaviorDeviation": 4,
    "EventTypeID": 5,
}

WEIGHT_FEATURES = [
    "SensitivityScore",
    "PsychologyScore",
    "CommunicationRisk",
    "BehaviorDeviation",
]


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

def _build_logger() -> logging.Logger:
    logger = logging.getLogger("phase7_edge_weighting")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = _build_logger()


# --------------------------------------------------------------------------- #
# Custom exceptions
# --------------------------------------------------------------------------- #

class EdgeWeightingError(Exception):
    """Raised when dynamic edge weighting fails irrecoverably."""


class EdgeWeightValidationError(Exception):
    """Raised when produced edge_weight tensors fail validation."""


# --------------------------------------------------------------------------- #
# Core module
# --------------------------------------------------------------------------- #

class DynamicEdgeWeighting(nn.Module):
    """
    Learnable dynamic edge weighting module.

    Computes:
        Weight = alpha' * SensitivityScore
                + beta'  * PsychologyScore
                + gamma' * CommunicationRisk
                + delta' * BehaviorDeviation

    where (alpha', beta', gamma', delta') = softmax(alpha, beta, gamma, delta)
    over four learnable scalar nn.Parameters, initialized equally (i.e. all
    raw parameters start at 0.0, so the initial softmax output is uniform:
    0.25 each).

    Because each of the four input features is itself min-max normalized to
    [0, 1] (see Phase 6 / edge_features.py), and the four softmax weights are
    non-negative and sum to 1, the resulting `Weight` is guaranteed to lie in
    [0, 1] as a convex combination of values in [0, 1].
    """

    def __init__(self) -> None:
        super().__init__()
        # Initialize equally: all raw logits at zero => softmax gives a
        # uniform distribution (0.25, 0.25, 0.25, 0.25) at the start of
        # training.
        self.alpha = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.delta = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def get_coefficients(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return the current softmax-normalized coefficients."""
        raw = torch.cat([self.alpha, self.beta, self.gamma, self.delta])
        normalized = F.softmax(raw, dim=0)
        return (
            normalized[0],
            normalized[1],
            normalized[2],
            normalized[3],
        )

    def forward(
        self,
        sensitivity_score: Tensor,
        psychology_score: Tensor,
        communication_risk: Tensor,
        behavior_deviation: Tensor,
    ) -> Tensor:
        """
        Compute per-edge weights from the four constituent (already
        normalized) feature tensors, each of shape (num_edges,).

        Returns:
            Tensor of shape (num_edges,) with values in [0, 1].
        """
        alpha_n, beta_n, gamma_n, delta_n = self.get_coefficients()

        weight = (
            alpha_n * sensitivity_score
            + beta_n * psychology_score
            + gamma_n * communication_risk
            + delta_n * behavior_deviation
        )
        # Numerical safety clamp: guards against floating point drift
        # pushing values infinitesimally outside [0, 1].
        weight = torch.clamp(weight, min=0.0, max=1.0)
        return weight


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #

def load_graph(graph_path: Path) -> HeteroData:
    """Load the Phase 6 heterogeneous graph object (with edge_attr) from disk."""
    if not graph_path.exists():
        raise FileNotFoundError(f"Heterogeneous graph not found at: {graph_path}")
    logger.info("Loading heterogeneous graph from %s", graph_path)
    try:
        data = torch.load(graph_path, weights_only=False)
    except TypeError:
        data = torch.load(graph_path)
    if not isinstance(data, HeteroData):
        raise EdgeWeightingError(f"Expected a HeteroData object, got {type(data)}")
    logger.info("Graph loaded successfully. Edge types: %s", data.edge_types)
    return data


def load_metadata(metadata_path: Path) -> Any:
    """
    Load the Phase 6 edge feature metadata.

    As of the serialization-compatibility fix in graph/edge_features.py,
    this file contains a plain nested dict (see EdgeFeatureMetadata /
    build_metadata_dict in that module) rather than a custom
    EdgeFeatureMetadata instance. Loading a plain dict requires no class
    definition to be present in this module's namespace, which is exactly
    what eliminates the previous:

        AttributeError: module '__main__' has no attribute
        'EdgeFeatureMetadata'

    This function performs no interpretation of the loaded object's type —
    it simply returns whatever pickle.load() produces (dict, for current
    files; any historical custom object, for old files still on disk).
    Type-specific handling lives in resolve_feature_index() below, keeping
    this loader a pure IO primitive.
    """
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"edge_feature_metadata.pkl not found at: {metadata_path}"
        )
    logger.info("Loading edge feature metadata from %s", metadata_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    return metadata


def resolve_feature_index(metadata: Any) -> Dict[str, int]:
    """
    Resolve the feature-name -> column-index mapping, preferring the
    metadata's recorded `feature_order` if available and consistent, and
    falling back to the module-level default otherwise.

    Supports both the current on-disk format (a plain dict, as produced by
    graph/edge_features.py's build_metadata_dict) and, for backward
    compatibility with any pre-existing .pkl files, an object exposing
    `feature_order` as an attribute. This dual-path access is purely a
    metadata-reading convenience — it does not alter any Dynamic Edge
    Weighting computation.
    """
    if isinstance(metadata, dict):
        feature_order = metadata.get("feature_order")
    else:
        feature_order = getattr(metadata, "feature_order", None)

    if feature_order and len(feature_order) == EDGE_ATTR_DIM:
        resolved = {name: idx for idx, name in enumerate(feature_order)}
        missing = set(WEIGHT_FEATURES) - set(resolved.keys())
        if not missing:
            logger.info("Resolved feature index mapping from metadata: %s", resolved)
            return resolved
        logger.warning(
            "Metadata feature_order missing required features %s; "
            "falling back to default index mapping.", missing
        )
    logger.warning(
        "Using default feature index mapping (metadata absent/incomplete)."
    )
    return dict(FEATURE_INDEX)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def validate_edge_weight(
    data: HeteroData, relations: List[Tuple[str, str, str]]
) -> None:
    """Validate edge_weight tensors across all dynamic relations."""
    logger.info("Validating edge_weight tensors...")
    for edge_type in relations:
        store = data[edge_type]
        if "edge_index" not in store:
            continue
        num_edges = store["edge_index"].shape[1]
        if num_edges == 0:
            logger.info("Edge type %s has 0 edges; skipping validation.", edge_type)
            continue

        if "edge_weight" not in store:
            raise EdgeWeightValidationError(
                f"edge_weight missing for edge type {edge_type}."
            )

        edge_weight = store["edge_weight"]

        if not isinstance(edge_weight, Tensor):
            raise EdgeWeightValidationError(
                f"edge_weight for {edge_type} is not a torch.Tensor."
            )

        if edge_weight.numel() != num_edges:
            raise EdgeWeightValidationError(
                f"edge_weight for {edge_type} has {edge_weight.numel()} "
                f"elements, expected {num_edges} (one per edge)."
            )

        if torch.isnan(edge_weight).any():
            raise EdgeWeightValidationError(
                f"edge_weight for {edge_type} contains NaN."
            )

        if torch.isinf(edge_weight).any():
            raise EdgeWeightValidationError(
                f"edge_weight for {edge_type} contains Inf."
            )

        min_val = float(edge_weight.min())
        max_val = float(edge_weight.max())
        if min_val < -1e-6 or max_val > 1 + 1e-6:
            raise EdgeWeightValidationError(
                f"edge_weight for {edge_type} out of [0,1] range: "
                f"[{min_val:.6f}, {max_val:.6f}]."
            )

        if "edge_attr" not in store:
            raise EdgeWeightValidationError(
                f"edge_attr missing for {edge_type}; Phase 7 requires Phase 6 "
                "edge_attr to be preserved (not overwritten)."
            )

        logger.info(
            "Validated %s: edge_weight shape=%s, range=[%.6f, %.6f]",
            edge_type, tuple(edge_weight.shape), min_val, max_val,
        )

    logger.info("All edge_weight tensors passed validation.")


# --------------------------------------------------------------------------- #
# Statistics reporting
# --------------------------------------------------------------------------- #

def print_edge_weight_statistics(
    data: HeteroData, relations: List[Tuple[str, str, str]]
) -> None:
    """Print min/max/mean/std of edge_weight for every dynamic edge type."""
    logger.info("=== Edge Weight Statistics ===")
    for edge_type in relations:
        store = data[edge_type]
        if "edge_weight" not in store:
            continue
        weight = store["edge_weight"]
        if weight.numel() == 0:
            logger.info("%s: no edges, skipping statistics.", edge_type)
            continue

        w_min = float(weight.min())
        w_max = float(weight.max())
        w_mean = float(weight.mean())
        w_std = float(weight.std(unbiased=False))

        logger.info(
            "Edge type %s | n=%d | min=%.6f | max=%.6f | mean=%.6f | std=%.6f",
            edge_type, weight.numel(), w_min, w_max, w_mean, w_std,
        )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def compute_dynamic_edge_weights(
    graph_path: Path = GRAPH_PATH,
    metadata_path: Path = METADATA_PATH,
    relations: List[Tuple[str, str, str]] = None,
) -> Tuple[HeteroData, DynamicEdgeWeighting]:
    """
    Full Phase 7 pipeline: load graph + metadata, compute learnable
    edge_weight per dynamic relation, validate, report statistics, and
    persist the updated graph.
    """
    relations = relations if relations is not None else DYNAMIC_RELATIONS

    try:
        data = load_graph(graph_path)
        metadata = load_metadata(metadata_path)
    except (FileNotFoundError, EdgeWeightingError) as exc:
        logger.error("Failed to load required inputs: %s", exc)
        raise

    feature_index = resolve_feature_index(metadata)

    weighting_module = DynamicEdgeWeighting()
    weighting_module.eval()  # inference-mode forward; parameters remain trainable

    present_relations = [r for r in relations if r in data.edge_types]
    missing_relations = [r for r in relations if r not in data.edge_types]
    if missing_relations:
        logger.warning(
            "The following expected dynamic relations are not present in "
            "the graph and will be skipped: %s", missing_relations
        )

    for edge_type in present_relations:
        store = data[edge_type]

        if "edge_index" not in store:
            raise EdgeWeightingError(f"edge_index missing for {edge_type}.")

        num_edges = store["edge_index"].shape[1]
        if num_edges == 0:
            logger.info("Edge type %s has 0 edges; assigning empty edge_weight.", edge_type)
            data[edge_type].edge_weight = torch.zeros(0, dtype=torch.float32)
            continue

        if "edge_attr" not in store:
            raise EdgeWeightingError(
                f"edge_attr missing for {edge_type}. Run Phase 6 "
                "(graph/edge_features.py) before Phase 7."
            )

        edge_attr = store["edge_attr"]
        if edge_attr.shape != (num_edges, EDGE_ATTR_DIM):
            raise EdgeWeightingError(
                f"edge_attr for {edge_type} has unexpected shape "
                f"{tuple(edge_attr.shape)}; expected ({num_edges}, {EDGE_ATTR_DIM})."
            )

        try:
            sensitivity_score = edge_attr[:, feature_index["SensitivityScore"]]
            psychology_score = edge_attr[:, feature_index["PsychologyScore"]]
            communication_risk = edge_attr[:, feature_index["CommunicationRisk"]]
            behavior_deviation = edge_attr[:, feature_index["BehaviorDeviation"]]
        except (KeyError, IndexError) as exc:
            raise EdgeWeightingError(
                f"Failed to slice required feature columns for {edge_type}: {exc}"
            ) from exc

        with torch.no_grad():
            edge_weight = weighting_module(
                sensitivity_score=sensitivity_score,
                psychology_score=psychology_score,
                communication_risk=communication_risk,
                behavior_deviation=behavior_deviation,
            )

        # edge_attr is intentionally left untouched; edge_weight is added
        # as a new, independent tensor on the same edge store.
        data[edge_type].edge_weight = edge_weight.to(torch.float32)

        logger.info(
            "Computed edge_weight for %s with shape %s",
            edge_type, tuple(edge_weight.shape),
        )

    alpha_n, beta_n, gamma_n, delta_n = weighting_module.get_coefficients()
    logger.info(
        "Learnable coefficients (post-softmax) | alpha=%.6f beta=%.6f "
        "gamma=%.6f delta=%.6f (sum=%.6f)",
        float(alpha_n), float(beta_n), float(gamma_n), float(delta_n),
        float(alpha_n + beta_n + gamma_n + delta_n),
    )

    validate_edge_weight(data, present_relations)
    print_edge_weight_statistics(data, present_relations)

    graph_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, graph_path)
    logger.info("Saved updated heterogeneous graph (with edge_weight) to %s", graph_path)

    return data, weighting_module


def main() -> None:
    logger.info("=== Phase 7: Dynamic Edge Weighting - START ===")
    try:
        compute_dynamic_edge_weights()
    except Exception as exc:
        logger.exception("Phase 7 failed with an unrecoverable error: %s", exc)
        sys.exit(1)
    logger.info("=== Phase 7: Dynamic Edge Weighting - COMPLETE ===")


if __name__ == "__main__":
    main()