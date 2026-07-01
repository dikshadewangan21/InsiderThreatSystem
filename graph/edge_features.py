"""
graph/edge_features.py

Phase 6 - Edge Feature Generation for the CERT Insider Threat Heterogeneous
Temporal Graph.

This module builds a 6-dimensional `edge_attr` tensor for every dynamic edge
type in the previously constructed heterogeneous graph (Phase 5 output):

    ('user', 'uses',        'pc')
    ('user', 'touches',     'file_extension')
    ('user', 'visits',      'website_domain')
    ('user', 'works_with',  'user')

Feature order (dim = 6):
    0 - Timestamp              (min-max normalized to [0, 1])
    1 - SensitivityScore        (joined by file extension, normalized [0, 1])
    2 - PsychologyScore         (joined by user_id, normalized [0, 1])
    3 - CommunicationRisk       (engineered, normalized [0, 1])
    4 - BehaviorDeviation       (raw column from psychology_features.csv,
                                  normalized [0, 1], kept distinct from
                                  PsychologyScore)
    5 - EventTypeID             (auto-encoded integer id per edge type,
                                  normalized [0, 1] by dividing by
                                  (num_event_types - 1))

This script does NOT re-stream the raw CERT event logs. All features are
derived strictly from:
    - graph/heterogeneous_graph.pt      (Phase 5 output, edge indices/timestamps)
    - data/processed/psychology_features.csv
    - file_sensitivity.csv
    - graph/preprocessing_artifacts.pkl (Phase 5 output, node index registries)

Outputs:
    - graph/heterogeneous_graph.pt  (overwritten, now carrying edge_attr)
    - graph/edge_feature_metadata.pkl (feature order + normalization stats)

    NOTE ON SERIALIZATION FORMAT:
    EdgeFeatureMetadata and NormalizationStats below remain ordinary
    dataclasses and are still used as the typed, in-memory representation
    while this module builds up the metadata. However, at the point of
    persisting to disk, the fully-built object is flattened into a plain
    nested dict (via dataclasses.asdict) before being pickled. Pickle does
    not serialize class *definitions* — it stores a reference to the
    class's module + qualified name and reconstructs the instance from
    that reference at load time. When this script is executed as
    `__main__`, that reference becomes '__main__.EdgeFeatureMetadata',
    which only resolves correctly if the loading process also defines
    (or imports) that exact class under `__main__`. Any other consumer
    module (e.g. graph/edge_weighting.py, run as its own `__main__`)
    would fail with AttributeError: module '__main__' has no attribute
    'EdgeFeatureMetadata'. Persisting a plain dict instead removes this
    module-identity coupling entirely: dicts carry no class reference and
    unpickle identically regardless of which script or module loads them,
    now and in the future, without requiring any shared class module.

Author: Senior AI Researcher / PyG Insider Threat Detection Pipeline
"""

from __future__ import annotations

import dataclasses
import logging
import pickle
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
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
ARTIFACTS_PATH = Path("graph/preprocessing_artifacts.pkl")
PSYCHOLOGY_FEATURES_PATH = Path("data/processed/psychology_features.csv")
FILE_SENSITIVITY_PATH = Path("data/processed/file_sensitivity.csv")
METADATA_OUTPUT_PATH = Path("graph/edge_feature_metadata.pkl")

EDGE_ATTR_DIM = 6
EPS = 1e-12

DYNAMIC_RELATIONS: List[Tuple[str, str, str]] = [
    ("user", "uses", "pc"),
    ("user", "touches", "file_extension"),
    ("user", "visits", "website_domain"),
    ("user", "works_with", "user"),
]

FEATURE_NAMES: List[str] = [
    "Timestamp",
    "SensitivityScore",
    "PsychologyScore",
    "CommunicationRisk",
    "BehaviorDeviation",
    "EventTypeID",
]


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

def _build_logger() -> logging.Logger:
    logger = logging.getLogger("phase6_edge_features")
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

class EdgeFeatureGenerationError(Exception):
    """Raised when edge feature generation fails irrecoverably."""


class GraphValidationError(Exception):
    """Raised when the produced edge_attr tensors fail validation."""


# --------------------------------------------------------------------------- #
# Metadata container
# --------------------------------------------------------------------------- #
#
# These dataclasses are the typed, in-memory representation used while this
# module builds up edge-feature metadata. They are NOT pickled directly
# (see build_metadata_dict / generate_edge_features below) — only the plain
# dict produced by dataclasses.asdict() is ever persisted to disk. Keeping
# them as dataclasses here preserves type safety and IDE/static-analysis
# support for the construction code in this module without affecting the
# on-disk serialization format.

@dataclass
class NormalizationStats:
    """Stores min/max (or other) normalization parameters for a scalar feature."""

    method: str
    min_value: float
    max_value: float
    extra: Dict[str, float] = field(default_factory=dict)


@dataclass
class EdgeFeatureMetadata:
    """Full metadata describing how edge_attr tensors were generated."""

    feature_order: List[str]
    feature_dim: int
    event_type_to_id: Dict[str, int]
    normalization: Dict[str, Dict[str, NormalizationStats]] = field(
        default_factory=dict
    )
    relation_event_type_map: Dict[str, str] = field(default_factory=dict)


def build_metadata_dict(metadata: EdgeFeatureMetadata) -> Dict[str, Any]:
    """
    Flatten an EdgeFeatureMetadata instance (including any nested
    NormalizationStats dataclasses inside `normalization`) into a plain,
    pickle-safe nested dict.

    dataclasses.asdict() recurses into dataclass fields as well as into
    dicts/lists/tuples that contain nested dataclass instances, so this
    single call fully converts both EdgeFeatureMetadata and every
    NormalizationStats value nested under `normalization` into ordinary
    dicts. The resulting structure carries no reference to any custom
    class, so it can be unpickled by any module regardless of that
    module's `__main__` namespace.
    """
    return asdict(metadata)


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #

def load_graph(graph_path: Path) -> HeteroData:
    """Load the Phase 5 heterogeneous graph object from disk."""
    if not graph_path.exists():
        raise FileNotFoundError(f"Heterogeneous graph not found at: {graph_path}")
    logger.info("Loading heterogeneous graph from %s", graph_path)
    try:
        data = torch.load(graph_path, weights_only=False)
    except TypeError:
        # Older torch versions do not support weights_only kwarg.
        data = torch.load(graph_path)
    if not isinstance(data, HeteroData):
        raise EdgeFeatureGenerationError(
            f"Expected a HeteroData object, got {type(data)}"
        )
    logger.info("Graph loaded successfully. Node types: %s", data.node_types)
    logger.info("Graph edge types: %s", data.edge_types)
    return data


def load_preprocessing_artifacts(path: Path) -> Dict[str, Any]:
    """
    Load the preprocessing_artifacts.pkl produced by Phase 5 (build_graph.py).

    This file contains the node-index registries that map every node type's
    integer graph index back to its original identifier string:
        uid_list       : list[str]         — position i == uid string for user node i
        ext_registry   : dict[str, int]    — extension string -> node index
        domain_registry: dict[str, int]    — domain string    -> node index
        pc_registry    : dict[str, int]    — pc string        -> node index
        user_index     : dict[str, int]    — uid string       -> node index

    These are REQUIRED for the psychology and sensitivity joins. Without them,
    _resolve_node_id_array falls back to raw integer indices which never match
    the string-indexed psych_df, causing every lookup to return NaN and every
    feature to collapse to a constant (global mean). This is the root cause of
    warnings 1-3 in the original script.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessing artifacts not found at: {path}. "
            "Ensure Phase 5 (build_graph.py) has been run to completion."
        )
    logger.info("Loading preprocessing artifacts from %s", path)
    with open(path, "rb") as f:
        artifacts = pickle.load(f)
    logger.info(
        "Artifacts loaded. Available keys: %s", list(artifacts.keys())
    )
    return artifacts


def build_node_id_maps(artifacts: Dict[str, Any]) -> Dict[str, Dict[int, str]]:
    """
    Invert the node registries from Phase 5 to produce int->str lookup dicts.

    Phase 5 stores:
        uid_list[i]       = uid string for user node i   (list, position == index)
        ext_registry[ext] = node index                   (dict, inverted here)
        domain_registry[d]= node index                   (dict, inverted here)
        pc_registry[pc]   = node index                   (dict, inverted here)

    These inverse maps let _resolve_node_id_array translate a tensor of
    integer node indices back to the original identifier strings that are
    used as keys in psych_df (user_id) and sensitivity_df (extension).

    Without this inversion every join silently fails with NaN because
    psych_df is indexed by string user IDs, not by integers 0,1,2,…
    """
    id_maps: Dict[str, Dict[int, str]] = {}

    # user: index position -> uid string  (uid_list is already ordered by index)
    uid_list = artifacts.get("uid_list", [])
    if uid_list:
        id_maps["user"] = {i: uid for i, uid in enumerate(uid_list)}
        logger.info("Built user id_map for %d users.", len(id_maps["user"]))
    else:
        logger.warning(
            "uid_list not found in artifacts. User-side joins will fall back "
            "to integer indices and psychology features will be constant."
        )

    # file_extension: invert ext_registry  (ext_string -> int)  to  (int -> ext_string)
    ext_registry = artifacts.get("ext_registry", {})
    if ext_registry:
        id_maps["file_extension"] = {v: k for k, v in ext_registry.items()}
        logger.info(
            "Built file_extension id_map for %d extensions.", len(id_maps["file_extension"])
        )
    else:
        logger.warning(
            "ext_registry not found in artifacts. Sensitivity join for "
            "file_extension edges will be constant."
        )

    # domain and pc: invert their registries as well (used by downstream phases
    # and by _resolve_node_id_array for completeness)
    for key, artifact_key in [("website_domain", "domain_registry"), ("pc", "pc_registry")]:
        reg = artifacts.get(artifact_key, {})
        if reg:
            id_maps[key] = {v: k for k, v in reg.items()}

    return id_maps


def load_psychology_features(path: Path) -> pd.DataFrame:
    """Load and lightly validate the psychology feature table."""
    if not path.exists():
        raise FileNotFoundError(f"psychology_features.csv not found at: {path}")
    logger.info("Loading psychology features from %s", path)
    df = pd.read_csv(path)
    if "user_id" not in df.columns:
        raise EdgeFeatureGenerationError(
            "psychology_features.csv must contain a 'user_id' column."
        )
    if "PsychologyScore" not in df.columns:
        raise EdgeFeatureGenerationError(
            "psychology_features.csv must contain a 'PsychologyScore' column."
        )
    if "BehaviorDeviation" not in df.columns:
        raise EdgeFeatureGenerationError(
            "psychology_features.csv must contain a 'BehaviorDeviation' column."
        )

    # Normalise user_id to lowercase stripped strings so they match the
    # uid_list produced by build_graph.py (which calls normalize_uid).
    df["user_id"] = df["user_id"].astype(str).str.strip().str.lower()
    df = df.drop_duplicates(subset="user_id", keep="last").set_index("user_id")
    logger.info("Loaded psychology features for %d unique users.", len(df))

    # Log a sample of the index so the join can be verified in logs.
    sample = list(df.index[:5])
    logger.info("psychology_features sample user_ids: %s", sample)
    return df


def load_file_sensitivity(path: Path) -> pd.DataFrame:
    """Load and aggregate file sensitivity scores by file extension."""
    if not path.exists():
        raise FileNotFoundError(f"file_sensitivity.csv not found at: {path}")
    logger.info("Loading file sensitivity table from %s", path)
    df = pd.read_csv(path)
    required_cols = {"extension", "SensitivityScore"}
    missing = required_cols - set(df.columns)
    if missing:
        raise EdgeFeatureGenerationError(
            f"file_sensitivity.csv is missing required columns: {missing}"
        )
    # Normalize extension strings to match build_graph.py's extract_file_ext
    # output (lowercase, leading dot, e.g. ".docx").
    df["extension"] = (
        df["extension"]
        .astype(str)
        .str.lower()
        .str.strip()
    )
    # Ensure leading dot so ".docx" matches ".docx" from the graph registry.
    df["extension"] = df["extension"].apply(
        lambda e: e if e.startswith(".") else f".{e}"
    )
    agg = df.groupby("extension", as_index=True)["SensitivityScore"].mean()
    logger.info(
        "Aggregated sensitivity scores for %d unique extensions. Sample: %s",
        len(agg), dict(list(agg.items())[:5]),
    )
    return agg.to_frame()


# --------------------------------------------------------------------------- #
# Normalization helpers
# --------------------------------------------------------------------------- #

def min_max_normalize(
    values: np.ndarray, name: str
) -> Tuple[np.ndarray, NormalizationStats]:
    """
    Min-max normalize a 1D numpy array to [0, 1].

    Handles degenerate cases (constant arrays, empty arrays, NaNs) safely.
    """
    values = np.asarray(values, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    if values.size == 0:
        stats = NormalizationStats(method="minmax", min_value=0.0, max_value=1.0)
        return values, stats

    vmin = float(np.min(values))
    vmax = float(np.max(values))

    if abs(vmax - vmin) < EPS:
        # Degenerate / constant feature: map to a constant mid-point rather
        # than dividing by ~0, avoiding NaN/Inf.
        normalized = np.full_like(values, fill_value=0.5)
        stats = NormalizationStats(method="minmax_constant", min_value=vmin, max_value=vmax)
        logger.warning(
            "Feature '%s' is constant (min == max == %.6f). "
            "Normalized to constant 0.5.", name, vmin
        )
        return normalized, stats

    normalized = (values - vmin) / (vmax - vmin)
    normalized = np.clip(normalized, 0.0, 1.0)
    stats = NormalizationStats(method="minmax", min_value=vmin, max_value=vmax)
    return normalized, stats


# --------------------------------------------------------------------------- #
# Event type encoding
# --------------------------------------------------------------------------- #

def build_event_type_vocabulary(
    relations: List[Tuple[str, str, str]]
) -> Tuple[Dict[str, int], Dict[str, str]]:
    """
    Automatically derive an event-type vocabulary from the relation names
    themselves (e.g. 'uses', 'touches', 'visits', 'works_with'), since each
    PyG relation in this heterogeneous graph corresponds 1:1 with a distinct
    CERT-derived event type. No hardcoded mapping is used: the vocabulary is
    built dynamically from whatever relations are present in the graph.
    """
    event_types = sorted({rel[1] for rel in relations})
    event_type_to_id = {etype: idx for idx, etype in enumerate(event_types)}
    relation_event_type_map = {
        "__".join(rel): rel[1] for rel in relations
    }
    logger.info("Discovered %d unique event types: %s", len(event_types), event_types)
    return event_type_to_id, relation_event_type_map


# --------------------------------------------------------------------------- #
# Communication risk
# --------------------------------------------------------------------------- #

USER_USER_COMM_COLUMNS = [
    "ExternalCommunicationScore",
    "EmailAttachmentFrequency",
    "CommunicationEntropy",
    "CommunicationReduction",
]

GENERIC_COMM_COLUMNS = [
    "ExternalCommunicationScore",
    "EmailAttachmentFrequency",
    "AccessFrequencyShift",
    "ResourceExploration",
]


def compute_user_user_communication_risk(
    psych_df: pd.DataFrame,
    src_user_ids: np.ndarray,
    dst_user_ids: np.ndarray,
) -> np.ndarray:
    """
    Compute communication risk for user-user ('works_with') edges.

    Root cause of previous constant output: src_user_ids and dst_user_ids
    were raw integer node indices (0, 1, 2, …) because _resolve_node_id_array
    had no id_map to consult. psych_df is indexed by string user IDs, so
    composite.reindex([0, 1, 2, ...]) produced all NaN, all filled with the
    same default_val — constant output.

    Fix: src_user_ids/dst_user_ids are now resolved to uid strings via
    node_id_maps["user"] before this function is called, so the reindex
    correctly hits the psych_df string index.
    """
    available_cols = [c for c in USER_USER_COMM_COLUMNS if c in psych_df.columns]
    if not available_cols:
        logger.warning(
            "None of the expected communication columns %s found in "
            "psychology_features.csv. Falling back to zero communication risk "
            "for user-user edges.", USER_USER_COMM_COLUMNS
        )
        return np.zeros(len(src_user_ids), dtype=np.float64)

    logger.info(
        "Computing user-user communication risk from columns: %s", available_cols
    )

    sub = psych_df[available_cols].copy()
    for col in available_cols:
        col_vals = sub[col].to_numpy(dtype=np.float64)
        norm_vals, _ = min_max_normalize(col_vals, name=f"comm_component::{col}")
        sub[col] = norm_vals

    composite = sub.mean(axis=1)

    default_val = float(composite.mean()) if len(composite) > 0 else 0.0
    src_scores = composite.reindex(src_user_ids).fillna(default_val).to_numpy()
    dst_scores = composite.reindex(dst_user_ids).fillna(default_val).to_numpy()
    edge_risk = (src_scores + dst_scores) / 2.0
    return edge_risk


def compute_generic_communication_risk(
    psych_df: pd.DataFrame,
    user_ids: np.ndarray,
    event_type_id: int,
    num_event_types: int,
) -> np.ndarray:
    """
    Compute communication risk for non user-user edges (uses/touches/visits).

    Root cause of previous constant output: user_ids were raw integers, so
    composite.reindex(user_ids) returned all NaN, filled with one global mean
    — constant output for every edge.

    Fix: user_ids are now uid strings (resolved from node_id_maps["user"]
    before this call), so reindex correctly hits the psych_df string index.
    """
    available_cols = [c for c in GENERIC_COMM_COLUMNS if c in psych_df.columns]
    if not available_cols:
        logger.warning(
            "None of the expected generic communication-risk columns %s "
            "found in psychology_features.csv. Falling back to event-type "
            "only communication risk.", GENERIC_COMM_COLUMNS
        )
        event_type_prior = (
            float(event_type_id) / float(max(num_event_types - 1, 1))
        )
        return np.full(len(user_ids), fill_value=event_type_prior, dtype=np.float64)

    logger.info(
        "Computing generic communication risk from columns: %s", available_cols
    )

    sub = psych_df[available_cols].copy()
    for col in available_cols:
        col_vals = sub[col].to_numpy(dtype=np.float64)
        norm_vals, _ = min_max_normalize(col_vals, name=f"comm_component::{col}")
        sub[col] = norm_vals

    composite = sub.mean(axis=1)
    default_val = float(composite.mean()) if len(composite) > 0 else 0.0
    user_scores = composite.reindex(user_ids).fillna(default_val).to_numpy()

    event_type_prior = float(event_type_id) / float(max(num_event_types - 1, 1))
    edge_risk = 0.85 * user_scores + 0.15 * event_type_prior
    return edge_risk


# --------------------------------------------------------------------------- #
# Core per-relation feature builder
# --------------------------------------------------------------------------- #

def _resolve_node_id_array(
    data: HeteroData,
    node_type: str,
    indices: Tensor,
    id_map: Optional[Dict[int, Any]] = None,
) -> np.ndarray:
    """
    Resolve raw integer node indices back to their original identifiers.

    FIX — id_map parameter (new):
        If `id_map` is provided (an int->str dict built from Phase 5 artifacts),
        it is used directly and exclusively. This is the authoritative path.

        Previous behaviour: the function searched for node-store attributes
        like 'node_ids', 'ids', 'names' that Phase 5 never sets. The fallback
        was raw integer indices, which silently broke every downstream join
        because psych_df is indexed by string user IDs, not by 0/1/2/…

    The original graph-attribute search is retained as a secondary fallback
    for callers that do have those attributes, maintaining backwards compatibility.
    """
    indices_np = indices.detach().cpu().numpy()

    # --- Primary path: explicit id_map from Phase 5 artifacts (always correct) ---
    if id_map is not None:
        return np.array(
            [id_map.get(int(i), int(i)) for i in indices_np], dtype=object
        )

    # --- Secondary path: search node store attributes (legacy / other pipelines) ---
    candidate_attrs = [
        "node_id_map",
        "node_ids",
        "ids",
        "user_id",
        "names",
    ]
    node_store = data[node_type]
    for attr in candidate_attrs:
        if attr in node_store:
            raw = node_store[attr]
            if attr == "node_id_map" and isinstance(raw, dict):
                inverse_map = {v: k for k, v in raw.items()}
                return np.array(
                    [inverse_map.get(int(i), int(i)) for i in indices_np],
                    dtype=object,
                )
            if isinstance(raw, (list, np.ndarray, pd.Index)):
                raw_arr = np.asarray(raw, dtype=object)
                try:
                    return raw_arr[indices_np]
                except IndexError:
                    logger.warning(
                        "Index out of range while resolving '%s' for node "
                        "type '%s'; falling back to raw integer indices.",
                        attr, node_type,
                    )
                    break

    # --- Final fallback: raw positional indices (joins will fail silently) ---
    logger.warning(
        "No id_map provided and no node-store identity attribute found for "
        "node type '%s'. Using raw integer indices — downstream joins will "
        "likely produce constant features. Pass node_id_maps from artifacts.",
        node_type,
    )
    return indices_np.astype(object)


def _extract_edge_timestamps(
    data: HeteroData, edge_type: Tuple[str, str, str], num_edges: int
) -> np.ndarray:
    """
    Extract raw (un-normalized) edge timestamps for a given relation.

    Attribute search order: edge_time, edge_timestamp, timestamp, time.
    The first two are the names used by Phase 5 (build_graph.py); the last
    two are retained for compatibility.

    works_with edges — intentional synthetic timestamp:
        works_with edges are derived from the LDAP supervisor field (an
        organisational-structure snapshot), not from any time-ordered event
        log. There is no natural timestamp for "UserA shares a supervisor with
        UserB". Assigning a synthetic monotonic sequence is therefore the
        correct behaviour here, not a bug. It is explicitly flagged in logs
        and should NOT be treated as a data-quality issue when it appears for
        this specific relation. Future phases that require temporal ordering
        for works_with edges should either:
          (a) use a separate structural-edge stream that does not have a
              temporal component, or
          (b) inherit the minimum of the two users' dynamic-edge timestamps.
        Neither is implementable here without re-streaming the raw event logs,
        which Phase 6 explicitly does not do.
    """
    store = data[edge_type]
    # Phase 5 attribute names first, then legacy names.
    for attr in ("edge_time", "edge_timestamp", "timestamp", "time"):
        if attr in store:
            raw = store[attr]
            arr = (
                raw.detach().cpu().numpy()
                if isinstance(raw, Tensor)
                else np.asarray(raw)
            )
            arr = arr.astype(np.float64).reshape(-1)
            if arr.shape[0] == num_edges:
                logger.info(
                    "Found timestamp attribute '%s' on %s (%d values).",
                    attr, edge_type, num_edges,
                )
                return arr
            logger.warning(
                "Timestamp attribute '%s' on edge type %s has shape %s, "
                "expected (%d,). Ignoring.", attr, edge_type, arr.shape, num_edges
            )

    if edge_type == ("user", "works_with", "user"):
        logger.info(
            "works_with edges carry no real timestamp (they are structural, "
            "derived from LDAP supervisor relationships, not from event logs). "
            "Using a synthetic monotonic sequence. This is expected behaviour."
        )
    else:
        logger.warning(
            "No timestamp attribute found on edge type %s. Using a synthetic "
            "monotonic sequence as a fallback so downstream shapes remain "
            "consistent. This should be investigated if unexpected.", edge_type
        )
    return np.arange(num_edges, dtype=np.float64)


def build_edge_features_for_relation(
    data: HeteroData,
    edge_type: Tuple[str, str, str],
    psych_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    event_type_to_id: Dict[str, int],
    node_id_maps: Optional[Dict[str, Dict[int, Any]]] = None,
) -> Tuple[Tensor, Dict[str, NormalizationStats]]:
    """
    Build the (num_edges, 6) edge_attr tensor for a single dynamic relation.

    FIX — node_id_maps parameter (new):
        All calls to _resolve_node_id_array now pass the appropriate int->str
        lookup dict from node_id_maps. This ensures user integer node indices
        are translated to uid strings before the psych_df join, and extension
        integer indices are translated to extension strings before the
        sensitivity join.

        Without this translation every join silently returned NaN (the integer
        keys 0, 1, 2 do not exist in a DataFrame indexed by "abc@cert.org" or
        ".docx"), causing all features except Timestamp to collapse to a global
        mean constant.

    Returns the tensor plus the per-feature normalization stats for persistence.
    """
    src_type, relation, dst_type = edge_type
    store = data[edge_type]

    if "edge_index" not in store:
        raise EdgeFeatureGenerationError(
            f"edge_type {edge_type} has no edge_index; cannot build features."
        )

    edge_index = store["edge_index"]
    num_edges = edge_index.shape[1]
    logger.info("Building edge_attr for %s (%d edges)", edge_type, num_edges)

    if num_edges == 0:
        logger.warning("Edge type %s has zero edges. Skipping.", edge_type)
        empty = torch.zeros((0, EDGE_ATTR_DIM), dtype=torch.float32)
        return empty, {}

    src_idx, dst_idx = edge_index[0], edge_index[1]
    num_event_types = max(len(event_type_to_id), 1)
    event_type_id = event_type_to_id[relation]

    # Resolve id_maps for src and dst node types.
    # node_id_maps is a dict: node_type_str -> {int_index: id_string}
    src_id_map = (node_id_maps or {}).get(src_type)
    dst_id_map = (node_id_maps or {}).get(dst_type)

    feature_stats: Dict[str, NormalizationStats] = {}

    # --- Feature 0: Timestamp -------------------------------------------- #
    raw_timestamps = _extract_edge_timestamps(data, edge_type, num_edges)
    timestamp_norm, ts_stats = min_max_normalize(raw_timestamps, name="Timestamp")
    feature_stats["Timestamp"] = ts_stats

    # --- Resolve user-side identifiers for psychology join --------------- #
    # FIX: pass src_id_map / dst_id_map so integer indices resolve to uid
    # strings, enabling correct .map(psych_lookup) hits.
    if src_type == "user":
        user_node_ids = _resolve_node_id_array(
            data, "user", src_idx, id_map=src_id_map
        )
    elif dst_type == "user":
        user_node_ids = _resolve_node_id_array(
            data, "user", dst_idx, id_map=dst_id_map
        )
    else:
        raise EdgeFeatureGenerationError(
            f"Dynamic relation {edge_type} does not involve a 'user' node; "
            "cannot join psychology features."
        )

    # Log match rate to verify the join is working correctly.
    n_matched = int(pd.Series(user_node_ids).isin(psych_df.index).sum())
    logger.info(
        "%s: psychology join match rate %d / %d (%.1f%%)",
        edge_type, n_matched, num_edges, 100.0 * n_matched / max(num_edges, 1),
    )

    # --- Feature 2: PsychologyScore --------------------------------------- #
    psych_lookup = psych_df["PsychologyScore"]
    default_psych = float(psych_lookup.mean()) if len(psych_lookup) > 0 else 0.0
    raw_psych_score = (
        pd.Series(user_node_ids)
        .map(psych_lookup)
        .fillna(default_psych)
        .to_numpy(dtype=np.float64)
    )
    psych_score_norm, psych_stats = min_max_normalize(
        raw_psych_score, name="PsychologyScore"
    )
    feature_stats["PsychologyScore"] = psych_stats

    # --- Feature 4: BehaviorDeviation ------------------------------------ #
    behavior_lookup = psych_df["BehaviorDeviation"]
    default_behavior = float(behavior_lookup.mean()) if len(behavior_lookup) > 0 else 0.0
    raw_behavior_deviation = (
        pd.Series(user_node_ids)
        .map(behavior_lookup)
        .fillna(default_behavior)
        .to_numpy(dtype=np.float64)
    )
    behavior_norm, behavior_stats = min_max_normalize(
        raw_behavior_deviation, name="BehaviorDeviation"
    )
    feature_stats["BehaviorDeviation"] = behavior_stats

    # --- Feature 1: SensitivityScore --------------------------------------- #
    if relation == "touches" and dst_type == "file_extension":
        # FIX: resolve extension integer indices to extension strings via
        # dst_id_map before mapping to sensitivity_df.
        # Previously ext_node_ids contained integers like [0, 1, 2, …]; the
        # map then looked up "0", "1", "2" in an index of ".docx", ".pdf", …
        # — never matching, always NaN, always the global mean constant.
        ext_node_ids = _resolve_node_id_array(
            data, "file_extension", dst_idx, id_map=dst_id_map
        )
        ext_series = pd.Series(ext_node_ids).astype(str).str.lower().str.strip()

        n_ext_matched = int(ext_series.isin(sensitivity_df.index).sum())
        logger.info(
            "%s: sensitivity join match rate %d / %d (%.1f%%)",
            edge_type, n_ext_matched, num_edges,
            100.0 * n_ext_matched / max(num_edges, 1),
        )

        sens_lookup = sensitivity_df["SensitivityScore"]
        default_sens = float(sens_lookup.mean()) if len(sens_lookup) > 0 else 0.0
        raw_sensitivity = ext_series.map(sens_lookup).fillna(default_sens).to_numpy(
            dtype=np.float64
        )
    else:
        default_sens = (
            float(sensitivity_df["SensitivityScore"].mean())
            if len(sensitivity_df) > 0
            else 0.0
        )
        raw_sensitivity = np.full(num_edges, fill_value=default_sens, dtype=np.float64)

    sensitivity_norm, sens_stats = min_max_normalize(
        raw_sensitivity, name="SensitivityScore"
    )
    feature_stats["SensitivityScore"] = sens_stats

    # --- Feature 3: CommunicationRisk -------------------------------------- #
    # FIX: pass uid-string arrays (resolved above) not integer-index arrays.
    # compute_*_communication_risk calls composite.reindex(user_ids) — this
    # only hits psych_df when user_ids are strings like "abc@cert.org",
    # not integers like 0, 1, 2.
    if edge_type == ("user", "works_with", "user"):
        src_user_ids = _resolve_node_id_array(
            data, "user", src_idx, id_map=src_id_map
        )
        dst_user_ids = _resolve_node_id_array(
            data, "user", dst_idx, id_map=dst_id_map
        )
        raw_comm_risk = compute_user_user_communication_risk(
            psych_df, src_user_ids, dst_user_ids
        )
    else:
        raw_comm_risk = compute_generic_communication_risk(
            psych_df, user_node_ids, event_type_id, num_event_types
        )
    comm_risk_norm, comm_stats = min_max_normalize(
        raw_comm_risk, name="CommunicationRisk"
    )
    feature_stats["CommunicationRisk"] = comm_stats

    # --- Feature 5: EventTypeID --------------------------------------------- #
    event_type_id_arr = np.full(num_edges, fill_value=float(event_type_id))
    event_type_norm = event_type_id_arr / float(max(num_event_types - 1, 1))
    event_type_norm = np.clip(event_type_norm, 0.0, 1.0)
    feature_stats["EventTypeID"] = NormalizationStats(
        method="fixed_divide",
        min_value=0.0,
        max_value=float(max(num_event_types - 1, 1)),
        extra={"event_type_id": float(event_type_id)},
    )

    edge_attr_np = np.stack(
        [
            timestamp_norm,
            sensitivity_norm,
            psych_score_norm,
            comm_risk_norm,
            behavior_norm,
            event_type_norm,
        ],
        axis=1,
    ).astype(np.float32)

    edge_attr = torch.from_numpy(edge_attr_np)
    return edge_attr, feature_stats


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def validate_edge_attr(
    data: HeteroData, relations: List[Tuple[str, str, str]]
) -> None:
    """Validate edge_attr tensors across all dynamic relations."""
    logger.info("Validating edge_attr tensors...")
    for edge_type in relations:
        store = data[edge_type]
        if "edge_index" not in store:
            continue
        num_edges = store["edge_index"].shape[1]
        if num_edges == 0:
            logger.info("Edge type %s has 0 edges; skipping validation.", edge_type)
            continue

        if "edge_attr" not in store:
            raise GraphValidationError(
                f"edge_attr missing for edge type {edge_type}."
            )

        edge_attr = store["edge_attr"]

        if not isinstance(edge_attr, Tensor):
            raise GraphValidationError(
                f"edge_attr for {edge_type} is not a torch.Tensor."
            )

        expected_shape = (num_edges, EDGE_ATTR_DIM)
        if tuple(edge_attr.shape) != expected_shape:
            raise GraphValidationError(
                f"edge_attr for {edge_type} has shape {tuple(edge_attr.shape)}, "
                f"expected {expected_shape}."
            )

        if torch.isnan(edge_attr).any():
            raise GraphValidationError(f"edge_attr for {edge_type} contains NaN.")

        if torch.isinf(edge_attr).any():
            raise GraphValidationError(f"edge_attr for {edge_type} contains Inf.")

        min_val = float(edge_attr.min())
        max_val = float(edge_attr.max())
        if min_val < -1e-6 or max_val > 1 + 1e-6:
            raise GraphValidationError(
                f"edge_attr for {edge_type} is not normalized to [0,1]. "
                f"Found range [{min_val:.6f}, {max_val:.6f}]."
            )

        logger.info(
            "Validated %s: shape=%s, range=[%.6f, %.6f], dtype=%s",
            edge_type, tuple(edge_attr.shape), min_val, max_val, edge_attr.dtype,
        )

    logger.info("All edge_attr tensors passed validation.")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def generate_edge_features(
    graph_path: Path = GRAPH_PATH,
    artifacts_path: Path = ARTIFACTS_PATH,
    psychology_path: Path = PSYCHOLOGY_FEATURES_PATH,
    sensitivity_path: Path = FILE_SENSITIVITY_PATH,
    metadata_output_path: Path = METADATA_OUTPUT_PATH,
    relations: Optional[List[Tuple[str, str, str]]] = None,
) -> HeteroData:
    """
    Full Phase 6 pipeline: load inputs, compute edge_attr for every dynamic
    relation, validate, and persist outputs.

    FIX — artifacts_path parameter (new):
        The Phase 5 preprocessing_artifacts.pkl is now loaded to obtain
        node-index registries (uid_list, ext_registry, domain_registry,
        pc_registry). These are inverted into int->str lookup dicts
        (node_id_maps) and passed to every call of
        build_edge_features_for_relation.

        Without this, _resolve_node_id_array returned raw integer node
        indices, psych_df.map(integer_series) hit zero keys, every feature
        collapsed to the global mean — all four constant-feature warnings.

    SERIALIZATION FIX:
        The EdgeFeatureMetadata instance built here is no longer pickled
        directly. Immediately before writing to disk it is converted to a
        plain nested dict via build_metadata_dict() (dataclasses.asdict),
        and that dict is what gets pickled. This eliminates the
        module-identity coupling that caused
        `AttributeError: module '__main__' has no attribute
        'EdgeFeatureMetadata'` when graph/edge_weighting.py (a different
        `__main__` context) attempted to unpickle the file. The in-memory
        `metadata` object and all of its fields are unchanged; only the
        on-disk representation differs.
    """
    relations = relations if relations is not None else DYNAMIC_RELATIONS

    try:
        data = load_graph(graph_path)
        artifacts = load_preprocessing_artifacts(artifacts_path)
        psych_df = load_psychology_features(psychology_path)
        sensitivity_df = load_file_sensitivity(sensitivity_path)
    except (FileNotFoundError, EdgeFeatureGenerationError) as exc:
        logger.error("Failed to load required inputs: %s", exc)
        raise

    # Build int->str lookup dicts from Phase 5 registries.
    node_id_maps = build_node_id_maps(artifacts)

    event_type_to_id, relation_event_type_map = build_event_type_vocabulary(relations)

    metadata = EdgeFeatureMetadata(
        feature_order=FEATURE_NAMES,
        feature_dim=EDGE_ATTR_DIM,
        event_type_to_id=event_type_to_id,
        relation_event_type_map=relation_event_type_map,
    )

    present_relations = [r for r in relations if r in data.edge_types]
    missing_relations = [r for r in relations if r not in data.edge_types]
    if missing_relations:
        logger.warning(
            "The following expected dynamic relations are not present in "
            "the graph and will be skipped: %s", missing_relations
        )

    for edge_type in present_relations:
        try:
            edge_attr, feature_stats = build_edge_features_for_relation(
                data=data,
                edge_type=edge_type,
                psych_df=psych_df,
                sensitivity_df=sensitivity_df,
                event_type_to_id=event_type_to_id,
                node_id_maps=node_id_maps,
            )
        except Exception as exc:
            logger.error("Failed to build edge_attr for %s: %s", edge_type, exc)
            raise EdgeFeatureGenerationError(
                f"Edge feature generation failed for relation {edge_type}"
            ) from exc

        data[edge_type].edge_attr = edge_attr
        metadata.normalization["__".join(edge_type)] = feature_stats
        logger.info(
            "Assigned edge_attr to %s with final shape %s",
            edge_type, tuple(edge_attr.shape),
        )

    validate_edge_attr(data, present_relations)

    graph_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, graph_path)
    logger.info("Overwrote heterogeneous graph at %s", graph_path)

    # --- Serialization-compatibility fix -------------------------------- #
    # Flatten the EdgeFeatureMetadata (and any nested NormalizationStats)
    # into a plain dict before pickling. Plain dicts carry no class
    # reference, so they unpickle identically in any module/__main__
    # context, permanently resolving the cross-module AttributeError.
    metadata_dict = build_metadata_dict(metadata)

    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_output_path, "wb") as f:
        pickle.dump(metadata_dict, f)
    logger.info("Saved edge feature metadata to %s", metadata_output_path)

    return data


def main() -> None:
    logger.info("=== Phase 6: Edge Feature Generation - START ===")
    try:
        generate_edge_features()
    except Exception as exc:
        logger.exception("Phase 6 failed with an unrecoverable error: %s", exc)
        sys.exit(1)
    logger.info("=== Phase 6: Edge Feature Generation - COMPLETE ===")


if __name__ == "__main__":
    main()