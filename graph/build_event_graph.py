"""
build_event_graph.py

Event-level Temporal Heterogeneous Graph Builder for CERT-style Insider
Threat Detection pipelines.

--------------------------------------------------------------------------
DESIGN CONTRACT (do not violate in downstream phases)
--------------------------------------------------------------------------
- Every row of `unified_events.csv` whose `event_type` maps to a known
  temporal relation becomes EXACTLY ONE temporal edge. There is no
  groupby/aggregation of interactions: a PC visited 483 times by a user
  produces 483 separate (User -> PC) edges, each with its own timestamp.
- Structural relations (User-Department, User-Role, User-Team,
  User-BusinessUnit, User-User "works_with") are created once per user
  from the LDAP directory snapshot, since they are not events.
- User node features come verbatim from `fused_features.csv`. They are
  never normalized, rescaled, or recomputed here.
- Timestamps are stored as raw Unix integers. No normalization.
- Output artifacts keys are fixed and MUST NOT be renamed; later phases
  (edge_features.py, edge_weighting.py, temporal_preprocessing.py,
  tgn.py, hetero_gat.py, classifier.py) depend on this exact schema.

--------------------------------------------------------------------------
EXPLICIT ASSUMPTIONS (documented because the source spec leaves them
implicit; change the constants below if your data differs)
--------------------------------------------------------------------------
1. `unified_events.csv` contains an `event_type` column whose values are
   drawn from CERT's canonical categories: {"logon", "logoff", "device",
   "http", "file", "email", ...}. Only "logon"/"logoff"/"device" (-> PC),
   "http" (-> WebsiteDomain), and "file" (-> FileExtension) have a
   corresponding temporal edge type in the required schema. Any other
   event_type value (e.g. "email", which has no declared node/edge type
   in this schema) is intentionally NOT converted into an edge. Rows are
   still counted and reported in the validation summary
   (`skipped_event_rows`) so nothing silently disappears.
2. A directory snapshot file `data/processed/ldap.csv` is expected with
   columns: user_id, department, role, team, business_unit, supervisor.
   Adjust `--ldap` if your LDAP export lives elsewhere or is named
   differently.
3. A user_id present in `unified_events.csv` but absent from
   `fused_features.csv` cannot be assigned a feature vector without
   violating the "do not recompute user features" rule, so such rows are
   dropped and counted under `dropped_unknown_user_rows`. In a
   well-formed CERT export this count should be 0.
4. WebsiteDomain nodes are derived by stripping scheme/"www."/path from
   the `resource` field. FileExtension nodes are derived by taking the
   substring after the final "." in the `resource` field (filename).

Run:
    python build_event_graph.py \
        --events data/processed/unified_events.csv \
        --fused-features data/processed/fused_features.csv \
        --file-sensitivity data/processed/file_sensitivity.csv \
        --ldap data/processed/ldap.csv \
        --output-dir graph/output \
        --chunk-size 500000
"""

from __future__ import annotations

import argparse
import gc
import itertools
import logging
import os
import pickle
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_PSUTIL = False


# ============================================================================
# Logging
# ============================================================================

def setup_logging(log_path: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("event_graph_builder")
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


LOGGER = logging.getLogger("event_graph_builder")


def log_memory(tag: str) -> None:
    if _HAS_PSUTIL:
        proc = psutil.Process(os.getpid())
        rss_mb = proc.memory_info().rss / (1024 ** 2)
        LOGGER.info(f"[MEMORY] {tag}: {rss_mb:,.1f} MB RSS")
    else:
        LOGGER.info(f"[MEMORY] {tag}: psutil not installed, skipping")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class GraphBuildConfig:
    events_path: Path = Path("data/processed/unified_events.csv")
    fused_features_path: Path = Path("data/processed/fused_features.csv")
    file_sensitivity_path: Path = Path("data/processed/file_sensitivity.csv")
    ldap_path: Path = Path("data/processed/ldap.csv")

    output_dir: Path = Path("graph/output")

    chunk_size: int = 500_000

    user_id_col: str = "user_id"
    timestamp_col: str = "timestamp"
    event_type_col: str = "event_type"
    resource_col: str = "resource"
    target_user_col: str = "target_user"

    # Populated in __post_init__, not passed by the caller.
    graph_output_path: Path = field(init=False)
    artifacts_output_path: Path = field(init=False)
    log_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.graph_output_path = self.output_dir / "heterogeneous_graph.pt"
        self.artifacts_output_path = self.output_dir / "preprocessing_artifacts.pkl"
        self.log_path = self.output_dir / "build_event_graph.log"


# Maps a lowercased CERT event_type value to the (src_node_type, dst_node_type)
# temporal relation it belongs to. Only relations declared in the schema.
EVENT_TYPE_TO_RELATION: Dict[str, Tuple[str, str]] = {
    "logon": ("User", "PC"),
    "logoff": ("User", "PC"),
    "device": ("User", "PC"),
    "http": ("User", "WebsiteDomain"),
    "file": ("User", "FileExtension"),
}

TEMPORAL_RELATION_NAME: Dict[Tuple[str, str], str] = {
    ("User", "PC"): "accesses",
    ("User", "WebsiteDomain"): "visits",
    ("User", "FileExtension"): "touches",
}

def _to_unix_seconds(series: pd.Series) -> np.ndarray:
    """
    Convert a column of timestamp strings (e.g. "2010-01-02 06:49:00") to
    Unix epoch seconds as int64.

    CERT-style logs store timestamps as human-readable datetime strings,
    NOT raw Unix integers. Calling `.astype(np.int64)` directly on such a
    string ("2010-01-02 06:49:00" -> int) raises ValueError, since that is
    not a valid integer literal - it must first be parsed as a datetime.

    errors="coerce" ensures a malformed/unparseable timestamp becomes NaT
    instead of raising and killing a multi-hour streaming job; NaT rows
    are mapped to epoch 0 and counted so they remain visible rather than
    silently vanishing.
    """
    parsed = pd.to_datetime(series, errors="coerce")
    invalid = int(parsed.isna().sum())
    # IMPORTANT: do not assume nanosecond resolution here. Since pandas
    # 2.x, pd.to_datetime infers whatever resolution the input strings
    # support (s / ms / us / ns) - viewing that as int64 and dividing by
    # a hardcoded 10**9 is WRONG whenever the inferred resolution isn't
    # ns (e.g. plain "YYYY-MM-DD HH:MM:SS" strings often infer as
    # microsecond or second resolution), and it fails silently rather
    # than raising - it just produces the wrong Unix timestamp. Casting
    # explicitly to datetime64[s] first makes the result resolution-
    # independent: the underlying int64 view is then already whole
    # Unix seconds, no division needed.
    is_na = parsed.isna().to_numpy()
    seconds = parsed.astype("datetime64[s]").to_numpy().view("int64")
    seconds = np.where(is_na, 0, seconds).astype(np.int64)
    return seconds, invalid


_DOMAIN_RE = re.compile(r"^(?:https?://)?(?:www\.)?([^/:?#]+)")
_EXTENSION_RE = re.compile(r"\.([a-z0-9]{1,10})$")


def _normalize_id(series: pd.Series) -> pd.Series:
    """
    Canonical join-key normalization for user_id (and any other identifier
    used as a cross-file key: PC ids, etc).

    This MUST be applied identically to every file that contributes to a
    join, or joins silently fail with a 100% mismatch rate even when the
    underlying identifier is "the same" to a human reader. Specifically
    normalizes away the differences that are most common between a CERT
    LDAP export, a fused-features export, and a raw events export:
      - numeric dtype vs string dtype (e.g. 1024 vs "1024" vs "1024.0")
      - case (e.g. "MSC0001" vs "msc0001")
      - surrounding whitespace and stray quote characters
    Does NOT strip or reinterpret leading zeros - if your IDs are
    zero-padded codes (e.g. "0007"), that padding is preserved, because
    guessing whether "7" and "0007" are "the same" id is not safe to do
    silently.
    """
    s = series.astype(str).str.strip()
    s = s.str.replace(r'^["\']|["\']$', "", regex=True)
    # Collapse an accidental trailing ".0" produced when a numeric-looking
    # ID column got read as float upstream (e.g. by Excel/pandas dtype
    # inference) - this is the single most common source of a 100%-failed
    # numeric-id join and is safe to normalize (never a legitimate id).
    s = s.str.replace(r"\.0$", "", regex=True)
    return s.str.upper()


# ============================================================================
# Registries
# ============================================================================

class Registry:
    """
    Incrementally-growable string -> contiguous integer index registry.

    Safe to extend across streamed chunks: new keys are appended at the
    end, existing keys always keep their original index. This is what
    lets us build node vocabularies (PC ids, domains, extensions, ...)
    without ever loading the full event file into memory at once.
    """

    __slots__ = ("_name", "_str_to_idx", "_idx_to_str")

    def __init__(self, name: str) -> None:
        self._name = name
        self._str_to_idx: Dict[str, int] = {}
        self._idx_to_str: List[str] = []

    def __len__(self) -> int:
        return len(self._idx_to_str)

    @property
    def name(self) -> str:
        return self._name

    def get_or_create(self, key: str) -> int:
        idx = self._str_to_idx.get(key)
        if idx is None:
            idx = len(self._idx_to_str)
            self._str_to_idx[key] = idx
            self._idx_to_str.append(key)
        return idx

    def bulk_get_or_create(self, keys: pd.Series) -> np.ndarray:
        """
        Vectorized registration of a pandas Series of string keys.
        Returns an int64 numpy array of indices aligned with `keys`.
        """
        unique_keys = pd.unique(keys)
        for k in unique_keys:
            if k not in self._str_to_idx:
                idx = len(self._idx_to_str)
                self._str_to_idx[k] = idx
                self._idx_to_str.append(k)
        return keys.map(self._str_to_idx).to_numpy(dtype=np.int64)

    def to_list(self) -> List[str]:
        return list(self._idx_to_str)

    def to_dict(self) -> Dict[str, int]:
        return dict(self._str_to_idx)


# ============================================================================
# Edge accumulation (streaming-friendly)
# ============================================================================

class EdgeAccumulator:
    """
    Accumulates per-chunk numpy arrays for a single temporal relation
    without repeatedly re-concatenating the full edge list on every
    chunk. Concatenation happens exactly once, in `finalize()`, which
    keeps per-chunk cost O(chunk_size) instead of O(total_edges_so_far).
    """

    def __init__(self) -> None:
        self._src: List[np.ndarray] = []
        self._dst: List[np.ndarray] = []
        self._edge_time: List[np.ndarray] = []
        self._raw_timestamp: List[np.ndarray] = []
        self._event_type: List[np.ndarray] = []
        self._event_index: List[np.ndarray] = []
        self._resource: List[np.ndarray] = []
        self._target_user: List[np.ndarray] = []
        self.num_edges: int = 0

    def add_chunk(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        edge_time: np.ndarray,
        raw_timestamp: np.ndarray,
        event_type: np.ndarray,
        event_index: np.ndarray,
        resource: np.ndarray,
        target_user: np.ndarray,
    ) -> None:
        self._src.append(src)
        self._dst.append(dst)
        self._edge_time.append(edge_time)
        self._raw_timestamp.append(raw_timestamp)
        self._event_type.append(event_type)
        self._event_index.append(event_index)
        self._resource.append(resource)
        self._target_user.append(target_user)
        self.num_edges += src.shape[0]

    def finalize(self) -> Dict[str, torch.Tensor]:
        if self.num_edges == 0:
            empty_long = torch.empty((0,), dtype=torch.long)
            return {
                "edge_index": torch.empty((2, 0), dtype=torch.long),
                "edge_time": empty_long,
                "raw_timestamp": empty_long,
                "event_type": empty_long,
                "event_index": empty_long,
                "resource": empty_long,
                "target_user": empty_long,
            }

        src = np.concatenate(self._src)
        dst = np.concatenate(self._dst)
        edge_time = np.concatenate(self._edge_time)
        raw_timestamp = np.concatenate(self._raw_timestamp)
        event_type = np.concatenate(self._event_type)
        event_index = np.concatenate(self._event_index)
        resource = np.concatenate(self._resource)
        target_user = np.concatenate(self._target_user)

        # Required: edges within a temporal relation must be chronological.
        order = np.argsort(edge_time, kind="stable")
        edge_index = np.stack([src[order], dst[order]], axis=0)

        return {
            "edge_index": torch.from_numpy(edge_index).long(),
            "edge_time": torch.from_numpy(edge_time[order]).long(),
            "raw_timestamp": torch.from_numpy(raw_timestamp[order]).long(),
            "event_type": torch.from_numpy(event_type[order]).long(),
            "event_index": torch.from_numpy(event_index[order]).long(),
            "resource": torch.from_numpy(resource[order]).long(),
            "target_user": torch.from_numpy(target_user[order]).long(),
        }


# ============================================================================
# Builder
# ============================================================================

class EventGraphBuilder:
    """
    Orchestrates the full CERT unified_events.csv -> HeteroData pipeline:
    load static tables (LDAP, fused features, sensitivity), build
    structural edges once, stream event-level temporal edges chunk by
    chunk, assemble a PyG HeteroData object, validate it, and persist it
    together with the preprocessing artifacts future phases depend on.
    """

    def __init__(self, config: GraphBuildConfig) -> None:
        self.config = config

        self.dept_registry = Registry("Department")
        self.role_registry = Registry("Role")
        self.team_registry = Registry("Team")
        self.bu_registry = Registry("BusinessUnit")
        self.pc_registry = Registry("PC")
        self.website_registry = Registry("WebsiteDomain")
        self.ext_registry = Registry("FileExtension")
        self.event_type_registry = Registry("EventType")

        self.uid_list: List[str] = []
        self.uid_to_index: Dict[str, int] = {}
        self.user_features: Optional[torch.Tensor] = None
        self.user_feature_columns: List[str] = []

        self.uid_to_dept: Dict[str, str] = {}
        self.uid_to_role: Dict[str, str] = {}
        self.uid_to_team: Dict[str, str] = {}
        self.uid_to_bu: Dict[str, str] = {}
        self.uid_to_supervisor: Dict[str, str] = {}

        self.extension_sensitivity: Dict[str, float] = {}

        self.structural_edges: Dict[Tuple[str, str, str], torch.Tensor] = {}
        self.edge_data: Dict[Tuple[str, str], Dict[str, torch.Tensor]] = {}
        self.graph: Optional[HeteroData] = None

        self.stats: Dict[str, int] = defaultdict(int)
        self._unknown_user_examples_logged = False

    # ------------------------------------------------------------------ #
    # Static table loading
    # ------------------------------------------------------------------ #

    def load_ldap(self) -> None:
        cfg = self.config
        LOGGER.info(f"Loading LDAP directory snapshot from {cfg.ldap_path}")
        df = pd.read_csv(cfg.ldap_path)

        raw_dtype = df[cfg.user_id_col].dtype
        raw_samples = df[cfg.user_id_col].astype(str).head(3).tolist()

        # Normalize to the SAME join-key convention used everywhere else:
        # stringified, trimmed, and case-folded. Case is the most common
        # silent mismatch between LDAP exports and other CERT tables.
        df[cfg.user_id_col] = _normalize_id(df[cfg.user_id_col])

        LOGGER.info(
            f"[DIAG] LDAP.csv user_id -> source dtype={raw_dtype}, "
            f"raw samples={raw_samples}, normalized samples={df[cfg.user_id_col].head(3).tolist()}"
        )

        for col in ["department", "role", "team", "business_unit", "supervisor"]:
            if col not in df.columns:
                LOGGER.warning(f"LDAP file missing column '{col}'; filling with 'unknown'")
                df[col] = "unknown"
            df[col] = df[col].fillna("unknown").astype(str).str.strip()

        self.uid_to_dept = dict(zip(df[cfg.user_id_col], df["department"]))
        self.uid_to_role = dict(zip(df[cfg.user_id_col], df["role"]))
        self.uid_to_team = dict(zip(df[cfg.user_id_col], df["team"]))
        self.uid_to_bu = dict(zip(df[cfg.user_id_col], df["business_unit"]))
        self.uid_to_supervisor = dict(zip(df[cfg.user_id_col], df["supervisor"]))
        LOGGER.info(f"Loaded LDAP records for {len(df):,} identities")

    def load_fused_features(self) -> None:
        cfg = self.config
        LOGGER.info(f"Loading fused user features from {cfg.fused_features_path}")
        df = pd.read_csv(cfg.fused_features_path)

        if cfg.user_id_col not in df.columns:
            raise ValueError(f"'{cfg.user_id_col}' column missing from fused_features.csv")

        raw_dtype = df[cfg.user_id_col].dtype
        raw_samples = df[cfg.user_id_col].astype(str).head(3).tolist()

        df[cfg.user_id_col] = _normalize_id(df[cfg.user_id_col])

        LOGGER.info(
            f"[DIAG] fused_features.csv user_id -> source dtype={raw_dtype}, "
            f"raw samples={raw_samples}, normalized samples={df[cfg.user_id_col].head(3).tolist()}"
        )

        before = len(df)
        df = df.drop_duplicates(subset=[cfg.user_id_col], keep="first").reset_index(drop=True)
        if len(df) != before:
            LOGGER.warning(f"Dropped {before - len(df):,} duplicate user_id rows from fused_features.csv")

        self.uid_list = df[cfg.user_id_col].tolist()
        self.uid_to_index = {uid: i for i, uid in enumerate(self.uid_list)}

        feature_cols = [c for c in df.columns if c != cfg.user_id_col]
        feature_df = df[feature_cols]

        non_numeric = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            LOGGER.warning(
                f"Dropping {len(non_numeric)} non-numeric feature column(s) that cannot "
                f"be placed in a float tensor as-is: {non_numeric}"
            )
            feature_df = feature_df.drop(columns=non_numeric)

        nan_count = int(feature_df.isna().sum().sum())
        if nan_count:
            LOGGER.warning(
                f"{nan_count:,} NaN value(s) found in fused user features; zero-filling "
                f"for tensor validity only (values themselves are not recomputed)."
            )
            feature_df = feature_df.fillna(0.0)

        self.user_feature_columns = feature_df.columns.tolist()
        self.user_features = torch.tensor(feature_df.to_numpy(dtype=np.float32))
        LOGGER.info(
            f"Loaded {len(self.uid_list):,} users with "
            f"{self.user_features.shape[1]} raw feature dimensions each"
        )

    def load_file_sensitivity(self) -> None:
        cfg = self.config
        LOGGER.info(f"Loading file sensitivity table from {cfg.file_sensitivity_path}")
        df = pd.read_csv(cfg.file_sensitivity_path)
        df["extension"] = df["extension"].astype(str).str.lower().str.strip().str.lstrip(".")
        df = df.drop_duplicates(subset=["extension"], keep="first")
        self.extension_sensitivity = dict(zip(df["extension"], df["SensitivityScore"].astype(float)))
        LOGGER.info(f"Loaded sensitivity scores for {len(self.extension_sensitivity):,} file extensions")

    # ------------------------------------------------------------------ #
    # Structural (non-temporal) edges - built once from LDAP
    # ------------------------------------------------------------------ #

    def build_structural_edges(self) -> None:
        LOGGER.info("Building structural edges from LDAP (each exists exactly once per user)")

        dept_src: List[int] = []
        dept_dst: List[int] = []
        role_src: List[int] = []
        role_dst: List[int] = []
        team_src: List[int] = []
        team_dst: List[int] = []
        bu_src: List[int] = []
        bu_dst: List[int] = []

        for uid, uidx in self.uid_to_index.items():
            dept_src.append(uidx)
            dept_dst.append(self.dept_registry.get_or_create(self.uid_to_dept.get(uid, "unknown")))

            role_src.append(uidx)
            role_dst.append(self.role_registry.get_or_create(self.uid_to_role.get(uid, "unknown")))

            team_src.append(uidx)
            team_dst.append(self.team_registry.get_or_create(self.uid_to_team.get(uid, "unknown")))

            bu_src.append(uidx)
            bu_dst.append(self.bu_registry.get_or_create(self.uid_to_bu.get(uid, "unknown")))

        self.structural_edges = {
            ("User", "belongs_to", "Department"): self._edge_index(dept_src, dept_dst),
            ("User", "has_role", "Role"): self._edge_index(role_src, role_dst),
            ("User", "member_of", "Team"): self._edge_index(team_src, team_dst),
            ("User", "part_of", "BusinessUnit"): self._edge_index(bu_src, bu_dst),
            ("User", "works_with", "User"): self._build_works_with_edges(),
        }

        for rel, ei in self.structural_edges.items():
            LOGGER.info(f"[EDGES] {rel[0]}-{rel[1]}->{rel[2]}: {ei.shape[1]:,} structural edges")

    def _build_works_with_edges(self) -> torch.Tensor:
        """
        Users who share the same supervisor are considered co-workers.
        This is a structural fact (who works with whom), not an event,
        so it is created exactly once per qualifying user pair, in both
        directions (works_with is symmetric).
        """
        supervisors = [self.uid_to_supervisor.get(uid) for uid in self.uid_list]
        df = pd.DataFrame({"idx": np.arange(len(self.uid_list)), "supervisor": supervisors})
        df = df.dropna(subset=["supervisor"])
        df = df[df["supervisor"] != "unknown"]

        src_parts: List[np.ndarray] = []
        dst_parts: List[np.ndarray] = []

        for _, group in df.groupby("supervisor"):
            idxs = group["idx"].to_numpy()
            if len(idxs) < 2:
                continue
            combos = np.array(list(itertools.permutations(idxs, 2)), dtype=np.int64)
            src_parts.append(combos[:, 0])
            dst_parts.append(combos[:, 1])

        if not src_parts:
            return torch.empty((2, 0), dtype=torch.long)

        src = np.concatenate(src_parts)
        dst = np.concatenate(dst_parts)
        return torch.from_numpy(np.stack([src, dst])).long()

    @staticmethod
    def _edge_index(src: List[int], dst: List[int]) -> torch.Tensor:
        if not src:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor([src, dst], dtype=torch.long)

    # ------------------------------------------------------------------ #
    # Temporal edges - streamed, one edge per event row
    # ------------------------------------------------------------------ #

    def process_events_streaming(self) -> None:
        cfg = self.config
        LOGGER.info(f"Streaming events from {cfg.events_path} in chunks of {cfg.chunk_size:,} rows")

        accumulators: Dict[Tuple[str, str], EdgeAccumulator] = {
            ("User", "PC"): EdgeAccumulator(),
            ("User", "WebsiteDomain"): EdgeAccumulator(),
            ("User", "FileExtension"): EdgeAccumulator(),
        }

        total_rows = 0
        global_row_offset = 0
        t_start = time.time()

        dtype_map = {
            cfg.user_id_col: "string",
            cfg.event_type_col: "string",
            cfg.resource_col: "string",
            cfg.target_user_col: "string",
        }

        reader = pd.read_csv(
            cfg.events_path,
            chunksize=cfg.chunk_size,
            dtype=dtype_map,
            usecols=[
                cfg.user_id_col,
                cfg.timestamp_col,
                cfg.event_type_col,
                cfg.resource_col,
                cfg.target_user_col,
            ],
        )

        for chunk_idx, chunk in enumerate(reader):
            chunk_rows = len(chunk)
            total_rows += chunk_rows

            chunk[cfg.event_type_col] = chunk[cfg.event_type_col].str.lower().str.strip()
            # Use the SAME normalization as load_ldap/load_fused_features.
            # Using .str.strip() here alone (previous behavior) preserves
            # case, which silently fails every join if any source file's
            # casing differs even slightly - this was the root cause of
            # 30M/32.7M rows being dropped as "unknown user".
            chunk[cfg.user_id_col] = _normalize_id(chunk[cfg.user_id_col])

            chunk["_row_idx"] = np.arange(
                global_row_offset, global_row_offset + chunk_rows, dtype=np.int64
            )
            global_row_offset += chunk_rows

            if chunk_idx == 0:
                unique_types = sorted(chunk[cfg.event_type_col].dropna().unique().tolist())
                LOGGER.info(f"[DIAG] unique event_type values in first chunk: {unique_types}")
                LOGGER.info(
                    f"[DIAG] first chunk normalized user_id samples: "
                    f"{chunk[cfg.user_id_col].head(3).tolist()} "
                    f"(compare against the fused_features/LDAP samples logged above)"
                )
                unmatched = chunk.loc[
                    ~chunk[cfg.user_id_col].isin(self.uid_to_index.keys()), cfg.user_id_col
                ].unique()
                if len(unmatched) > 0:
                    LOGGER.warning(
                        f"[DIAG] {len(unmatched):,} distinct user_id value(s) in the first "
                        f"chunk do not exist in fused_features.csv. Examples: "
                        f"{list(unmatched[:5])}"
                    )

            mapped_mask = chunk[cfg.event_type_col].isin(EVENT_TYPE_TO_RELATION.keys())
            self.stats["skipped_event_rows"] += int((~mapped_mask).sum())

            for event_key, (src_type, dst_type) in EVENT_TYPE_TO_RELATION.items():
                key_mask = chunk[cfg.event_type_col] == event_key
                if not key_mask.any():
                    continue
                sub = chunk.loc[key_mask]
                self._process_relation_subframe(
                    sub, src_type, dst_type, accumulators[(src_type, dst_type)]
                )

            if chunk_idx % 5 == 0:
                elapsed = time.time() - t_start
                rate = total_rows / elapsed if elapsed > 0 else 0.0
                LOGGER.info(
                    f"[STREAM] chunk={chunk_idx:,} rows_so_far={total_rows:,} "
                    f"rate={rate:,.0f} rows/sec"
                )
                log_memory(f"after chunk {chunk_idx}")

            del chunk
            gc.collect()

        elapsed = time.time() - t_start
        LOGGER.info(f"Finished streaming {total_rows:,} rows in {elapsed:,.1f}s")

        skipped = self.stats["skipped_event_rows"]
        if skipped:
            LOGGER.warning(
                f"{skipped:,} row(s) had an event_type outside "
                f"{list(EVENT_TYPE_TO_RELATION.keys())} and have no declared "
                f"temporal edge type in this schema; they were not converted "
                f"into edges (see module docstring, assumption #1)."
            )

        self.stats["total_event_rows"] = total_rows

        self.edge_data = {rel: acc.finalize() for rel, acc in accumulators.items()}
        for rel, edge_payload in self.edge_data.items():
            LOGGER.info(
                f"[EDGES] {rel[0]}->{rel[1]}: "
                f"{edge_payload['edge_index'].shape[1]:,} temporal edges"
            )

    def _process_relation_subframe(
        self,
        sub: pd.DataFrame,
        src_type: str,
        dst_type: str,
        acc: EdgeAccumulator,
    ) -> None:
        cfg = self.config

        user_idx_all = self._encode_users(sub[cfg.user_id_col])
        valid_mask = user_idx_all != -1
        dropped = int((~valid_mask).sum())
        if dropped:
            self.stats["dropped_unknown_user_rows"] += dropped
            if not self._unknown_user_examples_logged:
                examples = sub.loc[~valid_mask, cfg.user_id_col].unique()[:5].tolist()
                LOGGER.warning(
                    f"[DIAG] Dropping rows with unknown user_id (not present in "
                    f"fused_features.csv). First examples seen: {examples}"
                )
                self._unknown_user_examples_logged = True
        if not valid_mask.any():
            return

        sub = sub.loc[valid_mask]
        user_idx = user_idx_all[valid_mask]

        if dst_type == "PC":
            dst_keys = sub[cfg.resource_col].fillna("UNKNOWN_PC").str.upper().str.strip()
            dst_idx = self.pc_registry.bulk_get_or_create(dst_keys)
        elif dst_type == "WebsiteDomain":
            domains = self._extract_domain(sub[cfg.resource_col])
            dst_idx = self.website_registry.bulk_get_or_create(domains)
        elif dst_type == "FileExtension":
            extensions = self._extract_extension(sub[cfg.resource_col])
            dst_idx = self.ext_registry.bulk_get_or_create(extensions)
        else:  # pragma: no cover - guarded by EVENT_TYPE_TO_RELATION contents
            raise ValueError(f"Unsupported destination node type: {dst_type}")

        resource_encoded = dst_idx  # resource *is* the destination node id here

        raw_timestamp, invalid_timestamps = _to_unix_seconds(sub[cfg.timestamp_col])
        if invalid_timestamps:
            self.stats["invalid_timestamp_rows"] += invalid_timestamps
        edge_time = raw_timestamp.copy()

        event_type_idx = self.event_type_registry.bulk_get_or_create(sub[cfg.event_type_col])
        event_index = sub["_row_idx"].to_numpy(dtype=np.int64)

        target_user_series = sub[cfg.target_user_col]
        has_target = target_user_series.notna().to_numpy()
        target_user_idx = np.full(len(sub), -1, dtype=np.int64)
        if has_target.any():
            normalized_targets = _normalize_id(target_user_series[has_target])
            target_user_idx[has_target] = self._encode_users(normalized_targets)

        acc.add_chunk(
            src=user_idx,
            dst=dst_idx,
            edge_time=edge_time,
            raw_timestamp=raw_timestamp,
            event_type=event_type_idx,
            event_index=event_index,
            resource=resource_encoded,
            target_user=target_user_idx,
        )

    def _encode_users(self, series: pd.Series) -> np.ndarray:
        """Vectorized user_id -> fixed registry index lookup. -1 = unknown user."""
        return series.map(self.uid_to_index).fillna(-1).astype(np.int64).to_numpy()

    @staticmethod
    def _extract_domain(resource: pd.Series) -> pd.Series:
        cleaned = resource.fillna("unknown_domain").str.lower().str.strip()
        extracted = cleaned.str.extract(_DOMAIN_RE, expand=False)
        return extracted.fillna(cleaned)

    @staticmethod
    def _extract_extension(resource: pd.Series) -> pd.Series:
        cleaned = resource.fillna("").str.lower().str.strip()
        ext = cleaned.str.extract(_EXTENSION_RE, expand=False)
        return ext.fillna("noext")

    # ------------------------------------------------------------------ #
    # Node feature construction for non-User node types
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_index_features(n: int) -> torch.Tensor:
        """Simple categorical-index feature: an embedding lookup table id."""
        return torch.arange(n, dtype=torch.long).unsqueeze(1)

    def _build_extension_features(self) -> torch.Tensor:
        n = len(self.ext_registry)
        idx = torch.arange(n, dtype=torch.float32)
        scores = torch.tensor(
            [self.extension_sensitivity.get(ext, 0.0) for ext in self.ext_registry.to_list()],
            dtype=torch.float32,
        )
        return torch.stack([idx, scores], dim=1)

    # ------------------------------------------------------------------ #
    # Graph assembly
    # ------------------------------------------------------------------ #

    def finalize_graph(self) -> HeteroData:
        LOGGER.info("Assembling HeteroData graph object")
        data = HeteroData()

        data["User"].x = self.user_features
        data["User"].num_nodes = len(self.uid_list)

        data["Department"].x = self._build_index_features(len(self.dept_registry))
        data["Department"].num_nodes = len(self.dept_registry)

        data["Role"].x = self._build_index_features(len(self.role_registry))
        data["Role"].num_nodes = len(self.role_registry)

        data["Team"].x = self._build_index_features(len(self.team_registry))
        data["Team"].num_nodes = len(self.team_registry)

        data["BusinessUnit"].x = self._build_index_features(len(self.bu_registry))
        data["BusinessUnit"].num_nodes = len(self.bu_registry)

        data["PC"].x = self._build_index_features(len(self.pc_registry))
        data["PC"].num_nodes = len(self.pc_registry)

        data["WebsiteDomain"].x = self._build_index_features(len(self.website_registry))
        data["WebsiteDomain"].num_nodes = len(self.website_registry)

        data["FileExtension"].x = self._build_extension_features()
        data["FileExtension"].num_nodes = len(self.ext_registry)

        for (src, rel, dst), ei in self.structural_edges.items():
            data[(src, rel, dst)].edge_index = ei

        for (src, dst), payload in self.edge_data.items():
            rel_name = TEMPORAL_RELATION_NAME[(src, dst)]
            store = data[(src, rel_name, dst)]
            store.edge_index = payload["edge_index"]
            store.edge_time = payload["edge_time"]
            store.raw_timestamp = payload["raw_timestamp"]
            store.event_type = payload["event_type"]
            store.event_index = payload["event_index"]
            store.resource = payload["resource"]
            store.target_user = payload["target_user"]

        self.graph = data
        LOGGER.info(str(data))
        return data

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate_graph(self) -> Dict[str, object]:
        LOGGER.info("Running graph validation suite")
        assert self.graph is not None, "finalize_graph() must run before validate_graph()"
        data = self.graph
        report: Dict[str, object] = {"passed": True, "issues": []}

        def flag(msg: str) -> None:
            report["issues"].append(msg)
            report["passed"] = False
            LOGGER.error(f"[VALIDATION] {msg}")

        # Duplicate node indices / missing IDs
        if len(set(self.uid_list)) != len(self.uid_list):
            flag("Duplicate user IDs detected in uid_list")

        for reg_name, reg in [
            ("dept_registry", self.dept_registry),
            ("role_registry", self.role_registry),
            ("team_registry", self.team_registry),
            ("bu_registry", self.bu_registry),
            ("pc_registry", self.pc_registry),
            ("domain_registry", self.website_registry),
            ("ext_registry", self.ext_registry),
        ]:
            keys = reg.to_list()
            if len(set(keys)) != len(keys):
                flag(f"Duplicate keys detected in {reg_name}")

        # Broken edge indices + empty edge types
        for edge_type in data.edge_types:
            src_type, _, dst_type = edge_type
            ei = data[edge_type].edge_index
            n_src = data[src_type].num_nodes
            n_dst = data[dst_type].num_nodes

            if ei.numel() == 0:
                flag(f"Edge type {edge_type} is empty")
                continue
            if ei.min().item() < 0:
                flag(f"Edge type {edge_type} has a negative node index")
            if ei[0].max().item() >= n_src:
                flag(f"Edge type {edge_type} has an out-of-range source index")
            if ei[1].max().item() >= n_dst:
                flag(f"Edge type {edge_type} has an out-of-range destination index")

        # NaN features
        for node_type in data.node_types:
            x = data[node_type].x
            if x is not None and torch.isnan(x).any():
                flag(f"NaN values found in node features for '{node_type}'")

        # Timestamp ordering within temporal relations
        for edge_type in data.edge_types:
            store = data[edge_type]
            if "edge_time" in store:
                et = store.edge_time
                if et.numel() > 1 and not torch.all(et[1:] >= et[:-1]):
                    flag(f"Edge type {edge_type} is not chronologically sorted")

        # Registry / feature-tensor row count consistency
        for node_type in data.node_types:
            n = data[node_type].num_nodes
            x = data[node_type].x
            if x is not None and x.shape[0] != n:
                flag(
                    f"Node type '{node_type}' feature row count "
                    f"({x.shape[0]}) does not match num_nodes ({n})"
                )

        if report["passed"]:
            LOGGER.info("[VALIDATION] All checks passed")
        else:
            LOGGER.warning(f"[VALIDATION] {len(report['issues'])} issue(s) found")

        return report

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save_artifacts(self, validation_report: Dict[str, object]) -> None:
        cfg = self.config
        assert self.graph is not None

        LOGGER.info(f"Saving graph to {cfg.graph_output_path}")
        torch.save(self.graph, cfg.graph_output_path)

        artifacts = {
            "user_index": self.uid_to_index,
            "uid_list": self.uid_list,
            "dept_registry": self.dept_registry.to_dict(),
            "role_registry": self.role_registry.to_dict(),
            "team_registry": self.team_registry.to_dict(),
            "bu_registry": self.bu_registry.to_dict(),
            "pc_registry": self.pc_registry.to_dict(),
            "domain_registry": self.website_registry.to_dict(),
            "ext_registry": self.ext_registry.to_dict(),
            "event_type_registry": self.event_type_registry.to_dict(),
            "uid_to_dept": self.uid_to_dept,
            "uid_to_role": self.uid_to_role,
            "uid_to_team": self.uid_to_team,
            "uid_to_bu": self.uid_to_bu,
            "uid_to_supervisor": self.uid_to_supervisor,
            "user_feature_columns": self.user_feature_columns,
            "stats": dict(self.stats),
            "validation_report": validation_report,
        }

        LOGGER.info(f"Saving preprocessing artifacts to {cfg.artifacts_output_path}")
        with open(cfg.artifacts_output_path, "wb") as f:
            pickle.dump(artifacts, f, protocol=pickle.HIGHEST_PROTOCOL)

        LOGGER.info("Save complete")

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #

    def run(self) -> HeteroData:
        t0 = time.time()
        LOGGER.info("=" * 78)
        LOGGER.info("Starting temporal heterogeneous event graph build")
        LOGGER.info("=" * 78)

        self.load_ldap()
        self.load_fused_features()
        self.load_file_sensitivity()
        log_memory("after loading static tables")

        self.build_structural_edges()
        self.process_events_streaming()
        log_memory("after streaming events")

        self.finalize_graph()
        validation_report = self.validate_graph()
        self.save_artifacts(validation_report)

        elapsed = time.time() - t0
        LOGGER.info("=" * 78)
        LOGGER.info("BUILD SUMMARY")
        LOGGER.info(f"  Users:                          {len(self.uid_list):,}")
        LOGGER.info(f"  Departments:                    {len(self.dept_registry):,}")
        LOGGER.info(f"  Roles:                          {len(self.role_registry):,}")
        LOGGER.info(f"  Teams:                          {len(self.team_registry):,}")
        LOGGER.info(f"  Business units:                 {len(self.bu_registry):,}")
        LOGGER.info(f"  PCs:                            {len(self.pc_registry):,}")
        LOGGER.info(f"  Website domains:                {len(self.website_registry):,}")
        LOGGER.info(f"  File extensions:                {len(self.ext_registry):,}")
        LOGGER.info(f"  Total event rows read:          {self.stats.get('total_event_rows', 0):,}")
        LOGGER.info(f"  Skipped (unmapped event_type):  {self.stats.get('skipped_event_rows', 0):,}")
        LOGGER.info(f"  Dropped (unknown user_id):      {self.stats.get('dropped_unknown_user_rows', 0):,}")
        LOGGER.info(f"  Invalid/unparseable timestamps: {self.stats.get('invalid_timestamp_rows', 0):,} (coerced to epoch 0)")
        assert self.graph is not None
        for edge_type in self.graph.edge_types:
            LOGGER.info(f"  Edges {edge_type}: {self.graph[edge_type].edge_index.shape[1]:,}")
        LOGGER.info(f"  Validation passed:               {validation_report['passed']}")
        LOGGER.info(f"  Total build time:                {elapsed:,.1f}s")
        LOGGER.info(f"  Graph saved to:                  {self.config.graph_output_path}")
        LOGGER.info(f"  Artifacts saved to:              {self.config.artifacts_output_path}")
        LOGGER.info("=" * 78)

        return self.graph


# ============================================================================
# CLI entry point
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an event-level temporal heterogeneous graph from CERT insider-threat logs."
    )
    parser.add_argument("--events", type=Path, default=Path("data/processed/unified_events.csv"))
    parser.add_argument("--fused-features", type=Path, default=Path("data/processed/fused_features.csv"))
    parser.add_argument("--file-sensitivity", type=Path, default=Path("data/processed/file_sensitivity.csv"))
    parser.add_argument("--ldap", type=Path, default=Path("data/processed/ldap.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("graph/output"))
    parser.add_argument("--chunk-size", type=int, default=500_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = GraphBuildConfig(
        events_path=args.events,
        fused_features_path=args.fused_features,
        file_sensitivity_path=args.file_sensitivity,
        ldap_path=args.ldap,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
    )

    global LOGGER
    LOGGER = setup_logging(config.log_path)

    builder = EventGraphBuilder(config)
    builder.run()


if __name__ == "__main__":
    main()