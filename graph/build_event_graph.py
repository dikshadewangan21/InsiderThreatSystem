"""
build_event_graph.py

Streaming, SHARD-BASED Event-level Temporal Heterogeneous Graph Builder
for CERT-style Insider Threat Detection pipelines.

--------------------------------------------------------------------------
WHY THIS IS A REWRITE, NOT A PATCH
--------------------------------------------------------------------------
The previous design used an `EdgeAccumulator` that appended every chunk's
numpy arrays to Python lists and only concatenated + sorted them once, at
the very end, in `finalize()`. That means RAM usage grows monotonically
with the *entire* dataset regardless of `--chunk-size`, and the final
`np.concatenate` + `np.argsort` over the full edge list is exactly where
100M+-row CERT exports blow past available RAM (ArrayMemoryError).

There is no chunk size that fixes that design — accumulate-then-sort-once
is fundamentally O(total_edges) in RAM by construction.

This rewrite removes accumulation entirely. Each chunk read from
`unified_events.csv` is converted straight into per-relation tensors,
sorted ONLY within that chunk, written to its own shard file on disk, and
then dropped from memory before the next chunk is read. Peak RAM is
O(chunk_size), forever, no matter how large the dataset is.

--------------------------------------------------------------------------
NEW ON-DISK LAYOUT (replaces the single heterogeneous_graph.pt)
--------------------------------------------------------------------------
graph/output/
  node_graph_skeleton.pt          <- HeteroData: node features + structural
                                      (non-temporal) edges ONLY. No event
                                      edges live here anymore. This is
                                      small (proportional to #users/#PCs/
                                      #domains/#extensions, never to
                                      #events) and safe to load whole.
  preprocessing_artifacts.pkl     <- registries, uid maps, stats, etc.
  edge_shard_manifest.json        <- authoritative index of every shard:
                                      relation, shard path, row count,
                                      min/max edge_time. Downstream code
                                      reads THIS, never globs the folder.
  edge_shards/
    User__accesses__PC/
      edge_shard_000000.pt
      edge_shard_000001.pt
      ...
    User__visits__WebsiteDomain/
      edge_shard_000000.pt
      ...
    User__touches__FileExtension/
      edge_shard_000000.pt
      ...

Each shard file is a plain dict of tensors for ONE relation, covering the
rows of ONE input chunk that mapped to that relation, sorted by
`edge_time` within that shard only (see "sorting semantics" below):

    {
        "edge_index":     LongTensor[2, E]
        "edge_time":      LongTensor[E]   (unix seconds, sorted within shard)
        "raw_timestamp":  LongTensor[E]
        "event_type":     LongTensor[E]
        "event_index":    LongTensor[E]   (row index in original CSV)
        "resource":       LongTensor[E]
        "target_user":    LongTensor[E]
        "src_node_type":  str
        "dst_node_type":  str
        "relation_name":  str
    }

--------------------------------------------------------------------------
SORTING SEMANTICS — READ THIS BEFORE WIRING UP TGN
--------------------------------------------------------------------------
Sorting only ever happens within a single chunk's rows for a single
relation (requirement: never materialize the full edge list to sort it
globally). That means:
  - Within one shard file, `edge_time` is guaranteed non-decreasing.
  - Across different shard files (even of the same relation), there is
    NO guarantee that shard N's timestamps precede shard N+1's, unless
    the source `unified_events.csv` itself is already roughly
    chronologically ordered (true of most CERT exports, since they are
    typically produced by a single time-ordered ETL pass — but this
    script does not assume or enforce it).

Two consumption modes are provided in `ShardedTemporalGraphReader` below:
  1. `iter_relation_shards()` / `iter_sequential_by_chunk()` — O(1) shards
     resident in memory at a time, correct chunk-local order, cheap. Use
     this for feature computation, statistics, or any TGN variant that
     tolerates approximately-chronological batches (most do, since real
     minibatches already violate strict global order at the boundaries).
  2. `iter_global_chronological()` — a true external k-way merge across
     ALL shards of ALL relations that yields edges in exact global time
     order, using `torch.load(..., mmap=True)` so shard tensors are
     memory-mapped rather than materialized; RAM cost is O(num_shards)
     small scalars for the heap, not O(num_edges). Use this only when an
     algorithm strictly requires global chronological order (e.g. a TGN
     memory module that is sensitive to cross-relation event ordering).
     It is slower (row-at-a-time Python loop) — batch its output.

--------------------------------------------------------------------------
DESIGN CONTRACT CARRIED OVER FROM THE ORIGINAL SPEC (unchanged)
--------------------------------------------------------------------------
- Every row of `unified_events.csv` whose `event_type` maps to a known
  temporal relation becomes EXACTLY ONE temporal edge (no aggregation).
- Structural relations (User-Department, User-Role, User-Team,
  User-BusinessUnit, User-User "works_with") are created once per user
  from the LDAP directory snapshot, since they are not events. They are
  node-scale, not event-scale, so they still live fully in RAM and in
  `node_graph_skeleton.pt`.
- User node features come verbatim from `fused_features.csv`. Never
  normalized, rescaled, or recomputed here.
- Timestamps are stored as raw Unix integers. No normalization.
- Output artifact keys/paths below are fixed and MUST NOT be renamed;
  `temporal_preprocessing.py` and `tgn.py` must be updated to consume
  `ShardedTemporalGraphReader` instead of loading one big `HeteroData`
  with event edges attached (see the bottom of this file for the
  before/after wiring note).

--------------------------------------------------------------------------
EXPLICIT ASSUMPTIONS (unchanged from original; still apply)
--------------------------------------------------------------------------
1. Only "logon"/"logoff"/"device" (-> PC), "http" (-> WebsiteDomain), and
   "file" (-> FileExtension) event_type values have a declared temporal
   edge type. Anything else (e.g. "email") is counted in
   `skipped_event_rows` and not converted into an edge.
2. `data/processed/ldap.csv` has columns: user_id, department, role,
   team, business_unit, supervisor.
3. A user_id present in events but absent from `fused_features.csv` is
   dropped and counted under `dropped_unknown_user_rows` (cannot assign
   a feature vector without violating the "never recompute" rule).
4. WebsiteDomain = scheme/"www."/path stripped from `resource`.
   FileExtension = substring after the final "." in `resource`.

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
import heapq
import itertools
import json
import logging
import os
import pickle
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

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
# Shared parsing / normalization helpers (unchanged — these were correct)
# ============================================================================

def _to_unix_seconds(series: pd.Series) -> Tuple[np.ndarray, int]:
    """
    Convert timestamp strings to Unix epoch seconds (int64), resolution-
    independent. See original module notes: casting the parsed datetimes
    to datetime64[s] before viewing as int64 avoids the classic silent
    ns/us/s scale bug. errors="coerce" -> invalid rows become epoch 0 and
    are counted, never raise mid-stream.
    """
    parsed = pd.to_datetime(series, errors="coerce")
    invalid = int(parsed.isna().sum())
    is_na = parsed.isna().to_numpy()
    seconds = parsed.astype("datetime64[s]").to_numpy().view("int64")
    seconds = np.where(is_na, 0, seconds).astype(np.int64)
    return seconds, invalid


_DOMAIN_RE = re.compile(r"^(?:https?://)?(?:www\.)?([^/:?#]+)")
_EXTENSION_RE = re.compile(r"\.([a-z0-9]{1,10})$")


def _normalize_id(series: pd.Series) -> pd.Series:
    """
    Canonical join-key normalization for user_id / PC ids / etc. Must be
    applied identically to every source file that participates in a join.
    Handles numeric-vs-string dtype drift, case drift, whitespace/quote
    noise, and a stray trailing ".0" from upstream float coercion. Does
    NOT touch leading zeros (unsafe to guess).
    """
    s = series.astype(str).str.strip()
    s = s.str.replace(r'^["\']|["\']$', "", regex=True)
    s = s.str.replace(r"\.0$", "", regex=True)
    return s.str.upper()


# ============================================================================
# Registries — node-scale, not event-scale, so they stay resident in RAM
# for the whole streaming pass. This is fine: #PCs/#domains/#extensions
# is bounded by the org's infrastructure, not by #events.
# ============================================================================

class Registry:
    """Incrementally-growable string -> contiguous integer index registry."""

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
    skeleton_output_path: Path = field(init=False)
    artifacts_output_path: Path = field(init=False)
    manifest_path: Path = field(init=False)
    shard_root: Path = field(init=False)
    log_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.skeleton_output_path = self.output_dir / "node_graph_skeleton.pt"
        self.artifacts_output_path = self.output_dir / "preprocessing_artifacts.pkl"
        self.manifest_path = self.output_dir / "edge_shard_manifest.json"
        self.shard_root = self.output_dir / "edge_shards"
        self.shard_root.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "build_event_graph.log"


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


def _relation_dirname(relation: Tuple[str, str]) -> str:
    src, dst = relation
    rel_name = TEMPORAL_RELATION_NAME[relation]
    return f"{src}__{rel_name}__{dst}"


# ============================================================================
# Shard manifest — the authoritative, small, JSON-serializable index of
# every shard on disk. Downstream code reads this; it never globs
# directories or infers shard existence from filenames.
# ============================================================================

@dataclass
class ShardMeta:
    relation: str          # e.g. "User__accesses__PC"
    src_type: str
    dst_type: str
    shard_index: int
    path: str
    num_edges: int
    min_edge_time: Optional[int]
    max_edge_time: Optional[int]
    source_chunk_index: int  # which input CSV chunk this shard came from


class EdgeShardManifest:
    """
    Tracks every shard written during the build and (de)serializes to a
    single small JSON file. This file is O(num_shards), never
    O(num_edges) — safe to hold fully in RAM and re-load anywhere.
    """

    def __init__(self) -> None:
        self.shards: List[ShardMeta] = []

    def add(self, meta: ShardMeta) -> None:
        self.shards.append(meta)

    def by_relation(self) -> Dict[str, List[ShardMeta]]:
        out: Dict[str, List[ShardMeta]] = defaultdict(list)
        for m in self.shards:
            out[m.relation].append(m)
        for rel in out:
            out[rel].sort(key=lambda m: m.shard_index)
        return dict(out)

    def total_edges(self, relation: Optional[str] = None) -> int:
        if relation is None:
            return sum(m.num_edges for m in self.shards)
        return sum(m.num_edges for m in self.shards if m.relation == relation)

    def save(self, path: Path) -> None:
        payload = {"shards": [asdict(m) for m in self.shards]}
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "EdgeShardManifest":
        with open(path, "r") as f:
            payload = json.load(f)
        manifest = cls()
        for row in payload["shards"]:
            manifest.add(ShardMeta(**row))
        return manifest


# ============================================================================
# Builder
# ============================================================================

class EventGraphBuilder:
    """
    Orchestrates the CERT unified_events.csv -> sharded temporal graph
    pipeline:
      1. Load static tables (LDAP, fused features, sensitivity) — small,
         resident in RAM for the whole run.
      2. Build structural edges once (node-scale).
      3. Stream event rows chunk by chunk. For each chunk: split by
         relation, encode straight to tensors, sort ONLY within that
         chunk, write ONE shard file per relation per chunk, then
         explicitly drop and gc.collect() before the next chunk is read.
         Nothing accumulates across chunks except the (small) registries
         and (small) scalar stats counters.
      4. Persist the node/structural skeleton, the shard manifest, and
         the preprocessing artifacts.
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
        self.skeleton_graph: Optional[HeteroData] = None

        # Shard bookkeeping — O(num_shards), never O(num_edges).
        self.manifest = EdgeShardManifest()
        self._shard_counters: Dict[Tuple[str, str], int] = defaultdict(int)
        for relation in EVENT_TYPE_TO_RELATION.values():
            (self.config.shard_root / _relation_dirname(relation)).mkdir(
                parents=True, exist_ok=True
            )

        self.stats: Dict[str, int] = defaultdict(int)
        self._unknown_user_examples_logged = False

    # ------------------------------------------------------------------ #
    # Static table loading (unchanged — these are node-scale, not
    # event-scale, so keeping them fully in RAM was never the problem)
    # ------------------------------------------------------------------ #

    def load_ldap(self) -> None:
        cfg = self.config
        LOGGER.info(f"Loading LDAP directory snapshot from {cfg.ldap_path}")
        df = pd.read_csv(cfg.ldap_path)

        raw_dtype = df[cfg.user_id_col].dtype
        raw_samples = df[cfg.user_id_col].astype(str).head(3).tolist()
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
    # Structural (non-temporal) edges - built once from LDAP, node-scale
    # ------------------------------------------------------------------ #

    def build_structural_edges(self) -> None:
        LOGGER.info("Building structural edges from LDAP (each exists exactly once per user)")

        dept_src, dept_dst = [], []
        role_src, role_dst = [], []
        team_src, team_dst = [], []
        bu_src, bu_dst = [], []

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
    # Temporal edges — TRUE streaming: one shard written per relation
    # per chunk, nothing retained across chunks.
    # ------------------------------------------------------------------ #

    def process_events_streaming(self) -> None:
        cfg = self.config
        LOGGER.info(f"Streaming events from {cfg.events_path} in chunks of {cfg.chunk_size:,} rows")
        LOGGER.info(
            "Each chunk is converted to per-relation shards and written to disk "
            "immediately; no edge data is retained across chunk boundaries."
        )

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
            chunk[cfg.user_id_col] = _normalize_id(chunk[cfg.user_id_col])

            chunk["_row_idx"] = np.arange(
                global_row_offset, global_row_offset + chunk_rows, dtype=np.int64
            )
            global_row_offset += chunk_rows

            if chunk_idx == 0:
                unique_types = sorted(chunk[cfg.event_type_col].dropna().unique().tolist())
                LOGGER.info(f"[DIAG] unique event_type values in first chunk: {unique_types}")
                unmatched = chunk.loc[
                    ~chunk[cfg.user_id_col].isin(self.uid_to_index.keys()), cfg.user_id_col
                ].unique()
                if len(unmatched) > 0:
                    LOGGER.warning(
                        f"[DIAG] {len(unmatched):,} distinct user_id value(s) in the first "
                        f"chunk do not exist in fused_features.csv. Examples: {list(unmatched[:5])}"
                    )

            mapped_mask = chunk[cfg.event_type_col].isin(EVENT_TYPE_TO_RELATION.keys())
            self.stats["skipped_event_rows"] += int((~mapped_mask).sum())

            for event_key, relation in EVENT_TYPE_TO_RELATION.items():
                key_mask = chunk[cfg.event_type_col] == event_key
                if not key_mask.any():
                    continue
                sub = chunk.loc[key_mask]
                payload = self._encode_relation_subframe(sub, relation)
                del sub
                if payload is None:
                    continue
                # Sort ONLY within this chunk's rows for this relation,
                # then write immediately and drop from memory.
                self._write_shard(relation, payload, source_chunk_index=chunk_idx)
                del payload

            if chunk_idx % 5 == 0:
                elapsed = time.time() - t_start
                rate = total_rows / elapsed if elapsed > 0 else 0.0
                LOGGER.info(
                    f"[STREAM] chunk={chunk_idx:,} rows_so_far={total_rows:,} "
                    f"rate={rate:,.0f} rows/sec shards_written={len(self.manifest.shards):,}"
                )
                log_memory(f"after chunk {chunk_idx}")

            # Nothing from this chunk survives past this point.
            del chunk
            gc.collect()

        elapsed = time.time() - t_start
        LOGGER.info(f"Finished streaming {total_rows:,} rows in {elapsed:,.1f}s")

        skipped = self.stats["skipped_event_rows"]
        if skipped:
            LOGGER.warning(
                f"{skipped:,} row(s) had an event_type outside "
                f"{list(EVENT_TYPE_TO_RELATION.keys())} and were not converted "
                f"into edges (see module docstring, assumption #1)."
            )

        self.stats["total_event_rows"] = total_rows
        self.stats["total_shards_written"] = len(self.manifest.shards)
        self.stats["total_edges_written"] = self.manifest.total_edges()

        for relation, shards in self.manifest.by_relation().items():
            LOGGER.info(
                f"[EDGES] {relation}: {sum(s.num_edges for s in shards):,} temporal edges "
                f"across {len(shards):,} shard(s)"
            )

    def _encode_relation_subframe(
        self, sub: pd.DataFrame, relation: Tuple[str, str]
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Encode one chunk's rows for one relation into raw numpy arrays.
        Returns None if every row in `sub` had to be dropped (unknown
        user). Does NOT sort and does NOT write to disk — that happens
        in `_write_shard`, kept separate so this stays a pure, testable
        encode step.
        """
        cfg = self.config
        src_type, dst_type = relation

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
            return None

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
        else:  # pragma: no cover
            raise ValueError(f"Unsupported destination node type: {dst_type}")

        raw_timestamp, invalid_timestamps = _to_unix_seconds(sub[cfg.timestamp_col])
        if invalid_timestamps:
            self.stats["invalid_timestamp_rows"] += invalid_timestamps

        event_type_idx = self.event_type_registry.bulk_get_or_create(sub[cfg.event_type_col])
        event_index = sub["_row_idx"].to_numpy(dtype=np.int64)

        target_user_series = sub[cfg.target_user_col]
        has_target = target_user_series.notna().to_numpy()
        target_user_idx = np.full(len(sub), -1, dtype=np.int64)
        if has_target.any():
            normalized_targets = _normalize_id(target_user_series[has_target])
            target_user_idx[has_target] = self._encode_users(normalized_targets)

        return {
            "src": user_idx,
            "dst": dst_idx,
            "edge_time": raw_timestamp,        # edge_time == raw_timestamp (seconds, unnormalized)
            "raw_timestamp": raw_timestamp.copy(),
            "event_type": event_type_idx,
            "event_index": event_index,
            "resource": dst_idx,               # resource *is* the destination node id here
            "target_user": target_user_idx,
        }

    def _write_shard(
        self,
        relation: Tuple[str, str],
        payload: Dict[str, np.ndarray],
        source_chunk_index: int,
    ) -> ShardMeta:
        """
        Sort within this chunk's rows only, convert to tensors, write to
        its own shard file, and record it in the manifest. Nothing here
        is retained by `self` beyond the small `ShardMeta` and the
        per-relation shard counter.
        """
        src_type, dst_type = relation
        rel_name = _relation_dirname(relation)

        order = np.argsort(payload["edge_time"], kind="stable")
        num_edges = int(order.shape[0])

        edge_index = np.stack([payload["src"][order], payload["dst"][order]], axis=0)
        tensors = {
            "edge_index": torch.from_numpy(edge_index).long(),
            "edge_time": torch.from_numpy(payload["edge_time"][order]).long(),
            "raw_timestamp": torch.from_numpy(payload["raw_timestamp"][order]).long(),
            "event_type": torch.from_numpy(payload["event_type"][order]).long(),
            "event_index": torch.from_numpy(payload["event_index"][order]).long(),
            "resource": torch.from_numpy(payload["resource"][order]).long(),
            "target_user": torch.from_numpy(payload["target_user"][order]).long(),
            "src_node_type": src_type,
            "dst_node_type": dst_type,
            "relation_name": TEMPORAL_RELATION_NAME[relation],
        }

        shard_idx = self._shard_counters[relation]
        self._shard_counters[relation] += 1
        shard_path = self.config.shard_root / rel_name / f"edge_shard_{shard_idx:06d}.pt"
        torch.save(tensors, shard_path)

        meta = ShardMeta(
            relation=rel_name,
            src_type=src_type,
            dst_type=dst_type,
            shard_index=shard_idx,
            path=str(shard_path),
            num_edges=num_edges,
            min_edge_time=int(tensors["edge_time"][0].item()) if num_edges else None,
            max_edge_time=int(tensors["edge_time"][-1].item()) if num_edges else None,
            source_chunk_index=source_chunk_index,
        )
        self.manifest.add(meta)

        del tensors, edge_index, order
        return meta

    def _encode_users(self, series: pd.Series) -> np.ndarray:
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
    # Node feature construction for non-User node types (node-scale)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_index_features(n: int) -> torch.Tensor:
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
    # Skeleton graph assembly — nodes + structural edges ONLY. Event
    # edges never touch this object; they live purely as shards.
    # ------------------------------------------------------------------ #

    def finalize_skeleton_graph(self) -> HeteroData:
        LOGGER.info("Assembling node/structural skeleton graph (no event edges attached)")
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

        self.skeleton_graph = data
        LOGGER.info(str(data))
        return data

    # ------------------------------------------------------------------ #
    # Validation — streams shard-by-shard, never loads all shards at once
    # ------------------------------------------------------------------ #

    def validate_skeleton(self) -> Dict[str, object]:
        LOGGER.info("Validating node/structural skeleton")
        assert self.skeleton_graph is not None
        data = self.skeleton_graph
        report: Dict[str, object] = {"passed": True, "issues": []}

        def flag(msg: str) -> None:
            report["issues"].append(msg)
            report["passed"] = False
            LOGGER.error(f"[VALIDATION] {msg}")

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

        for edge_type in data.edge_types:
            src_type, _, dst_type = edge_type
            ei = data[edge_type].edge_index
            n_src = data[src_type].num_nodes
            n_dst = data[dst_type].num_nodes
            if ei.numel() == 0:
                flag(f"Structural edge type {edge_type} is empty")
                continue
            if ei.min().item() < 0:
                flag(f"Structural edge type {edge_type} has a negative node index")
            if ei[0].max().item() >= n_src:
                flag(f"Structural edge type {edge_type} has an out-of-range source index")
            if ei[1].max().item() >= n_dst:
                flag(f"Structural edge type {edge_type} has an out-of-range destination index")

        for node_type in data.node_types:
            x = data[node_type].x
            if x is not None and torch.isnan(x).any():
                flag(f"NaN values found in node features for '{node_type}'")
            n = data[node_type].num_nodes
            if x is not None and x.shape[0] != n:
                flag(
                    f"Node type '{node_type}' feature row count ({x.shape[0]}) "
                    f"does not match num_nodes ({n})"
                )

        if report["passed"]:
            LOGGER.info("[VALIDATION] Skeleton checks passed")
        else:
            LOGGER.warning(f"[VALIDATION] {len(report['issues'])} skeleton issue(s) found")

        return report

    def validate_shards(self) -> Dict[str, object]:
        """
        Streaming validation: loads ONE shard at a time (bounded by
        chunk_size, same memory ceiling as the build itself), checks
        index bounds against node counts and within-shard chronological
        order, then drops it before moving to the next.
        """
        LOGGER.info("Validating edge shards (streaming, one shard at a time)")
        report: Dict[str, object] = {"passed": True, "issues": [], "per_relation_edges": {}}

        def flag(msg: str) -> None:
            report["issues"].append(msg)
            report["passed"] = False
            LOGGER.error(f"[VALIDATION] {msg}")

        node_counts = {
            "User": len(self.uid_list),
            "PC": len(self.pc_registry),
            "WebsiteDomain": len(self.website_registry),
            "FileExtension": len(self.ext_registry),
        }

        for relation, shards in self.manifest.by_relation().items():
            total = 0
            for meta in shards:
                tensors = torch.load(meta.path, map_location="cpu")
                ei = tensors["edge_index"]
                n_src = node_counts[tensors["src_node_type"]]
                n_dst = node_counts[tensors["dst_node_type"]]

                if ei.numel() > 0:
                    if ei.min().item() < 0:
                        flag(f"Shard {meta.path} has a negative node index")
                    if ei[0].max().item() >= n_src:
                        flag(f"Shard {meta.path} has an out-of-range source index")
                    if ei[1].max().item() >= n_dst:
                        flag(f"Shard {meta.path} has an out-of-range destination index")

                et = tensors["edge_time"]
                if et.numel() > 1 and not torch.all(et[1:] >= et[:-1]):
                    flag(f"Shard {meta.path} is not chronologically sorted within itself")

                total += ei.shape[1]
                del tensors
            report["per_relation_edges"][relation] = total

        gc.collect()

        if report["passed"]:
            LOGGER.info("[VALIDATION] All shard checks passed")
        else:
            LOGGER.warning(f"[VALIDATION] {len(report['issues'])} shard issue(s) found")

        return report

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save_artifacts(self, skeleton_report: Dict[str, object], shard_report: Dict[str, object]) -> None:
        cfg = self.config
        assert self.skeleton_graph is not None

        LOGGER.info(f"Saving node/structural skeleton to {cfg.skeleton_output_path}")
        torch.save(self.skeleton_graph, cfg.skeleton_output_path)

        LOGGER.info(f"Saving edge shard manifest ({len(self.manifest.shards):,} shards) to {cfg.manifest_path}")
        self.manifest.save(cfg.manifest_path)

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
            "skeleton_validation_report": skeleton_report,
            "shard_validation_report": shard_report,
        }

        LOGGER.info(f"Saving preprocessing artifacts to {cfg.artifacts_output_path}")
        with open(cfg.artifacts_output_path, "wb") as f:
            pickle.dump(artifacts, f, protocol=pickle.HIGHEST_PROTOCOL)

        LOGGER.info("Save complete")

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        t0 = time.time()
        LOGGER.info("=" * 78)
        LOGGER.info("Starting streaming, shard-based temporal event graph build")
        LOGGER.info("=" * 78)

        self.load_ldap()
        self.load_fused_features()
        self.load_file_sensitivity()
        log_memory("after loading static tables")

        self.build_structural_edges()
        self.process_events_streaming()
        log_memory("after streaming events (shards on disk, nothing accumulated)")

        self.finalize_skeleton_graph()
        skeleton_report = self.validate_skeleton()
        shard_report = self.validate_shards()
        self.save_artifacts(skeleton_report, shard_report)

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
        LOGGER.info(f"  Total shards written:           {self.stats.get('total_shards_written', 0):,}")
        for relation, shards in self.manifest.by_relation().items():
            LOGGER.info(f"  Edges {relation}: {sum(s.num_edges for s in shards):,} across {len(shards):,} shard(s)")
        LOGGER.info(f"  Skeleton validation passed:      {skeleton_report['passed']}")
        LOGGER.info(f"  Shard validation passed:         {shard_report['passed']}")
        LOGGER.info(f"  Total build time:                {elapsed:,.1f}s")
        LOGGER.info(f"  Skeleton graph saved to:         {self.config.skeleton_output_path}")
        LOGGER.info(f"  Shard manifest saved to:         {self.config.manifest_path}")
        LOGGER.info(f"  Artifacts saved to:              {self.config.artifacts_output_path}")
        LOGGER.info("=" * 78)


# ============================================================================
# Downstream reader — this is what temporal_preprocessing.py / tgn.py must
# use instead of `torch.load(heterogeneous_graph.pt)`.
# ============================================================================

class ShardedTemporalGraphReader:
    """
    Read-side companion to the builder above. Loads the small skeleton
    graph, the small manifest, and then streams edge shards on demand —
    never all at once.

    Typical TGN wiring:

        reader = ShardedTemporalGraphReader(output_dir)
        skeleton = reader.load_skeleton()          # nodes + structural edges
        artifacts = reader.load_artifacts()         # registries, uid maps

        model = build_tgn(skeleton, artifacts)

        for relation in reader.manifest.by_relation():
            for tensors in reader.iter_relation_shards(relation):
                model.process_temporal_batch(tensors)   # update memory, etc.
                del tensors   # shard is freed as soon as the caller drops it

    Or, when a caller genuinely needs exact global chronological order
    across relations (not just per-relation order), use
    `iter_global_chronological()` instead of nesting the loop above.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.manifest = EdgeShardManifest.load(self.output_dir / "edge_shard_manifest.json")

    def load_skeleton(self) -> HeteroData:
        return torch.load(self.output_dir / "node_graph_skeleton.pt", weights_only=False)

    def load_artifacts(self) -> Dict[str, object]:
        with open(self.output_dir / "preprocessing_artifacts.pkl", "rb") as f:
            return pickle.load(f)

    def iter_relation_shards(self, relation: str) -> Iterator[Dict[str, object]]:
        """
        Yield one relation's shards in shard-index (== chunk-arrival)
        order. Exactly one shard resident in memory at a time.
        """
        shards = self.manifest.by_relation().get(relation, [])
        for meta in shards:
            tensors = torch.load(meta.path, map_location="cpu")
            yield tensors
            del tensors

    def iter_sequential_by_chunk(self) -> Iterator[Tuple[str, Dict[str, object]]]:
        """
        Yield (relation, tensors) pairs ordered by the source chunk index
        they were derived from, then by relation name within a chunk.
        This preserves the original CSV's row order at chunk granularity
        across all relations — a good default for CERT exports, which
        are typically already close to chronological end-to-end. It is
        NOT a guarantee of exact global chronological order (see the
        module docstring); use `iter_global_chronological()` if that is
        required.
        """
        by_chunk: Dict[int, List[ShardMeta]] = defaultdict(list)
        for m in self.manifest.shards:
            by_chunk[m.source_chunk_index].append(m)

        for chunk_idx in sorted(by_chunk.keys()):
            for meta in sorted(by_chunk[chunk_idx], key=lambda m: m.relation):
                tensors = torch.load(meta.path, map_location="cpu")
                yield meta.relation, tensors
                del tensors

    def iter_global_chronological(self) -> Iterator[Dict[str, object]]:
        """
        True external k-way merge across every shard of every relation,
        yielding individual edges in exact global `edge_time` order.

        Memory behavior: every shard is opened with
        `torch.load(path, mmap=True)`, which memory-maps tensor storage
        instead of materializing it — the heap only ever holds one
        scalar timestamp per open shard, and the OS pages in the rest on
        demand. This scales to arbitrarily many shards, unlike loading
        every shard fully (which would reproduce the original OOM at
        read time instead of write time).

        This is a row-at-a-time Python generator, so it is CPU-slower
        than `iter_relation_shards` — batch its output for anything
        performance sensitive:

            batch = []
            for edge in reader.iter_global_chronological():
                batch.append(edge)
                if len(batch) == 100_000:
                    process(batch); batch = []
        """
        heap: List[Tuple[int, int, int]] = []  # (edge_time, handle_id, cursor)
        handles: Dict[int, Dict[str, object]] = {}

        handle_id = 0
        for meta in self.manifest.shards:
            if meta.num_edges == 0:
                continue
            try:
                tensors = torch.load(meta.path, map_location="cpu", mmap=True)
            except TypeError:
                # Older torch without mmap kwarg support — fall back to a
                # normal load. Still correct, just not memory-mapped.
                LOGGER.warning(
                    "torch.load(mmap=True) unsupported by this torch version; "
                    "falling back to full shard loads for global chronological "
                    "merge (requires torch>=2.1 for the memory-mapped path)."
                )
                tensors = torch.load(meta.path, map_location="cpu")
            handles[handle_id] = {"meta": meta, "tensors": tensors, "cursor": 0}
            first_time = int(tensors["edge_time"][0].item())
            heapq.heappush(heap, (first_time, handle_id, 0))
            handle_id += 1

        while heap:
            _, hid, cursor = heapq.heappop(heap)
            h = handles[hid]
            tensors = h["tensors"]

            yield {
                "src_node_type": tensors["src_node_type"],
                "dst_node_type": tensors["dst_node_type"],
                "relation_name": tensors["relation_name"],
                "src": int(tensors["edge_index"][0, cursor].item()),
                "dst": int(tensors["edge_index"][1, cursor].item()),
                "edge_time": int(tensors["edge_time"][cursor].item()),
                "raw_timestamp": int(tensors["raw_timestamp"][cursor].item()),
                "event_type": int(tensors["event_type"][cursor].item()),
                "event_index": int(tensors["event_index"][cursor].item()),
                "resource": int(tensors["resource"][cursor].item()),
                "target_user": int(tensors["target_user"][cursor].item()),
            }

            next_cursor = cursor + 1
            if next_cursor < tensors["edge_time"].shape[0]:
                h["cursor"] = next_cursor
                next_time = int(tensors["edge_time"][next_cursor].item())
                heapq.heappush(heap, (next_time, hid, next_cursor))
            else:
                # Shard exhausted — release it.
                del handles[hid]["tensors"]
                del handles[hid]


# ============================================================================
# CLI entry point
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a sharded, streaming temporal heterogeneous graph from CERT insider-threat logs."
    )
    parser.add_argument("--events", type=Path, default=Path("data/processed/unified_events.csv"))
    parser.add_argument("--fused-features", type=Path, default=Path("data/processed/fused_features.csv"))
    parser.add_argument("--file-sensitivity", type=Path, default=Path("data/processed/file_sensitivity.csv"))
    parser.add_argument("--ldap", type=Path, default=Path("data/raw/LDAP.csv"))
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