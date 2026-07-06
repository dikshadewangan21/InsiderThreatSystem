"""
graph/edge_features.py
======================
Production streaming edge feature engine for the CERT insider-threat
temporal heterogeneous graph pipeline.

Two-pass design — memory ceiling O(chunk_size) + O(num_users):

  Pass 1  Stream every shard in source-chunk order (approx. chronological).
          Compute per-user temporal state (delta_t, session_duration,
          rolling_count, interaction_count) and write one small .tmp.pt
          per shard.  State dicts are O(num_users) — never O(num_edges).

  Pass 2  Stream every shard again.  Load shard + its .tmp.pt, compute all
          40 scalar features + 16-dim sinusoidal temporal encoding, validate,
          write output feature shard, delete .tmp.pt, free memory.

Run:
    python graph/edge_features.py \
        --graph-output-dir graph/output \
        --psychology-csv   data/processed/psychology_features.csv \
        --fused-csv        data/processed/fused_features.csv \
        --behavior-csv     data/processed/behavior_features.csv
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import pickle
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(log_path: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger("edge_features")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


LOGGER = logging.getLogger("edge_features")


def _log_mem(tag: str) -> None:
    if _HAS_PSUTIL:
        mb = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
        LOGGER.info("[MEM] %s: %.1f MB RSS", tag, mb)

# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------

FEATURE_NAMES: List[str] = [
    # Temporal (0-10)
    "hour_of_day", "minute_of_hour", "weekday", "weekend_flag",
    "month", "quarter", "is_after_hours", "is_working_hours",
    "time_since_last_event", "session_duration", "time_gap_prev_action",
    # Behavior (11-17)
    "user_behavior_score", "anomaly_score", "activity_frequency",
    "rolling_action_count", "login_frequency", "device_frequency",
    "website_frequency",
    # Psychological (18-20)
    "psychology_score", "behavior_deviation", "after_hours_score",
    # Organizational (21-25)
    "dept_encoded", "role_encoded", "team_encoded", "bu_encoded",
    "has_manager",
    # Resource (26-29)
    "file_sensitivity", "domain_popularity", "pc_popularity", "extension_risk",
    # Graph (30-32)
    "source_degree", "destination_degree", "historical_interaction_count",
    # Event type one-hot (33-35)
    "is_logon_event", "is_device_event", "is_http_event",
    # Relation type one-hot (36-38)
    "rel_accesses_pc", "rel_visits_web", "rel_touches_file",
    # Target-user flag (39)
    "has_target_user",
]
NUM_SCALAR: int = len(FEATURE_NAMES)   # 40
TEMPORAL_DIM: int = 16
FEATURE_VERSION: str = "1.0.0"

_FI: Dict[str, int] = {name: i for i, name in enumerate(FEATURE_NAMES)}

SESSION_GAP_S: int = 8 * 3600                    # 8 h → new session
_LOG_DENOM: float = float(np.log1p(365 * 24 * 3600))   # ~1 year in seconds

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EdgeFeatureConfig:
    graph_output_dir: Path = Path("graph/output")
    psychology_csv:   Path = Path("data/processed/psychology_features.csv")
    fused_csv:        Path = Path("data/processed/fused_features.csv")
    behavior_csv:     Path = Path("data/processed/behavior_features.csv")
    temporal_encoding_dim: int   = TEMPORAL_DIM
    temporal_max_period:   float = 1.0e6

    # Derived — set in __post_init__
    manifest_path:    Path = field(init=False)
    skeleton_path:    Path = field(init=False)
    artifacts_path:   Path = field(init=False)
    output_shard_dir: Path = field(init=False)
    tmp_dir:          Path = field(init=False)
    feature_manifest: Path = field(init=False)
    log_path:         Path = field(init=False)

    def __post_init__(self) -> None:
        d = Path(self.graph_output_dir)
        self.manifest_path    = d / "edge_shard_manifest.json"
        self.skeleton_path    = d / "node_graph_skeleton.pt"
        self.artifacts_path   = d / "preprocessing_artifacts.pkl"
        self.output_shard_dir = d / "edge_feature_shards"
        self.tmp_dir          = self.output_shard_dir / ".tmp"
        self.feature_manifest = d / "feature_manifest.json"
        self.log_path         = d / "edge_features.log"
        self.output_shard_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _resolve_shard_path(raw: str, graph_output_dir: Path) -> Path:
    """Resolve a shard path that may be absolute or relative."""
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p
    # relative to cwd
    if p.exists():
        return p.resolve()
    # relative to project root (parent of graph/output)
    candidate = graph_output_dir.parent / p
    if candidate.exists():
        return candidate
    # last resort: as-is
    return p


def _tmp_path(cfg: EdgeFeatureConfig, relation: str, shard_index: int) -> Path:
    safe = relation.replace("\\", "_").replace("/", "_")
    return cfg.tmp_dir / f"{safe}_{shard_index:06d}.tmp.pt"


def _feature_shard_path(cfg: EdgeFeatureConfig, relation: str, shard_index: int) -> Path:
    rel_dir = cfg.output_shard_dir / relation
    rel_dir.mkdir(parents=True, exist_ok=True)
    return rel_dir / f"edge_features_{shard_index:06d}.pt"

# ---------------------------------------------------------------------------
# Static lookup tables (loaded once, held in RAM for the whole run)
# ---------------------------------------------------------------------------

@dataclass
class StaticLookups:
    uid_list:          List[str]
    uid_to_index:      Dict[str, int]
    uid_to_dept:       Dict[str, str]
    uid_to_role:       Dict[str, str]
    uid_to_team:       Dict[str, str]
    uid_to_bu:         Dict[str, str]
    uid_to_supervisor: Dict[str, str]
    dept_registry:     Dict[str, int]
    role_registry:     Dict[str, int]
    team_registry:     Dict[str, int]
    bu_registry:       Dict[str, int]
    pc_registry:       Dict[str, int]
    domain_registry:   Dict[str, int]
    ext_registry:      Dict[str, int]
    event_type_registry: Dict[str, int]

    # Per-user float arrays indexed by user_idx (length = num_users)
    user_behavior_score: np.ndarray
    user_anomaly_score:  np.ndarray
    user_psychology:     np.ndarray
    user_behav_dev:      np.ndarray
    user_aft_hrs:        np.ndarray
    user_activity_freq:  np.ndarray
    user_login_freq:     np.ndarray
    user_device_freq:    np.ndarray
    user_website_freq:   np.ndarray
    user_out_degree:     np.ndarray   # log1p-scaled, normalized

    # Per-extension sensitivity indexed by ext_idx
    ext_sensitivity: np.ndarray

    max_pc_idx:     int
    max_domain_idx: int
    max_ext_idx:    int
    max_dept_id:    int
    max_role_id:    int
    max_team_id:    int
    max_bu_id:      int


def _pick(df: pd.DataFrame, candidates: List[str], default: float = 0.0) -> np.ndarray:
    for c in candidates:
        if c in df.columns:
            return df[c].fillna(default).to_numpy(dtype=np.float32)
    return np.full(len(df), default, dtype=np.float32)


def _user_arr(uid_list: List[str], df: pd.DataFrame,
              candidates: List[str], default: float = 0.0) -> np.ndarray:
    """Build per-user float32 array from a DataFrame indexed by upper-case user_id."""
    n = len(uid_list)
    out = np.full(n, default, dtype=np.float32)
    col = next((c for c in candidates if c in df.columns), None)
    if col is None:
        return out
    for i, uid in enumerate(uid_list):
        if uid in df.index:
            v = df.at[uid, col]
            out[i] = float(v) if pd.notna(v) else default
    return out


def load_static_lookups(cfg: EdgeFeatureConfig) -> StaticLookups:
    LOGGER.info("Loading preprocessing_artifacts.pkl …")
    with open(cfg.artifacts_path, "rb") as f:
        art = pickle.load(f)

    uid_list: List[str]      = art["uid_list"]
    uid_to_index             = art["user_index"]
    n_users                  = len(uid_list)
    dept_reg: Dict[str, int] = art["dept_registry"]
    role_reg: Dict[str, int] = art["role_registry"]
    team_reg: Dict[str, int] = art["team_registry"]
    bu_reg:   Dict[str, int] = art["bu_registry"]
    pc_reg:   Dict[str, int] = art["pc_registry"]
    dom_reg:  Dict[str, int] = art["domain_registry"]
    ext_reg:  Dict[str, int] = art["ext_registry"]

    # Skeleton: user out-degree from works_with edges
    LOGGER.info("Loading node_graph_skeleton.pt …")
    skeleton = torch.load(cfg.skeleton_path, weights_only=False)
    user_out_degree = np.zeros(n_users, dtype=np.float32)
    ww_key = ("User", "works_with", "User")
    if ww_key in skeleton.edge_types:
        ei = skeleton[ww_key].edge_index
        src_ids, counts = torch.unique(ei[0], return_counts=True)
        for sid, cnt in zip(src_ids.tolist(), counts.tolist()):
            if sid < n_users:
                user_out_degree[sid] = float(cnt)
    max_deg = float(user_out_degree.max()) if user_out_degree.max() > 0 else 1.0
    user_out_degree = np.log1p(user_out_degree) / np.log1p(max_deg)

    # Extension sensitivity from skeleton FileExtension node features
    n_ext = len(ext_reg)
    ext_sensitivity = np.zeros(n_ext, dtype=np.float32)
    if "FileExtension" in skeleton.node_types:
        fx = skeleton["FileExtension"].x
        if fx is not None and fx.dim() == 2 and fx.shape[1] >= 2:
            scores = fx[:, 1].numpy()
            for i in range(min(n_ext, len(scores))):
                ext_sensitivity[i] = float(scores[i])
    del skeleton

    # Psychology features — normalize user_id to UPPER to match uid_list
    LOGGER.info("Loading psychology_features.csv …")
    psych_df = pd.read_csv(cfg.psychology_csv)
    psych_df["user_id"] = psych_df["user_id"].astype(str).str.strip().str.upper()
    psych_df = psych_df.drop_duplicates("user_id").set_index("user_id")

    user_psychology = _user_arr(uid_list, psych_df,
                                ["PsychologyScore", "psychology_score"])
    user_behav_dev  = _user_arr(uid_list, psych_df,
                                ["BehaviorDeviation", "behavior_deviation"])
    user_aft_hrs    = _user_arr(uid_list, psych_df,
                                ["AfterHoursScore", "after_hours_score"])

    # Fused / behavior features
    LOGGER.info("Loading fused_features.csv …")
    fused_df = pd.read_csv(cfg.fused_csv)
    fused_df["user_id"] = fused_df["user_id"].astype(str).str.strip().str.upper()
    fused_df = fused_df.drop_duplicates("user_id").set_index("user_id")

    total_ev = _user_arr(uid_list, fused_df, ["TotalEvents", "total_events"])
    login_c  = _user_arr(uid_list, fused_df, ["LoginCount",  "login_count"])
    device_c = _user_arr(uid_list, fused_df, ["DeviceCount", "device_count"])
    http_c   = _user_arr(uid_list, fused_df, ["HttpCount",   "http_count"])

    # If BehaviorDeviation missing from psych, try fused
    if user_behav_dev.sum() == 0.0:
        user_behav_dev = _user_arr(uid_list, fused_df,
                                   ["BehaviorDeviation", "behavior_deviation"])

    safe_total  = np.where(total_ev > 0, total_ev, 1.0)
    max_total   = float(total_ev.max()) if total_ev.max() > 0 else 1.0

    LOGGER.info(
        "Lookups ready — %d users | %d ext | %d PCs | %d domains",
        n_users, n_ext, len(pc_reg), len(dom_reg),
    )

    return StaticLookups(
        uid_list=uid_list,
        uid_to_index=uid_to_index,
        uid_to_dept=art["uid_to_dept"],
        uid_to_role=art["uid_to_role"],
        uid_to_team=art["uid_to_team"],
        uid_to_bu=art["uid_to_bu"],
        uid_to_supervisor=art["uid_to_supervisor"],
        dept_registry=dept_reg,
        role_registry=role_reg,
        team_registry=team_reg,
        bu_registry=bu_reg,
        pc_registry=pc_reg,
        domain_registry=dom_reg,
        ext_registry=ext_reg,
        event_type_registry=art["event_type_registry"],
        user_behavior_score=user_behav_dev,
        user_anomaly_score=user_psychology,
        user_psychology=user_psychology,
        user_behav_dev=user_behav_dev,
        user_aft_hrs=user_aft_hrs,
        user_activity_freq=(total_ev / max_total).astype(np.float32),
        user_login_freq=(login_c / safe_total).astype(np.float32),
        user_device_freq=(device_c / safe_total).astype(np.float32),
        user_website_freq=(http_c / safe_total).astype(np.float32),
        user_out_degree=user_out_degree,
        ext_sensitivity=ext_sensitivity,
        max_pc_idx=max(len(pc_reg) - 1, 1),
        max_domain_idx=max(len(dom_reg) - 1, 1),
        max_ext_idx=max(len(ext_reg) - 1, 1),
        max_dept_id=max(len(dept_reg) - 1, 1),
        max_role_id=max(len(role_reg) - 1, 1),
        max_team_id=max(len(team_reg) - 1, 1),
        max_bu_id=max(len(bu_reg) - 1, 1),
    )

# ---------------------------------------------------------------------------
# Pass 1 — per-user temporal state machine
# ---------------------------------------------------------------------------

class UserTemporalState:
    """O(num_users) streaming state for cross-shard temporal features."""

    __slots__ = ("_last_global", "_last_rel", "_sess_start",
                 "_roll_count", "_inter_count")

    def __init__(self) -> None:
        self._last_global: Dict[int, int]             = {}
        self._last_rel:    Dict[Tuple[int, str], int] = {}
        self._sess_start:  Dict[int, int]             = {}
        self._roll_count:  Dict[int, int]             = defaultdict(int)
        self._inter_count: Dict[Tuple[int, int], int] = defaultdict(int)

    def process_shard(self, shard: dict, relation_name: str) -> dict:
        E        = int(shard["edge_index"].shape[1])
        src_arr  = shard["edge_index"][0].numpy()
        dst_arr  = shard["edge_index"][1].numpy()
        time_arr = shard["edge_time"].numpy()

        delta_global   = np.zeros(E, dtype=np.float32)
        delta_relation = np.zeros(E, dtype=np.float32)
        session_dur    = np.zeros(E, dtype=np.float32)
        rolling_count  = np.zeros(E, dtype=np.float32)
        interaction_ct = np.zeros(E, dtype=np.float32)

        for i in range(E):
            uid  = int(src_arr[i])
            did  = int(dst_arr[i])
            t    = int(time_arr[i])
            rkey = (uid, relation_name)

            last_g = self._last_global.get(uid)
            if last_g is not None and t >= last_g:
                delta_global[i] = float(t - last_g)

            last_r = self._last_rel.get(rkey)
            if last_r is not None and t >= last_r:
                delta_relation[i] = float(t - last_r)

            ss = self._sess_start.get(uid)
            if ss is None:
                self._sess_start[uid] = t
            else:
                gap = (t - last_g) if last_g is not None else 0
                if gap > SESSION_GAP_S:
                    self._sess_start[uid] = t
                else:
                    session_dur[i] = float(max(t - self._sess_start[uid], 0))

            self._roll_count[uid] += 1
            rolling_count[i]  = float(self._roll_count[uid])

            ik = (uid, did)
            interaction_ct[i] = float(self._inter_count[ik])
            self._inter_count[ik] += 1

            self._last_global[uid] = t
            self._last_rel[rkey]   = t

        return {
            "delta_global":   delta_global,
            "delta_relation": delta_relation,
            "session_dur":    session_dur,
            "rolling_count":  rolling_count,
            "interaction_ct": interaction_ct,
        }


def run_pass1(cfg: EdgeFeatureConfig, manifest: dict) -> None:
    LOGGER.info("=" * 72)
    LOGGER.info("PASS 1/2  — temporal state streaming (source-chunk order)")
    state   = UserTemporalState()
    shards  = manifest["shards"]
    ordered = sorted(shards, key=lambda m: (m["source_chunk_index"], m["relation"]))
    total   = len(ordered)
    t0      = time.time()

    for idx, meta in enumerate(ordered):
        shard_path = _resolve_shard_path(meta["path"], Path(cfg.graph_output_dir))
        shard      = torch.load(shard_path, weights_only=False)
        rel        = str(shard.get("relation_name", meta["relation"]))

        temporal   = state.process_shard(shard, rel)
        tmp        = _tmp_path(cfg, meta["relation"], meta["shard_index"])
        torch.save(temporal, tmp)

        del shard, temporal
        gc.collect()

        if idx % 10 == 0 or idx == total - 1:
            elapsed = time.time() - t0
            eta     = (elapsed / (idx + 1)) * (total - idx - 1) if idx > 0 else 0.0
            LOGGER.info("  [P1] %d/%d  elapsed=%.1fs  ETA=%.1fs", idx + 1, total, elapsed, eta)

    _log_mem("after pass 1")
    LOGGER.info("PASS 1/2  complete — %d .tmp.pt files written to %s", total, cfg.tmp_dir)


# ---------------------------------------------------------------------------
# Sinusoidal temporal encoding (TGN-style)
# ---------------------------------------------------------------------------

def sinusoidal_encoding(
    timestamps: torch.Tensor, dim: int, max_period: float
) -> torch.Tensor:
    """Fixed sinusoidal time encoding. timestamps: LongTensor[E]."""
    t      = timestamps.float().unsqueeze(-1)               # [E, 1]
    i      = torch.arange(dim, dtype=torch.float32).unsqueeze(0)  # [1, dim]
    freqs  = 1.0 / (max_period ** (i / max(dim, 1)))
    angles = t * freqs
    return torch.where(i % 2 == 0, torch.sin(angles), torch.cos(angles))  # [E, dim]


# ---------------------------------------------------------------------------
# Pass 2 — full feature computation for one shard
# ---------------------------------------------------------------------------

def _lognorm(arr: np.ndarray) -> np.ndarray:
    """log1p-normalize to [0, 1] using 1-year seconds as denominator."""
    return (np.log1p(np.maximum(arr, 0.0)) / _LOG_DENOM).clip(0.0, 1.0).astype(np.float32)


def _safe_idx(idx: int, max_val: int) -> float:
    return float(idx) / float(max(max_val, 1))


def process_shard(
    cfg:     EdgeFeatureConfig,
    lookups: StaticLookups,
    meta:    dict,
) -> dict:
    """
    Load one edge shard + its .tmp.pt, compute the full [E, 40] feature
    tensor and [E, 16] temporal encoding tensor.  Returns a dict ready
    to be saved as the output feature shard.
    """
    shard_path = _resolve_shard_path(meta["path"], Path(cfg.graph_output_dir))
    shard      = torch.load(shard_path, weights_only=False)
    tmp_path   = _tmp_path(cfg, meta["relation"], meta["shard_index"])
    temporal   = torch.load(tmp_path, weights_only=False)

    E           = int(shard["edge_index"].shape[1])
    src_idx_t   = shard["edge_index"][0]           # LongTensor[E]
    dst_idx_t   = shard["edge_index"][1]           # LongTensor[E]
    edge_time_t = shard["edge_time"]               # LongTensor[E]
    event_type_t= shard["event_type"]              # LongTensor[E]
    target_user = shard["target_user"]             # LongTensor[E]  (-1 = none)
    dst_node_type = str(shard["dst_node_type"])
    relation_name = str(shard.get("relation_name", meta["relation"]))

    src_np  = src_idx_t.numpy()
    dst_np  = dst_idx_t.numpy()
    ev_np   = event_type_t.numpy()
    ts_np   = edge_time_t.numpy()

    feat = np.zeros((E, NUM_SCALAR), dtype=np.float32)

    # ------------------------------------------------------------------ #
    # 0-7  Calendar features (vectorized via pandas)
    # ------------------------------------------------------------------ #
    dti = pd.to_datetime(ts_np, unit="s", utc=True)
    feat[:, _FI["hour_of_day"]]     = (dti.hour.to_numpy(dtype=np.float32) / 23.0)
    feat[:, _FI["minute_of_hour"]]  = (dti.minute.to_numpy(dtype=np.float32) / 59.0)
    feat[:, _FI["weekday"]]         = (dti.dayofweek.to_numpy(dtype=np.float32) / 6.0)
    feat[:, _FI["weekend_flag"]]    = (dti.dayofweek.to_numpy() >= 5).astype(np.float32)
    feat[:, _FI["month"]]           = (dti.month.to_numpy(dtype=np.float32) / 12.0)
    feat[:, _FI["quarter"]]         = (dti.quarter.to_numpy(dtype=np.float32) / 4.0)

    hour_np = dti.hour.to_numpy()
    after_hours   = ((hour_np < 7) | (hour_np >= 19)).astype(np.float32)
    working_hours = (~(after_hours.astype(bool)) &
                     (dti.dayofweek.to_numpy() < 5)).astype(np.float32)
    feat[:, _FI["is_after_hours"]]   = after_hours
    feat[:, _FI["is_working_hours"]] = working_hours

    # ------------------------------------------------------------------ #
    # 8-10  Cross-shard temporal deltas (from Pass 1 .tmp.pt)
    # ------------------------------------------------------------------ #
    feat[:, _FI["time_since_last_event"]] = _lognorm(temporal["delta_global"])
    feat[:, _FI["session_duration"]]      = _lognorm(temporal["session_dur"])
    feat[:, _FI["time_gap_prev_action"]]  = _lognorm(temporal["delta_relation"])

    # ------------------------------------------------------------------ #
    # 11-17  Behavior features  (per-user, indexed by src user_idx)
    # ------------------------------------------------------------------ #
    feat[:, _FI["user_behavior_score"]] = lookups.user_behavior_score[src_np].clip(0, 1)
    feat[:, _FI["anomaly_score"]]       = lookups.user_anomaly_score[src_np].clip(0, 1)
    feat[:, _FI["activity_frequency"]]  = lookups.user_activity_freq[src_np].clip(0, 1)

    # rolling_action_count: log1p-normalize against log1p(max observed count)
    rc = temporal["rolling_count"]
    max_rc = float(rc.max()) if rc.max() > 0 else 1.0
    feat[:, _FI["rolling_action_count"]] = (
        np.log1p(rc) / np.log1p(max_rc)
    ).clip(0, 1)

    feat[:, _FI["login_frequency"]]   = lookups.user_login_freq[src_np].clip(0, 1)
    feat[:, _FI["device_frequency"]]  = lookups.user_device_freq[src_np].clip(0, 1)
    feat[:, _FI["website_frequency"]] = lookups.user_website_freq[src_np].clip(0, 1)

    # ------------------------------------------------------------------ #
    # 18-20  Psychological features
    # ------------------------------------------------------------------ #
    feat[:, _FI["psychology_score"]]   = lookups.user_psychology[src_np].clip(0, 1)
    feat[:, _FI["behavior_deviation"]] = lookups.user_behav_dev[src_np].clip(0, 1)
    feat[:, _FI["after_hours_score"]]  = lookups.user_aft_hrs[src_np].clip(0, 1)

    # ------------------------------------------------------------------ #
    # 21-25  Organizational features
    # ------------------------------------------------------------------ #
    def _org_encode(uid_arr: np.ndarray,
                    uid_list: List[str],
                    uid_to_attr: Dict[str, str],
                    registry: Dict[str, int],
                    max_id: int) -> np.ndarray:
        out = np.zeros(len(uid_arr), dtype=np.float32)
        for i, uidx in enumerate(uid_arr):
            uid = uid_list[uidx] if 0 <= uidx < len(uid_list) else ""
            attr = uid_to_attr.get(uid, "unknown")
            enc  = registry.get(attr, 0)
            out[i] = float(enc) / float(max(max_id, 1))
        return out

    feat[:, _FI["dept_encoded"]] = _org_encode(
        src_np, lookups.uid_list, lookups.uid_to_dept,
        lookups.dept_registry, lookups.max_dept_id)
    feat[:, _FI["role_encoded"]] = _org_encode(
        src_np, lookups.uid_list, lookups.uid_to_role,
        lookups.role_registry, lookups.max_role_id)
    feat[:, _FI["team_encoded"]] = _org_encode(
        src_np, lookups.uid_list, lookups.uid_to_team,
        lookups.team_registry, lookups.max_team_id)
    feat[:, _FI["bu_encoded"]] = _org_encode(
        src_np, lookups.uid_list, lookups.uid_to_bu,
        lookups.bu_registry, lookups.max_bu_id)

    has_mgr = np.array(
        [1.0 if (lookups.uid_list[u] in lookups.uid_to_supervisor
                 and lookups.uid_to_supervisor.get(lookups.uid_list[u], "") not in ("", "unknown", "nan"))
         else 0.0
         for u in src_np], dtype=np.float32
    )
    feat[:, _FI["has_manager"]] = has_mgr

    # ------------------------------------------------------------------ #
    # 26-29  Resource features
    # ------------------------------------------------------------------ #
    if dst_node_type == "FileExtension":
        ext_arr = dst_np.clip(0, lookups.max_ext_idx)
        feat[:, _FI["file_sensitivity"]] = lookups.ext_sensitivity[ext_arr].clip(0, 1)
        feat[:, _FI["extension_risk"]]   = lookups.ext_sensitivity[ext_arr].clip(0, 1)
        feat[:, _FI["domain_popularity"]]= 0.0
        feat[:, _FI["pc_popularity"]]    = 0.0
    elif dst_node_type == "WebsiteDomain":
        # Lower registry index = encountered earlier = more common = more popular
        feat[:, _FI["domain_popularity"]] = (
            1.0 - dst_np.clip(0, lookups.max_domain_idx).astype(np.float32)
            / lookups.max_domain_idx
        )
        feat[:, _FI["file_sensitivity"]] = 0.0
        feat[:, _FI["extension_risk"]]   = 0.0
        feat[:, _FI["pc_popularity"]]    = 0.0
    elif dst_node_type == "PC":
        feat[:, _FI["pc_popularity"]] = (
            1.0 - dst_np.clip(0, lookups.max_pc_idx).astype(np.float32)
            / lookups.max_pc_idx
        )
        feat[:, _FI["file_sensitivity"]] = 0.0
        feat[:, _FI["extension_risk"]]   = 0.0
        feat[:, _FI["domain_popularity"]]= 0.0

    # ------------------------------------------------------------------ #
    # 30-32  Graph features
    # ------------------------------------------------------------------ #
    feat[:, _FI["source_degree"]] = lookups.user_out_degree[src_np].clip(0, 1)

    if dst_node_type == "PC":
        max_dst = lookups.max_pc_idx
    elif dst_node_type == "WebsiteDomain":
        max_dst = lookups.max_domain_idx
    else:
        max_dst = lookups.max_ext_idx
    feat[:, _FI["destination_degree"]] = (
        1.0 - dst_np.clip(0, max_dst).astype(np.float32) / max(max_dst, 1)
    )

    ic = temporal["interaction_ct"]
    max_ic = float(ic.max()) if ic.max() > 0 else 1.0
    feat[:, _FI["historical_interaction_count"]] = (
        np.log1p(ic) / np.log1p(max_ic)
    ).clip(0, 1)

    # ------------------------------------------------------------------ #
    # 33-35  Event-type features
    # ------------------------------------------------------------------ #
    et_reg   = lookups.event_type_registry
    logon_ids  = {et_reg.get(k, -9) for k in ("logon", "logoff") if k in et_reg}
    device_ids = {et_reg.get("device", -9)}
    http_ids   = {et_reg.get("http", -9)}

    feat[:, _FI["is_logon_event"]]  = np.isin(ev_np, list(logon_ids)).astype(np.float32)
    feat[:, _FI["is_device_event"]] = np.isin(ev_np, list(device_ids)).astype(np.float32)
    feat[:, _FI["is_http_event"]]   = np.isin(ev_np, list(http_ids)).astype(np.float32)

    # ------------------------------------------------------------------ #
    # 36-38  Relation-type one-hot
    # ------------------------------------------------------------------ #
    feat[:, _FI["rel_accesses_pc"]]   = 1.0 if dst_node_type == "PC"            else 0.0
    feat[:, _FI["rel_visits_web"]]    = 1.0 if dst_node_type == "WebsiteDomain" else 0.0
    feat[:, _FI["rel_touches_file"]]  = 1.0 if dst_node_type == "FileExtension" else 0.0

    # ------------------------------------------------------------------ #
    # 39  Target-user flag
    # ------------------------------------------------------------------ #
    feat[:, _FI["has_target_user"]] = (target_user.numpy() != -1).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Sinusoidal temporal encoding
    # ------------------------------------------------------------------ #
    temporal_enc = sinusoidal_encoding(
        edge_time_t, cfg.temporal_encoding_dim, cfg.temporal_max_period
    )

    out = {
        "edge_index":        shard["edge_index"],
        "edge_time":         edge_time_t,
        "features":          torch.from_numpy(feat),
        "temporal_encoding": temporal_enc,
        "feature_names":     FEATURE_NAMES,
        "feature_version":   FEATURE_VERSION,
        "relation":          meta["relation"],
        "src_node_type":     meta["src_type"],
        "dst_node_type":     meta["dst_type"],
        "shard_index":       meta["shard_index"],
        "num_edges":         E,
    }

    del shard, temporal
    gc.collect()
    return out


# ---------------------------------------------------------------------------
# Per-shard validation
# ---------------------------------------------------------------------------

def validate_feature_shard(out: dict, meta: dict) -> List[str]:
    """Return list of error strings (empty = passed)."""
    errors: List[str] = []
    rel = meta["relation"]
    si  = meta["shard_index"]
    E   = meta["num_edges"]

    feat = out["features"]
    tenc = out["temporal_encoding"]

    if feat.shape != (E, NUM_SCALAR):
        errors.append(f"{rel}[{si}]: feature shape {tuple(feat.shape)} != ({E}, {NUM_SCALAR})")
    if tenc.shape != (E, TEMPORAL_DIM):
        errors.append(f"{rel}[{si}]: temporal_encoding shape {tuple(tenc.shape)} != ({E}, {TEMPORAL_DIM})")
    if torch.isnan(feat).any():
        bad = [FEATURE_NAMES[i] for i in torch.isnan(feat).any(dim=0).nonzero().flatten().tolist()]
        errors.append(f"{rel}[{si}]: NaN in columns {bad}")
    if torch.isinf(feat).any():
        bad = [FEATURE_NAMES[i] for i in torch.isinf(feat).any(dim=0).nonzero().flatten().tolist()]
        errors.append(f"{rel}[{si}]: Inf in columns {bad}")
    if torch.isnan(tenc).any() or torch.isinf(tenc).any():
        errors.append(f"{rel}[{si}]: NaN/Inf in temporal_encoding")
    if out["num_edges"] != E:
        errors.append(f"{rel}[{si}]: num_edges mismatch")
    if len(out["feature_names"]) != NUM_SCALAR:
        errors.append(f"{rel}[{si}]: feature_names length {len(out['feature_names'])} != {NUM_SCALAR}")
    return errors


# ---------------------------------------------------------------------------
# Pass 2 — orchestrate full feature computation
# ---------------------------------------------------------------------------

def run_pass2(
    cfg:      EdgeFeatureConfig,
    manifest: dict,
    lookups:  StaticLookups,
) -> List[dict]:
    LOGGER.info("=" * 72)
    LOGGER.info("PASS 2/2  — full feature computation, one shard at a time")

    shards      = manifest["shards"]
    total       = len(shards)
    shard_metas = []
    total_edges = 0
    all_errors: List[str] = []
    t0 = time.time()

    for idx, meta in enumerate(shards):
        t_shard = time.time()

        out = process_shard(cfg, lookups, meta)

        errors = validate_feature_shard(out, meta)
        if errors:
            for e in errors:
                LOGGER.error("[VALIDATE] %s", e)
            all_errors.extend(errors)
        else:
            LOGGER.debug("[VALIDATE] %s[%d] passed", meta["relation"], meta["shard_index"])

        out_path = _feature_shard_path(cfg, meta["relation"], meta["shard_index"])
        torch.save(out, out_path)

        # Delete .tmp.pt
        tmp = _tmp_path(cfg, meta["relation"], meta["shard_index"])
        if tmp.exists():
            tmp.unlink()

        E        = out["num_edges"]
        elapsed  = time.time() - t_shard
        rate     = E / elapsed if elapsed > 0 else 0.0
        total_elapsed = time.time() - t0
        eta      = (total_elapsed / (idx + 1)) * (total - idx - 1) if idx > 0 else 0.0

        # Feature stats for logging (5 key features)
        f = out["features"]
        key_cols = ["time_since_last_event", "psychology_score",
                    "file_sensitivity", "rolling_action_count", "source_degree"]
        stats_str = "  ".join(
            f"{c}=[{f[:, _FI[c]].min():.3f},{f[:, _FI[c]].max():.3f}]"
            for c in key_cols if c in _FI
        )

        LOGGER.info(
            "[P2] %d/%d  %s[%d]  E=%d  %.0f rows/s  ETA=%.0fs | %s",
            idx + 1, total, meta["relation"], meta["shard_index"],
            E, rate, eta, stats_str,
        )
        _log_mem(f"shard {idx + 1}/{total}")

        shard_metas.append({
            "relation":      meta["relation"],
            "src_type":      meta["src_type"],
            "dst_type":      meta["dst_type"],
            "shard_index":   meta["shard_index"],
            "path":          str(out_path),
            "num_edges":     E,
            "feature_shape": [E, NUM_SCALAR],
            "source_chunk_index": meta["source_chunk_index"],
        })
        total_edges += E

        del out
        gc.collect()

    # Clean up tmp dir if empty
    try:
        cfg.tmp_dir.rmdir()
    except OSError:
        pass

    if all_errors:
        LOGGER.warning("PASS 2/2 finished with %d validation error(s)", len(all_errors))
    else:
        LOGGER.info("PASS 2/2 complete — all %d shards validated OK", total)

    LOGGER.info("Total edges processed: %d", total_edges)
    _log_mem("after pass 2")
    return shard_metas


# ---------------------------------------------------------------------------
# Feature manifest
# ---------------------------------------------------------------------------

def write_feature_manifest(
    cfg:         EdgeFeatureConfig,
    shard_metas: List[dict],
) -> None:
    payload = {
        "feature_version":       FEATURE_VERSION,
        "num_scalar_features":   NUM_SCALAR,
        "temporal_encoding_dim": TEMPORAL_DIM,
        "feature_names":         FEATURE_NAMES,
        "feature_dtype":         "float32",
        "session_gap_seconds":   SESSION_GAP_S,
        "total_edges":           sum(m["num_edges"] for m in shard_metas),
        "total_shards":          len(shard_metas),
        "shards":                shard_metas,
    }
    with open(cfg.feature_manifest, "w") as f:
        json.dump(payload, f, indent=2)
    LOGGER.info("Feature manifest saved → %s", cfg.feature_manifest)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run(cfg: Optional[EdgeFeatureConfig] = None) -> None:
    cfg = cfg or EdgeFeatureConfig()
    global LOGGER
    LOGGER = _setup_logging(cfg.log_path)

    LOGGER.info("=" * 72)
    LOGGER.info("Edge Feature Engine — start")
    LOGGER.info("  graph_output_dir : %s", cfg.graph_output_dir)
    LOGGER.info("  psychology_csv   : %s", cfg.psychology_csv)
    LOGGER.info("  fused_csv        : %s", cfg.fused_csv)
    LOGGER.info("  output_shard_dir : %s", cfg.output_shard_dir)
    LOGGER.info("=" * 72)
    t0 = time.time()

    with open(cfg.manifest_path, "r") as f:
        manifest = json.load(f)

    LOGGER.info("Manifest loaded — %d shards", len(manifest["shards"]))

    lookups = load_static_lookups(cfg)
    _log_mem("after loading static lookups")

    run_pass1(cfg, manifest)
    shard_metas = run_pass2(cfg, manifest, lookups)
    write_feature_manifest(cfg, shard_metas)

    elapsed = time.time() - t0
    LOGGER.info("=" * 72)
    LOGGER.info("Edge Feature Engine — DONE in %.1f s", elapsed)
    LOGGER.info("  Feature shards : %s", cfg.output_shard_dir)
    LOGGER.info("  Manifest       : %s", cfg.feature_manifest)
    LOGGER.info("  Log            : %s", cfg.log_path)
    LOGGER.info("  Scalar dims    : %d", NUM_SCALAR)
    LOGGER.info("  Temporal dims  : %d", TEMPORAL_DIM)
    LOGGER.info("=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Streaming edge feature engine for CERT insider-threat graph."
    )
    p.add_argument("--graph-output-dir", type=Path,
                   default=Path("graph/output"))
    p.add_argument("--psychology-csv",   type=Path,
                   default=Path("data/processed/psychology_features.csv"))
    p.add_argument("--fused-csv",        type=Path,
                   default=Path("data/processed/fused_features.csv"))
    p.add_argument("--behavior-csv",     type=Path,
                   default=Path("data/processed/behavior_features.csv"))
    p.add_argument("--temporal-encoding-dim", type=int, default=TEMPORAL_DIM)
    p.add_argument("--temporal-max-period",   type=float, default=1.0e6)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg  = EdgeFeatureConfig(
        graph_output_dir      = args.graph_output_dir,
        psychology_csv        = args.psychology_csv,
        fused_csv             = args.fused_csv,
        behavior_csv          = args.behavior_csv,
        temporal_encoding_dim = args.temporal_encoding_dim,
        temporal_max_period   = args.temporal_max_period,
    )
    run(cfg)


if __name__ == "__main__":
    main()