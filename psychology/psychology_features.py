"""
psychology_features.py  –  Phase 3
====================================
Advanced Behavioural Psychology Feature Engineering
Publication-quality insider-threat research pipeline.
Supports 30M+ events via chunk streaming. Peak RAM target < 2 GB.
Output: data/processed/psychology_features.csv
        data/processed/psychology_metadata.pkl
"""

import gc
import logging
import os
import pickle
import time
import traceback
import tracemalloc
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from scipy.stats import iqr
from sklearn.preprocessing import MinMaxScaler, RobustScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Test Mode Configuration
# ---------------------------------------------------------------------------
TEST_MODE = False
MAX_TEST_CHUNKS = 5

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("phase3.psychology")

# ---------------------------------------------------------------------------
# Paths (Project-Root Relative Auto-Detection)
# ---------------------------------------------------------------------------

# Automatically detects the root directory containing the data folders
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROC = PROJECT_ROOT / "data" / "processed"
DATA_RAW = PROJECT_ROOT / "data" / "raw"

UNUnified_PATH = DATA_PROC / "unified_events.csv"
LDAP_PATH = DATA_RAW / "LDAP.csv"
PSYCH_PATH = DATA_RAW / "psychometric.csv"
OUTPUT_CSV = DATA_PROC / "psychology_features.csv" if TEST_MODE else DATA_PROC / "psychology_features.csv"
OUTPUT_META = DATA_PROC / "psychology_metadata.pkl"

CHUNK_SIZE = 10000 if TEST_MODE else 500_000

# ---------------------------------------------------------------------------
# Configurable weights
# ---------------------------------------------------------------------------

# BehaviorDeviation sub-component weights
BD_WEIGHTS: Dict[str, float] = {
    "login_time_dev":        1.8,
    "daily_activity_dev":    1.6,
    "comm_dev":              1.5,
    "resource_diversity_dev":1.3,
    "website_dev":           1.2,
    "file_access_dev":       1.4,
    "usb_dev":               1.3,
    "login_freq_dev":        1.2,
    "session_duration_dev":  1.1,
    "iat_dev":               1.0,
}

# PsychologyScore fusion weights
PS_WEIGHTS: Dict[str, float] = {
    "BehaviorDeviation":           2.5,
    "AfterHoursScore":             1.6,
    "AccessFrequencyShift":        1.5,
    "PrivilegeEscalationIndicator":1.8,
    "FailedLoginRate":             1.7,
    "ResourceExploration":         1.4,
    "CommunicationReduction":      1.3,
    "USBUsageFrequency":           1.4,
    "RiskAcceleration":            1.6,
    "BehaviorTrendScore":          1.4,
    "ExternalCommunicationScore":  1.3,
    "NightActivityScore":          1.5,
    "BurstActivityScore":          1.4,
    "OffHoursResourceAccess":      1.3,
    "WeekendActivityScore":        1.0,
    "UniquePCUsage":               1.1,
    "UniqueWebsiteUsage":          1.0,
    "UniqueFileTypeUsage":         1.0,
    "EmailAttachmentFrequency":    1.1,
    "SessionDurationMean":         0.8,
    "SessionDurationStd":          0.9,
    "ResourceEntropy":             1.1,
    "BehaviorConsistency":         1.2,
    "CommunicationEntropy":        1.0,
    "ActivityEntropy":             1.0,
    "DeviceSwitchRate":            1.1,
    "ResourceNoveltyScore":         1.2,
    "SessionIrregularity":         1.1,
    "CommunicationVolatility":     1.1,
    "WebsiteNoveltyScore":         1.0,
    "FileNoveltyScore":            1.1,
    "USBNoveltyScore":             1.2,
    "DepartmentDeviation":         1.0,
    "RoleDeviation":               1.1,
    "TeamDeviation":               1.0,
    "TemporalConsistency":         1.0,
}

# Rolling window sizes (days)
SHORT_WIN = 7
LONG_WIN = 30
RECENT_WIN = 14

# ---------------------------------------------------------------------------
# Required output columns (enforced ordering)
# ---------------------------------------------------------------------------

REQUIRED_COLS = [
    "user_id",
    "AfterHoursScore",
    "CommunicationReduction",
    "ResourceExploration",
    "BehaviorDeviation",
    "AccessFrequencyShift",
    "WeekendActivityScore",
    "FailedLoginRate",
    "UniquePCUsage",
    "UniqueWebsiteUsage",
    "UniqueFileTypeUsage",
    "USBUsageFrequency",
    "EmailAttachmentFrequency",
    "ExternalCommunicationScore",
    "PrivilegeEscalationIndicator",
    "SessionDurationMean",
    "SessionDurationStd",
    "BurstActivityScore",
    "NightActivityScore",
    "ResourceEntropy",
    "BehaviorConsistency",
    "CommunicationEntropy",
    "ActivityEntropy",
    "DeviceSwitchRate",
    "ResourceNoveltyScore",
    "OffHoursResourceAccess",
    "SessionIrregularity",
    "CommunicationVolatility",
    "WebsiteNoveltyScore",
    "FileNoveltyScore",
    "USBNoveltyScore",
    "DepartmentDeviation",
    "RoleDeviation",
    "TeamDeviation",
    "BehaviorTrendScore",
    "TemporalConsistency",
    "RiskAcceleration",
    "PsychologyScore",
]

# Features using RobustScaler; rest use MinMaxScaler
ROBUST_FEATURES = {
    "SessionDurationMean", "SessionDurationStd", "BurstActivityScore",
    "NightActivityScore", "ResourceEntropy", "CommunicationEntropy",
    "ActivityEntropy", "DeviceSwitchRate", "SessionIrregularity",
    "CommunicationVolatility", "TemporalConsistency",
}

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _clip01(v: float) -> float:
    return float(np.clip(v, 0.0, 1.0))


def _safe_mean(arr: np.ndarray, default: float = 0.0) -> float:
    return float(np.mean(arr)) if len(arr) > 0 else default


def _safe_std(arr: np.ndarray, default: float = 1e-6) -> float:
    s = float(np.std(arr, ddof=0)) if len(arr) > 1 else default
    return max(s, 1e-9)


def _entropy_of(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    counts = series.value_counts(normalize=True).values
    return float(scipy_entropy(counts, base=2)) if len(counts) > 1 else 0.0


def _robust_zscore_norm(observed: float, median: float, iqr_val: float) -> float:
    """Robust z-score: |x - median| / (IQR/1.349), clipped to [0,3]/3."""
    scale = max(iqr_val / 1.349, 1e-9)
    return float(np.clip(abs(observed - median) / scale, 0.0, 3.0) / 3.0)


def _zscore_norm(observed: float, mean: float, std: float) -> float:
    return float(np.clip(abs(observed - mean) / max(std, 1e-9), 0.0, 3.0) / 3.0)


# ---------------------------------------------------------------------------
# Step 1 & 2 – Timestamp parsing, temporal fields, chronological sort
# ---------------------------------------------------------------------------


def parse_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")
    bad = df["timestamp"].isna().sum()
    if bad:
        log.warning("Dropping %d rows with unparseable timestamps.", bad)
        df = df.dropna(subset=["timestamp"])
    df["hour"]           = df["timestamp"].dt.hour.astype(np.int8)
    df["day"]            = df["timestamp"].dt.day.astype(np.int8)
    df["weekday"]        = df["timestamp"].dt.weekday.astype(np.int8)
    df["weekend"]        = (df["weekday"] >= 5).astype(np.int8)
    df["month"]          = df["timestamp"].dt.month.astype(np.int8)
    df["business_hours"] = (((df["hour"] >= 8) & (df["hour"] < 18)) &
                             (df["weekday"] < 5)).astype(np.int8)
    df["after_hours"]    = (1 - df["business_hours"]).astype(np.int8)
    df["night_hours"]    = ((df["hour"] >= 22) | (df["hour"] < 6)).astype(np.int8)
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Step 3 – Per-user historical baseline (computed on full dataset)
# ---------------------------------------------------------------------------

def _event_flag(df: pd.DataFrame, pattern: str, regex: bool = False) -> pd.Series:
    return df["event_type"].str.lower().str.contains(pattern, na=False, regex=regex)


def build_baselines(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    log.info("Building per-user historical baselines …")
    baselines: Dict[str, Dict[str, Any]] = {}

    for uid, udf in df.groupby("user_id"):
        udf = udf.sort_values("timestamp")
        n = len(udf)

        # Masks
        m_logon   = _event_flag(udf, "logon")
        m_logoff  = _event_flag(udf, "logoff")
        m_email   = _event_flag(udf, "email")
        m_file    = _event_flag(udf, "file")
        m_http    = _event_flag(udf, "http")
        m_device  = _event_flag(udf, r"device|removable", regex=True)
        m_failed  = _event_flag(udf, r"fail|failed", regex=True)

        # Login hours
        login_hrs  = udf.loc[m_logon,  "hour"].values.astype(float)
        logout_hrs = udf.loc[m_logoff, "hour"].values.astype(float)

        # Inter-arrival times (seconds)
        ts_sec = (udf["timestamp"].astype(np.int64) / 1e9).values
        iat    = np.diff(ts_sec)

        # Daily event counts
        udf2 = udf.copy()
        udf2["date"] = udf["timestamp"].dt.date
        daily = udf2.groupby("date").size().values.astype(float)

        # Session durations
        sessions = _session_durations(udf, m_logon, m_logoff)

        # Daily comm
        daily_comm_vals = (udf2.loc[m_email].groupby("date").size().values.astype(float)
                           if m_email.sum() > 0 else np.array([0.0]))

        # Resource set
        res_set = set(udf["resource"].dropna().astype(str).str.lower().unique())
        web_set = set(udf.loc[m_http, "resource"].dropna().astype(str).str.lower().unique())
        file_set= set(udf.loc[m_file, "resource"].dropna().astype(str).str.lower().unique())
        usb_set = set(udf.loc[m_device, "resource"].dropna().astype(str).str.lower().unique())

        # Weekly profile (weekday → mean count)
        weekly_counts: Dict[int, float] = {}
        for wd in range(7):
            wdf = udf2.loc[udf["weekday"] == wd]
            weekly_counts[wd] = float(len(wdf) / max(1, n)) if len(wdf) > 0 else 0.0

        # Hour-of-day profile
        hour_profile = np.zeros(24, dtype=float)
        for h, cnt in udf2.groupby("hour").size().items():
            hour_profile[int(h)] = float(cnt)
        if hour_profile.sum() > 0:
            hour_profile /= hour_profile.sum()

        baselines[uid] = {
            "n":                   n,
            "login_hr_mean":       _safe_mean(login_hrs, 9.0),
            "login_hr_std":        _safe_std(login_hrs),
            "login_hr_median":     float(np.median(login_hrs)) if len(login_hrs) else 9.0,
            "login_hr_iqr":        float(iqr(login_hrs)) if len(login_hrs) > 1 else 1.0,
            "logout_hr_mean":      _safe_mean(logout_hrs, 18.0),
            "logout_hr_std":       _safe_std(logout_hrs),
            "logout_hr_median":    float(np.median(logout_hrs)) if len(logout_hrs) else 18.0,
            "logout_hr_iqr":       float(iqr(logout_hrs)) if len(logout_hrs) > 1 else 1.0,
            "daily_mean":          _safe_mean(daily),
            "daily_std":           _safe_std(daily),
            "daily_median":        float(np.median(daily)) if len(daily) else 0.0,
            "daily_iqr":           float(iqr(daily)) if len(daily) > 1 else 1.0,
            "comm_daily_mean":     _safe_mean(daily_comm_vals),
            "comm_daily_std":      _safe_std(daily_comm_vals),
            "comm_daily_median":   float(np.median(daily_comm_vals)),
            "comm_daily_iqr":      float(iqr(daily_comm_vals)) if len(daily_comm_vals) > 1 else 1.0,
            "file_count":          float(m_file.sum()),
            "file_std":            max(float(m_file.sum()) * 0.3, 1.0),
            "web_count":           float(m_http.sum()),
            "web_std":             max(float(m_http.sum()) * 0.3, 1.0),
            "device_count":        float(m_device.sum()),
            "device_std":          max(float(m_device.sum()) * 0.3, 1.0),
            "login_freq":          float(m_logon.sum()) / max(n, 1),
            "login_freq_std":      max(float(m_logon.sum()) * 0.2 / max(n, 1), 1e-6),
            "session_dur_mean":    _safe_mean(np.array(sessions)) if sessions else 0.0,
            "session_dur_std":     _safe_std(np.array(sessions)) if len(sessions) > 1 else 1.0,
            "session_dur_median":  float(np.median(sessions)) if sessions else 0.0,
            "session_dur_iqr":     float(iqr(sessions)) if len(sessions) > 1 else 1.0,
            "iat_mean":            _safe_mean(iat),
            "iat_median":          float(np.median(iat)) if len(iat) else 0.0,
            "iat_iqr":             float(iqr(iat)) if len(iat) > 1 else 1.0,
            "iat_std":             _safe_std(iat),
            "after_hours_rate":    float(udf["after_hours"].mean()),
            "resource_set":        res_set,
            "web_set":             web_set,
            "file_set":            file_set,
            "usb_set":             usb_set,
            "weekly_counts":       weekly_counts,
            "hour_profile":        hour_profile,
        }

    log.info("Baselines built for %d users.", len(baselines))
    return baselines


def _session_durations(udf: pd.DataFrame, m_logon: pd.Series,
                       m_logoff: pd.Series) -> List[float]:
    logons  = udf.loc[m_logon,  "timestamp"].reset_index(drop=True)
    logoffs = udf.loc[m_logoff, "timestamp"].reset_index(drop=True)
    n = min(len(logons), len(logoffs))
    durations = []
    for i in range(n):
        diff = (logoffs.iloc[i] - logons.iloc[i]).total_seconds() / 3600.0
        if 0.0 < diff < 24.0:
            durations.append(diff)
    return durations


# ---------------------------------------------------------------------------
# Step 4 – Primary research-grade features
# ---------------------------------------------------------------------------


def feat_after_hours_score(udf: pd.DataFrame, bl: Dict) -> float:
    expected = max(bl["after_hours_rate"], 0.05)
    observed = float(udf["after_hours"].mean())
    return _clip01(observed / expected / 3.0)


def feat_communication_reduction(udf: pd.DataFrame, bl: Dict,
                                 short: int = SHORT_WIN) -> float:
    m_email = _event_flag(udf, "email")
    cutoff  = udf["timestamp"].max() - pd.Timedelta(days=short)
    recent  = udf.loc[m_email & (udf["timestamp"] >= cutoff)]
    udf2    = udf.copy(); udf2["date"] = udf["timestamp"].dt.date
    r_daily = recent.copy(); r_daily["date"] = recent["timestamp"].dt.date
    daily_r = float(r_daily.groupby("date").size().mean()) if len(recent) > 0 else 0.0
    hist    = max(bl["comm_daily_mean"], 1e-9)
    return _clip01((hist - daily_r) / hist)


def feat_resource_exploration(udf: pd.DataFrame, bl: Dict) -> float:
    cur = set(udf["resource"].dropna().astype(str).str.lower().unique())
    if not cur:
        return 0.0
    new_r = cur - bl["resource_set"]
    return _clip01(len(new_r) / len(cur))


def feat_behavior_deviation(udf: pd.DataFrame, bl: Dict) -> float:
    """
    Composite behavioral anomaly score.
    Each component: robust z-score → [0,1], fused via BD_WEIGHTS.
    """
    m_logon  = _event_flag(udf, "logon")
    m_logoff = _event_flag(udf, "logoff")
    m_email  = _event_flag(udf, "email")
    m_file   = _event_flag(udf, "file")
    m_http   = _event_flag(udf, "http")
    m_device = _event_flag(udf, r"device|removable", regex=True)

    login_hrs  = udf.loc[m_logon,  "hour"].values.astype(float)
    obs_login  = _safe_mean(login_hrs, bl["login_hr_mean"])

    udf2 = udf.copy(); udf2["date"] = udf["timestamp"].dt.date
    daily = udf2.groupby("date").size().values.astype(float)
    obs_daily = _safe_mean(daily)

    daily_comm_df = udf2.loc[m_email].groupby("date").size()
    obs_comm = float(daily_comm_df.mean()) if len(daily_comm_df) > 0 else 0.0

    obs_res_div = float(udf["resource"].nunique())

    obs_web = float(m_http.sum())
    obs_file = float(m_file.sum())
    obs_usb  = float(m_device.sum())

    n = max(len(udf), 1)
    obs_login_freq = float(m_logon.sum()) / n

    sessions = _session_durations(udf, m_logon, m_logoff)
    obs_sess = _safe_mean(np.array(sessions)) if sessions else bl["session_dur_mean"]

    ts_sec = (udf["timestamp"].astype(np.int64) / 1e9).values
    iat = np.diff(ts_sec)
    obs_iat = _safe_mean(iat) if len(iat) > 0 else bl["iat_mean"]

    components = {
        "login_time_dev":         _robust_zscore_norm(obs_login,
                                     bl["login_hr_median"], bl["login_hr_iqr"]),
        "daily_activity_dev":     _robust_zscore_norm(obs_daily,
                                     bl["daily_median"], bl["daily_iqr"]),
        "comm_dev":               _robust_zscore_norm(obs_comm,
                                     bl["comm_daily_median"], bl["comm_daily_iqr"]),
        "resource_diversity_dev": _zscore_norm(obs_res_div,
                                     bl["daily_mean"], bl["daily_std"]),
        "website_dev":            _zscore_norm(obs_web,
                                     bl["web_count"], bl["web_std"]),
        "file_access_dev":        _zscore_norm(obs_file,
                                     bl["file_count"], bl["file_std"]),
        "usb_dev":                _zscore_norm(obs_usb,
                                     bl["device_count"], bl["device_std"]),
        "login_freq_dev":         _zscore_norm(obs_login_freq,
                                     bl["login_freq"], bl["login_freq_std"]),
        "session_duration_dev":   _robust_zscore_norm(obs_sess,
                                     bl["session_dur_median"], bl["session_dur_iqr"]),
        "iat_dev":                _robust_zscore_norm(obs_iat,
                                     bl["iat_median"], bl["iat_iqr"]),
    }

    total_w = sum(BD_WEIGHTS[k] for k in components)
    score   = sum(BD_WEIGHTS[k] * v for k, v in components.items()) / total_w
    return _clip01(score)


def feat_access_frequency_shift(udf: pd.DataFrame, bl: Dict,
                                 recent: int = RECENT_WIN) -> float:
    udf2 = udf.copy(); udf2["date"] = udf["timestamp"].dt.date
    cutoff = udf["timestamp"].max() - pd.Timedelta(days=recent)
    rc = udf2.loc[udf["timestamp"] >= cutoff].groupby("date").size().values.astype(float)
    obs_r = _safe_mean(rc) if len(rc) > 0 else 0.0
    shift = (obs_r - bl["daily_median"]) / max(bl["daily_iqr"] / 1.349, 1e-9)
    return _clip01(shift / 3.0)


# ---------------------------------------------------------------------------
# Step 5 – Additional behavioural indicators
# ---------------------------------------------------------------------------


def feat_weekend_activity(udf: pd.DataFrame) -> float:
    r = float(udf["weekend"].mean())
    return _clip01(r / (2 / 7) / 2.0)


def feat_failed_login_rate(udf: pd.DataFrame) -> float:
    m_logon  = _event_flag(udf, "logon")
    m_failed = _event_flag(udf, r"fail|failed", regex=True)
    n_logon  = m_logon.sum()
    if n_logon == 0:
        return 0.0
    return _clip01(float((m_logon & m_failed).sum()) / n_logon)


def feat_unique_pc_usage(udf: pd.DataFrame) -> float:
    m = _event_flag(udf, r"logon|logoff|device", regex=True)
    return _clip01(float(udf.loc[m, "resource"].dropna().nunique()) / 20.0)


def feat_unique_website_usage(udf: pd.DataFrame) -> float:
    m = _event_flag(udf, "http")
    return _clip01(float(udf.loc[m, "resource"].dropna().nunique()) / 200.0)


def feat_unique_filetype_usage(udf: pd.DataFrame) -> float:
    m = _event_flag(udf, "file")
    files = udf.loc[m, "resource"].dropna().astype(str).str.lower()
    exts  = files.str.extract(r"\.(\w{1,6})$", expand=False).dropna()
    return _clip01(float(exts.nunique()) / 30.0)


def feat_usb_usage_frequency(udf: pd.DataFrame) -> float:
    m = _event_flag(udf, r"device|removable", regex=True)
    return _clip01(float(m.sum()) / max(len(udf), 1) / 0.1)


def feat_email_attachment_frequency(udf: pd.DataFrame) -> float:
    m_email  = _event_flag(udf, "email")
    n_email  = m_email.sum()
    if n_email == 0:
        return 0.0
    res      = udf.loc[m_email, "resource"].dropna().astype(str).str.lower()
    n_attach = int(res.str.contains(r"\.\w{2,6}$", regex=True, na=False).sum())
    return _clip01(n_attach / n_email)


def feat_external_communication(udf: pd.DataFrame) -> float:
    m_email = _event_flag(udf, "email")
    n_email = m_email.sum()
    if n_email == 0:
        return 0.0
    targets = udf.loc[m_email, "target_user"].dropna().astype(str).str.lower()
    ext     = targets.str.contains(r"@(?!.*internal|.*corp|.*company)", regex=True, na=False)
    return _clip01(float(ext.sum()) / n_email)


def feat_privilege_escalation(udf: pd.DataFrame) -> float:
    priv = udf["resource"].dropna().astype(str).str.lower().str.contains(
        r"admin|root|sudo|privilege|escalat|system32|passwd|shadow|secur",
        regex=True, na=False
    )
    return _clip01(float(priv.sum()) / max(len(udf), 1) / 0.05)


def feat_session_duration_stats(udf: pd.DataFrame) -> Tuple[float, float]:
    m_logon  = _event_flag(udf, "logon")
    m_logoff = _event_flag(udf, "logoff")
    sessions = _session_durations(udf, m_logon, m_logoff)
    if not sessions:
        return 0.0, 0.0
    arr = np.array(sessions)
    return _clip01(float(np.mean(arr)) / 12.0), _clip01(float(np.std(arr, ddof=0)) / 6.0)


def feat_burst_activity(udf: pd.DataFrame, window_min: int = 60) -> float:
    idx  = udf.set_index("timestamp").sort_index()
    rc   = idx.resample(f"{window_min}min").size()
    if rc.empty or rc.mean() < 1e-9:
        return 0.0
    ratio = float(rc.max()) / float(rc.mean())
    return _clip01((ratio - 1.0) / 9.0)


def feat_night_activity(udf: pd.DataFrame) -> float:
    return _clip01(float(udf["night_hours"].mean()) / 0.3)


def feat_resource_entropy(udf: pd.DataFrame) -> float:
    s   = udf["resource"].dropna().astype(str).str.lower()
    ent = _entropy_of(s)
    mx  = np.log2(max(s.nunique(), 2))
    return _clip01(ent / mx)


def feat_behavior_consistency(udf: pd.DataFrame, bl: Dict) -> float:
    udf2 = udf.copy(); udf2["date"] = udf["timestamp"].dt.date
    dc   = udf2.groupby("date").size()
    if len(dc) < 2:
        return 0.0
    cv = float(dc.std(ddof=0)) / max(float(dc.mean()), 1e-9)
    return _clip01(cv / 2.0)


def feat_communication_entropy(udf: pd.DataFrame) -> float:
    m = _event_flag(udf, "email")
    t = udf.loc[m, "target_user"].dropna().astype(str).str.lower()
    if t.empty:
        return 0.0
    return _clip01(_entropy_of(t) / np.log2(max(t.nunique(), 2)))


def feat_activity_entropy(udf: pd.DataFrame) -> float:
    et  = udf["event_type"].dropna().str.lower()
    ent = _entropy_of(et)
    mx  = np.log2(max(et.nunique(), 2))
    return _clip01(ent / mx)


def feat_device_switch_rate(udf: pd.DataFrame) -> float:
    m    = _event_flag(udf, r"logon|logoff", regex=True)
    devs = udf.loc[m, "resource"].dropna().astype(str).str.lower()
    if len(devs) < 2:
        return 0.0
    switches = int((devs != devs.shift()).sum()) - 1
    return _clip01(switches / max(len(devs), 1) / 0.5)


def feat_resource_novelty(udf: pd.DataFrame, bl: Dict) -> float:
    return feat_resource_exploration(udf, bl)


def feat_off_hours_resource_access(udf: pd.DataFrame) -> float:
    ah  = udf["after_hours"] == 1
    res = udf.loc[ah, "resource"].dropna().astype(str).str.lower()
    all_res = udf["resource"].dropna().astype(str).str.lower()
    if all_res.nunique() == 0:
        return 0.0
    return _clip01(res.nunique() / all_res.nunique())


def feat_session_irregularity(udf: pd.DataFrame, bl: Dict) -> float:
    m_logon  = _event_flag(udf, "logon")
    m_logoff = _event_flag(udf, "logoff")
    sessions = _session_durations(udf, m_logon, m_logoff)
    if not sessions:
        return 0.0
    arr = np.array(sessions)
    obs_std = float(np.std(arr, ddof=0))
    return _clip01(obs_std / max(bl["session_dur_std"], 1e-9) / 3.0)


def feat_communication_volatility(udf: pd.DataFrame, bl: Dict) -> float:
    m     = _event_flag(udf, "email")
    udf2  = udf.copy(); udf2["date"] = udf["timestamp"].dt.date
    dc    = udf2.loc[m].groupby("date").size().values.astype(float)
    if len(dc) < 2:
        return 0.0
    cv = float(np.std(dc, ddof=0)) / max(float(np.mean(dc)), 1e-9)
    return _clip01(cv / 2.0)


def feat_website_novelty(udf: pd.DataFrame, bl: Dict) -> float:
    m   = _event_flag(udf, "http")
    cur = set(udf.loc[m, "resource"].dropna().astype(str).str.lower().unique())
    if not cur:
        return 0.0
    new_w = cur - bl["web_set"]
    return _clip01(len(new_w) / len(cur))


def feat_file_novelty(udf: pd.DataFrame, bl: Dict) -> float:
    m   = _event_flag(udf, "file")
    cur = set(udf.loc[m, "resource"].dropna().astype(str).str.lower().unique())
    if not cur:
        return 0.0
    new_f = cur - bl["file_set"]
    return _clip01(len(new_f) / len(cur))


def feat_usb_novelty(udf: pd.DataFrame, bl: Dict) -> float:
    m   = _event_flag(udf, r"device|removable", regex=True)
    cur = set(udf.loc[m, "resource"].dropna().astype(str).str.lower().unique())
    if not cur:
        return 0.0
    new_u = cur - bl["usb_set"]
    return _clip01(len(new_u) / len(cur))


def feat_department_deviation(udf: pd.DataFrame,
                               dept_profiles: Dict[str, np.ndarray]) -> float:
    """Hour-profile deviation vs department mean profile."""
    dept = udf["department"].iloc[0] if "department" in udf.columns else None
    if dept is None or dept not in dept_profiles:
        return 0.0
    hour_counts = np.zeros(24, dtype=float)
    for h, cnt in udf.groupby("hour").size().items():
        hour_counts[int(h)] = float(cnt)
    if hour_counts.sum() > 0:
        hour_counts /= hour_counts.sum()
    dept_mean = dept_profiles[dept]
    diff = np.abs(hour_counts - dept_mean).mean()
    return _clip01(diff * 10.0)


def feat_role_deviation(udf: pd.DataFrame,
                         role_profiles: Dict[str, np.ndarray]) -> float:
    role = udf["role"].iloc[0] if "role" in udf.columns else None
    if role is None or role not in role_profiles:
        return 0.0
    hour_counts = np.zeros(24, dtype=float)
    for h, cnt in udf.groupby("hour").size().items():
        hour_counts[int(h)] = float(cnt)
    if hour_counts.sum() > 0:
        hour_counts /= hour_counts.sum()
    role_mean = role_profiles[role]
    diff = np.abs(hour_counts - role_mean).mean()
    return _clip01(diff * 10.0)


def feat_team_deviation(udf: pd.DataFrame,
                         team_profiles: Dict[str, np.ndarray]) -> float:
    team = udf["team"].iloc[0] if "team" in udf.columns else None
    if team is None or team not in team_profiles:
        return 0.0
    hour_counts = np.zeros(24, dtype=float)
    for h, cnt in udf.groupby("hour").size().items():
        hour_counts[int(h)] = float(cnt)
    if hour_counts.sum() > 0:
        hour_counts /= hour_counts.sum()
    team_mean = team_profiles[team]
    diff = np.abs(hour_counts - team_mean).mean()
    return _clip01(diff * 10.0)


def feat_behavior_trend(udf: pd.DataFrame, bl: Dict) -> float:
    """Linear trend of daily activity (slope > 0 → increasing anomaly)."""
    udf2 = udf.copy(); udf2["date"] = udf["timestamp"].dt.date
    dc    = udf2.groupby("date").size().values.astype(float)
    if len(dc) < 3:
        return 0.0
    x    = np.arange(len(dc), dtype=float)
    slope = float(np.polyfit(x, dc, 1)[0])
    # Normalise: slope of bl["daily_std"] per day → 0.5
    norm_slope = slope / max(bl["daily_std"], 1e-9)
    return _clip01((norm_slope + 3.0) / 6.0)


def feat_temporal_consistency(udf: pd.DataFrame, bl: Dict) -> float:
    """Cosine similarity of user's hour profile vs historical hour profile."""
    hour_counts = np.zeros(24, dtype=float)
    for h, cnt in udf.groupby("hour").size().items():
        hour_counts[int(h)] = float(cnt)
    if hour_counts.sum() == 0:
        return 0.0
    hour_counts /= hour_counts.sum()
    ref = bl["hour_profile"]
    dot = float(np.dot(hour_counts, ref))
    nr  = float(np.linalg.norm(hour_counts) * np.linalg.norm(ref))
    sim = dot / max(nr, 1e-9)
    # Low consistency → high score
    return _clip01(1.0 - sim)


def feat_risk_acceleration(udf: pd.DataFrame, bl: Dict,
                            short: int = SHORT_WIN,
                            medium: int = RECENT_WIN) -> float:
    """
    Second-order change: short-window risk vs medium-window risk vs long-term.
    RiskAcceleration = (short_mean - medium_mean) / max(medium_mean, 1e-9).
    """
    udf2 = udf.copy(); udf2["date"] = udf["timestamp"].dt.date
    t_max    = udf["timestamp"].max()
    cut_s    = t_max - pd.Timedelta(days=short)
    cut_m    = t_max - pd.Timedelta(days=medium)
    short_c  = udf2.loc[udf["timestamp"] >= cut_s].groupby("date").size().values
    medium_c = udf2.loc[(udf["timestamp"] >= cut_m) &
                         (udf["timestamp"] < cut_s)].groupby("date").size().values
    short_m  = float(np.mean(short_c))  if len(short_c)  > 0 else bl["daily_mean"]
    medium_m = float(np.mean(medium_c)) if len(medium_c) > 0 else bl["daily_mean"]
    accel    = (short_m - medium_m) / max(medium_m, 1e-9)
    return _clip01(accel / 2.0 + 0.5)  # centre at 0.5; >0.5 = accelerating


# ---------------------------------------------------------------------------
# Group (dept / role / team) profiles for deviation features
# ---------------------------------------------------------------------------


def build_group_profiles(df: pd.DataFrame,
                         col: str) -> Dict[str, np.ndarray]:
    if col not in df.columns:
        return {}
    profiles: Dict[str, np.ndarray] = {}
    for grp_val, gdf in df.groupby(col):
        hour_counts = np.zeros(24, dtype=float)
        for h, cnt in gdf.groupby("hour").size().items():
            hour_counts[int(h)] = float(cnt)
        if hour_counts.sum() > 0:
            hour_counts /= hour_counts.sum()
        profiles[str(grp_val)] = hour_counts
    return profiles


# ---------------------------------------------------------------------------
# LDAP & psychometric enrichment
# ---------------------------------------------------------------------------


def load_ldap(path: Path) -> pd.DataFrame:
    if not path.exists():
        log.warning("LDAP.csv not found at %s", path)
        return pd.DataFrame(columns=["user_id"])
    log.info("Loading LDAP data from %s …", path)
    ldap = pd.read_csv(path, dtype=str)
    ldap.columns = [c.strip().lower() for c in ldap.columns]
    col_map = {}
    for c in ldap.columns:
        if "user" in c and "id" in c:
            col_map[c] = "user_id"
        elif "dept" in c or "department" in c:
            col_map[c] = "department"
        elif "role" in c or "title" in c:
            col_map[c] = "role"
        elif "team" in c or "group" in c:
            col_map[c] = "team"
    ldap = ldap.rename(columns=col_map)
    if "user_id" not in ldap.columns:
        log.warning("LDAP.csv has no user_id column – skipping.")
        return pd.DataFrame(columns=["user_id"])
    keep = [c for c in ["user_id", "department", "role", "team"] if c in ldap.columns]
    return ldap[keep].drop_duplicates("user_id")


def load_psychometric(path: Path) -> pd.DataFrame:
    if not path.exists():
        log.warning("psychometric.csv not found at %s", path)
        return pd.DataFrame(columns=["user_id"])
    log.info("Loading psychometric data from %s …", path)
    psych = pd.read_csv(path, dtype=str)
    psych.columns = [c.strip().lower() for c in psych.columns]
    if "user_id" not in psych.columns:
        for c in psych.columns:
            if "user" in c:
                psych = psych.rename(columns={c: "user_id"})
                break
    if "user_id" not in psych.columns:
        return pd.DataFrame(columns=["user_id"])
    num_cols = []
    for c in psych.columns:
        if c == "user_id":
            continue
        try:
            psych[c] = pd.to_numeric(psych[c], errors="coerce")
            num_cols.append(c)
        except Exception:
            pass
    return psych[["user_id"] + num_cols].drop_duplicates("user_id")


# ---------------------------------------------------------------------------
# Per-user feature computation dispatcher
# ---------------------------------------------------------------------------


def compute_user_features(uid: str,
                            udf: pd.DataFrame,
                            bl: Dict,
                            dept_profiles: Dict,
                            role_profiles: Dict,
                            team_profiles: Dict) -> Dict[str, Any]:
    sm, ss = feat_session_duration_stats(udf)
    return {
        "user_id":                      uid,
        "AfterHoursScore":              feat_after_hours_score(udf, bl),
        "CommunicationReduction":       feat_communication_reduction(udf, bl),
        "ResourceExploration":          feat_resource_exploration(udf, bl),
        "BehaviorDeviation":            feat_behavior_deviation(udf, bl),
        "AccessFrequencyShift":         feat_access_frequency_shift(udf, bl),
        "WeekendActivityScore":         feat_weekend_activity(udf),
        "FailedLoginRate":              feat_failed_login_rate(udf),
        "UniquePCUsage":                feat_unique_pc_usage(udf),
        "UniqueWebsiteUsage":           feat_unique_website_usage(udf),
        "UniqueFileTypeUsage":          feat_unique_filetype_usage(udf),
        "USBUsageFrequency":            feat_usb_usage_frequency(udf),
        "EmailAttachmentFrequency":     feat_email_attachment_frequency(udf),
        "ExternalCommunicationScore":   feat_external_communication(udf),
        "PrivilegeEscalationIndicator": feat_privilege_escalation(udf),
        "SessionDurationMean":          sm,
        "SessionDurationStd":           ss,
        "BurstActivityScore":           feat_burst_activity(udf),
        "NightActivityScore":           feat_night_activity(udf),
        "ResourceEntropy":              feat_resource_entropy(udf),
        "BehaviorConsistency":          feat_behavior_consistency(udf, bl),
        "CommunicationEntropy":         feat_communication_entropy(udf),
        "ActivityEntropy":              feat_activity_entropy(udf),
        "DeviceSwitchRate":             feat_device_switch_rate(udf),
        "ResourceNoveltyScore":         feat_resource_novelty(udf, bl),
        "OffHoursResourceAccess":       feat_off_hours_resource_access(udf),
        "SessionIrregularity":          feat_session_irregularity(udf, bl),
        "CommunicationVolatility":      feat_communication_volatility(udf, bl),
        "WebsiteNoveltyScore":          feat_website_novelty(udf, bl),
        "FileNoveltyScore":             feat_file_novelty(udf, bl),
        "USBNoveltyScore":              feat_usb_novelty(udf, bl),
        "DepartmentDeviation":          feat_department_deviation(udf, dept_profiles),
        "RoleDeviation":                feat_role_deviation(udf, role_profiles),
        "TeamDeviation":                feat_team_deviation(udf, team_profiles),
        "BehaviorTrendScore":           feat_behavior_trend(udf, bl),
        "TemporalConsistency":          feat_temporal_consistency(udf, bl),
        "RiskAcceleration":             feat_risk_acceleration(udf, bl),
    }


# ---------------------------------------------------------------------------
# Step 6 – Normalization
# ---------------------------------------------------------------------------


def normalize_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    log.info("Normalizing features …")
    feature_cols = [c for c in REQUIRED_COLS if c not in ("user_id", "PsychologyScore")]
    scaler_info: Dict[str, Any] = {}

    robust_cols = [c for c in feature_cols if c in ROBUST_FEATURES and c in df.columns]
    minmax_cols = [c for c in feature_cols if c not in ROBUST_FEATURES and c in df.columns]

    if robust_cols:
        rs = RobustScaler(quantile_range=(10, 90))
        df[robust_cols] = rs.fit_transform(df[robust_cols].values)
        df[robust_cols] = df[robust_cols].clip(0.0, 1.0)
        scaler_info["robust"] = {"columns": robust_cols,
                                  "center": rs.center_.tolist(),
                                  "scale": rs.scale_.tolist()}

    if minmax_cols:
        mm = MinMaxScaler(feature_range=(0.0, 1.0))
        df[minmax_cols] = mm.fit_transform(df[minmax_cols].values)
        scaler_info["minmax"] = {"columns": minmax_cols,
                                  "scale": mm.scale_.tolist(),
                                  "min": mm.min_.tolist()}

    # Hard clip everything to [0,1]
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].clip(0.0, 1.0).fillna(0.0)

    return df, scaler_info


# ---------------------------------------------------------------------------
# Step 7 – PsychologyScore
# ---------------------------------------------------------------------------


def compute_psychology_score(row: pd.Series) -> float:
    total_w = 0.0
    weighted = 0.0
    for col, w in PS_WEIGHTS.items():
        if col in row.index:
            total_w   += w
            weighted  += w * float(row[col])
    return _clip01(weighted / total_w) if total_w > 0 else 0.0


# ---------------------------------------------------------------------------
# Step 10 – Metadata
# ---------------------------------------------------------------------------


def build_metadata(df: pd.DataFrame, scaler_info: Dict) -> Dict:
    feature_cols = [c for c in REQUIRED_COLS if c != "user_id"]
    stats: Dict[str, Dict] = {}
    for col in feature_cols:
        if col in df.columns:
            arr = df[col].values
            stats[col] = {
                "mean":   float(np.mean(arr)),
                "std":    float(np.std(arr, ddof=0)),
                "min":    float(np.min(arr)),
                "max":    float(np.max(arr)),
                "median": float(np.median(arr)),
                "iqr":    float(iqr(arr)),
            }
    return {
        "feature_names":       feature_cols,
        "feature_statistics":  stats,
        "normalization_method": scaler_info,
        "feature_ranges":      {col: (stats[col]["min"], stats[col]["max"])
                                 for col in stats},
        "robust_features":     list(ROBUST_FEATURES),
        "minmax_features":     [c for c in feature_cols if c not in ROBUST_FEATURES],
        "generated_at":        datetime.utcnow().isoformat(),
        "ps_weights":          PS_WEIGHTS,
        "bd_weights":          BD_WEIGHTS,
    }


# ---------------------------------------------------------------------------
# Step 12 – Validation
# ---------------------------------------------------------------------------


def validate(df: pd.DataFrame) -> None:
    log.info("Validating output …")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "BehaviorDeviation" not in df.columns or df["BehaviorDeviation"].isna().all():
        raise ValueError("BehaviorDeviation column is missing or entirely NaN.")

    if "PsychologyScore" not in df.columns or df["PsychologyScore"].isna().all():
        raise ValueError("PsychologyScore column is missing or entirely NaN.")

    nan_cols = [c for c in REQUIRED_COLS if df[c].isna().any()]
    if nan_cols:
        raise ValueError(f"NaN values found in columns: {nan_cols}")

    if df["user_id"].duplicated().any():
        dups = df.loc[df["user_id"].duplicated(keep=False), "user_id"].unique().tolist()
        raise ValueError(f"Duplicate user_ids detected: {dups[:10]}")

    if len(df) != df["user_id"].nunique():
        raise ValueError("Row count does not equal unique user count.")

    score_cols = [c for c in REQUIRED_COLS if c != "user_id"]
    for col in score_cols:
        lo = float(df[col].min())
        hi = float(df[col].max())
        if lo < -1e-5 or hi > 1.0 + 1e-5:
            raise ValueError(f"Column '{col}' out of [0,1]: min={lo:.6f}, max={hi:.6f}")

    log.info("Validation passed — %d users, %d feature columns.",
             len(df), len(df.columns) - 1)


# ---------------------------------------------------------------------------
# Step 9 – Chunk-based CSV loading
# ---------------------------------------------------------------------------


def load_events(path: Path, chunksize: int = CHUNK_SIZE) -> Tuple[pd.DataFrame, int]:
    log.info("Streaming events from '%s' (chunk=%d) …", path, chunksize)
    dtypes = {"user_id": str, "event_type": str, "resource": str, "target_user": str}
    chunks = []
    total  = 0
    chunks_processed = 0
    
    for i, chunk in enumerate(pd.read_csv(path, dtype=dtypes, chunksize=chunksize,
                                           low_memory=False), start=1):
        chunk  = parse_and_sort(chunk)
        chunks.append(chunk)
        total += len(chunk)
        chunks_processed = i
        if i % 10 == 0:
            log.info("  Chunk %d complete — %d rows total", i, total)
            
        if TEST_MODE and i >= MAX_TEST_CHUNKS:
            log.info("TEST MODE: stopping after %d chunks.", MAX_TEST_CHUNKS)
            break
            
    log.info("All chunks loaded. Total rows: %d", total)
    df = pd.concat(chunks, ignore_index=True)
    del chunks; gc.collect()
    return df, chunks_processed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    t0 = time.perf_counter()
    tracemalloc.start()

    log.info("=" * 72)
    log.info("Phase 3  –  Psychology Feature Engineering")
    log.info("=" * 72)

    # ── Input validation ─────────────────────────────────────────────────────
    if not UNUnified_PATH.exists():
        raise FileNotFoundError(f"Input not found: {UNUnified_PATH}")

    # ── Load events ──────────────────────────────────────────────────────────
    events, chunks_processed = load_events(UNUnified_PATH, chunksize=CHUNK_SIZE)
    events = events.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    log.info("Unique users: %d | Date range: %s → %s",
             events["user_id"].nunique(),
             events["timestamp"].min(),
             events["timestamp"].max())

    # ── LDAP enrichment ──────────────────────────────────────────────────────
    ldap  = load_ldap(LDAP_PATH)
    psych = load_psychometric(PSYCH_PATH)

    if "user_id" in ldap.columns and len(ldap) > 0:
        for col in ["department", "role", "team"]:
            if col in ldap.columns:
                uid_to_val = dict(zip(ldap["user_id"], ldap[col]))
                events[col] = events["user_id"].map(uid_to_val).fillna("unknown")

    # ── Group profiles ───────────────────────────────────────────────────────
    log.info("Building group activity profiles …")
    dept_profiles = build_group_profiles(events, "department")
    role_profiles = build_group_profiles(events, "role")
    team_profiles = build_group_profiles(events, "team")

    # ── Baselines ────────────────────────────────────────────────────────────
    baselines = build_baselines(events)

    # ── Per-user feature extraction ──────────────────────────────────────────
    log.info("Extracting per-user behavioural features …")
    rows: List[Dict] = []
    n_users = events["user_id"].nunique()
    for i, (uid, udf) in enumerate(events.groupby("user_id"), 1):
        bl = baselines.get(uid)
        if bl is None:
            log.warning("No baseline for user '%s' – skipping.", uid)
            continue
        try:
            row = compute_user_features(uid, udf, bl,
                                         dept_profiles, role_profiles, team_profiles)
            rows.append(row)
        except Exception as exc:
            log.error("Failed for user '%s': %s", uid, exc)
            log.debug(traceback.format_exc())
        if i % 500 == 0 or i == n_users:
            log.info("  … %d / %d users processed", i, n_users)

    if not rows:
        raise RuntimeError("No features generated – check input data.")

    features_df = pd.DataFrame(rows)
    del events, baselines; gc.collect()

    # ── Psychometric merge ───────────────────────────────────────────────────
    if len(psych) > 0 and "user_id" in psych.columns:
        log.info("Merging psychometric data …")
        num_pcols = [c for c in psych.columns if c != "user_id"]
        features_df = features_df.merge(psych[["user_id"] + num_pcols],
                                         on="user_id", how="left")
        for col in num_pcols:
            if col in features_df.columns:
                features_df[col] = pd.to_numeric(features_df[col], errors="coerce")
                col_min = features_df[col].min()
                col_max = features_df[col].max()
                if col_max > col_min:
                    features_df[col] = (features_df[col] - col_min) / (col_max - col_min)
                else:
                    features_df[col] = 0.0
                features_df[col] = features_df[col].fillna(0.0).clip(0.0, 1.0)

    # ── Normalization ────────────────────────────────────────────────────────
    features_df, scaler_info = normalize_features(features_df)

    # ── PsychologyScore ──────────────────────────────────────────────────────
    log.info("Computing PsychologyScore …")
    features_df["PsychologyScore"] = features_df.apply(
        compute_psychology_score, axis=1
    ).clip(0.0, 1.0)

    # ── Ensure all required columns present & fill NaN ───────────────────────
    for col in REQUIRED_COLS:
        if col not in features_df.columns:
            features_df[col] = 0.0
        elif col != "user_id":
            features_df[col] = features_df[col].fillna(0.0).clip(0.0, 1.0)

    # ── Re-order columns ─────────────────────────────────────────────────────
    extra = [c for c in features_df.columns if c not in REQUIRED_COLS]
    features_df = features_df[REQUIRED_COLS + extra]

    # ── Cast to float32 for TGN compatibility ────────────────────────────────
    for col in REQUIRED_COLS:
        if col != "user_id":
            features_df[col] = features_df[col].astype(np.float32)

    # ── Validate ─────────────────────────────────────────────────────────────
    validate(features_df)

    # ── Save output ──────────────────────────────────────────────────────────
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(OUTPUT_CSV, index=False)
    log.info("Features saved to '%s'", OUTPUT_CSV)

    # ── Metadata ─────────────────────────────────────────────────────────────
    log.info("Saving metadata to '%s' …", OUTPUT_META)
    meta = build_metadata(features_df, scaler_info)
    with open(OUTPUT_META, "wb") as fh:
        pickle.dump(meta, fh, protocol=pickle.HIGHEST_PROTOCOL)

    # ── Statistics summary ───────────────────────────────────────────────────
    score_cols = [c for c in REQUIRED_COLS if c != "user_id"]
    log.info("Feature summary:")
    for col in score_cols:
        if col in features_df.columns:
            arr = features_df[col].values
            log.info("  %-38s  mean=%6.4f  std=%6.4f  min=%6.4f  max=%6.4f",
                     col, arr.mean(), arr.std(), arr.min(), arr.max())

    # ── Timing & memory ──────────────────────────────────────────────────────
    t1 = time.perf_counter()
    cur_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    log.info("=" * 72)
    log.info("Processed chunks : %d", chunks_processed)
    log.info("Rows processed   : %d", len(features_df))
    log.info("Execution time   : %.2f s", t1 - t0)
    log.info("Peak RAM usage   : %.2f MB", peak_mem / 1024 / 1024)
    log.info("Output columns   : %d", len(features_df.columns))
    log.info("=" * 72)


if __name__ == "__main__":
    main()