"""
preprocessing/file_sensitivity_scoring.py

Phase 4 — File Sensitivity Scoring (CERT dataset, file.csv schema).

This script is built against the EXACT schema confirmed for this project's
data/raw/file.csv:

    id, date, user, pc, filename, content

It does NOT assume any of the following columns exist, because they do not
exist in this dataset: resource, extension, timestamp, event_type. Every
value used downstream is either parsed directly from the five real columns
above, or derived from them.

What each output score actually measures, and where it comes from:

    KeywordScore        : Per-file. Computed from `content` via a fixed,
                           editable keyword list (EXTENSION_RISK_MAP /
                           SENSITIVE_KEYWORDS below). Case-insensitive
                           whole-word matching, normalized to [0, 1].

    ExtensionRisk        : Per-file. Looked up from a fixed, editable
                           EXTENSION_RISK_MAP keyed on the extension parsed
                           out of `filename`. Unmapped extensions get a
                           documented default.

    AccessFrequency      : Per-file. Count of rows in file.csv referencing
                           that exact filename, accumulated across the
                           full streamed pass, then min-max normalized.

    UserDiversity        : Per-file. Count of distinct `user` values that
                           accessed that filename, accumulated across the
                           full streamed pass, then min-max normalized.

    DepartmentDiversity  : Per-file. Count of distinct departments (joined
                           from data/raw/LDAP/*.csv via user -> department,
                           same convention build_graph.py already uses)
                           among the users who accessed that filename, then
                           min-max normalized. If no LDAP file is found,
                           this column is explicitly 0.0 for every row, and
                           a loud warning is logged — it is never silently
                           guessed.

    TemporalImportance   : Per-file. Fraction of that file's accesses that
                           fall on a CERT-standard "after-hours" window
                           (before 07:00 or after 19:00, or on a weekend),
                           derived from the real `date` column. This is a
                           real signal computed from real timestamps in
                           your data — not a placeholder.

    SensitivityScore     : Weighted combination of the six scores above.
                           Weights are explicit constants (SCORE_WEIGHTS)
                           you can tune; not derived from anything hidden.

This script streams file.csv in chunks and never loads the full file into
memory. It performs two passes over the file:
    Pass 1: accumulate per-filename counts/sets/keyword sums needed for
            AccessFrequency, UserDiversity, DepartmentDiversity,
            TemporalImportance, and KeywordScore.
    Pass 2: is NOT needed — KeywordScore is accumulated in Pass 1 directly
            from `content` per chunk (mean across all rows for that
            filename), so this script reads file.csv exactly ONCE.

Output: data/processed/file_sensitivity.csv with columns, in this exact
order:
    filename, extension, KeywordScore, ExtensionRisk, AccessFrequency,
    UserDiversity, DepartmentDiversity, TemporalImportance,
    SensitivityScore

This output is directly compatible with the existing graph/edge_features.py,
which auto-detects the filename and sensitivity columns by name rather than
hardcoded position.

Run:
    python preprocessing/file_sensitivity_scoring.py
"""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def log_banner(title: str) -> None:
    """Print a visually distinct stage banner to the log."""
    bar = "=" * 60
    log.info(bar)
    log.info(title)
    log.info(bar)


# =========================================================
# MEMORY HELPER (psutil optional, never blocks execution)
# =========================================================

def get_current_memory_mb() -> float:
    """Return current process RSS memory in MB via psutil, or 0.0 if unavailable."""
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


class PeakMemoryTracker:
    """Tracks the highest RSS observed across repeated `.sample()` calls."""

    def __init__(self) -> None:
        self.peak_mb: float = 0.0
        self._psutil_available = self._check_psutil()

    @staticmethod
    def _check_psutil() -> bool:
        try:
            import psutil  # noqa: F401
            return True
        except ImportError:
            return False

    def sample(self) -> float:
        current = get_current_memory_mb()
        if current > self.peak_mb:
            self.peak_mb = current
        return current


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LDAP_DIR = RAW_DIR / "LDAP"

FILE_CSV_PATH = RAW_DIR / "file.csv"
OUTPUT_PATH = PROCESSED_DIR / "file_sensitivity.csv"

CHUNK_SIZE = 250_000

# =========================================================
# CONFIGURATION — explicit, editable, auditable
# =========================================================

# Case-insensitive, whole-word keyword groups used to compute KeywordScore
# from the `content` column. Each matched keyword in a row contributes to
# that row's raw score; the per-filename KeywordScore is the mean raw
# score across all rows for that filename, then min-max normalized.
SENSITIVE_KEYWORDS: List[str] = [
    # financial
    "salary", "payroll", "budget", "invoice", "bank account", "ssn",
    "social security", "credit card", "tax", "revenue", "wire transfer",
    # legal / confidential
    "confidential", "classified", "nda", "non-disclosure", "proprietary",
    "trade secret", "litigation", "lawsuit", "settlement",
    # credentials / security
    "password", "credentials", "private key", "api key", "secret key",
    "login", "vpn", "admin access",
    # HR / personnel
    "termination", "layoff", "resignation", "disciplinary", "performance review",
    # strategic / competitive
    "merger", "acquisition", "patent", "roadmap", "strategic plan",
]
_KEYWORD_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in SENSITIVE_KEYWORDS) + r")\b",
    re.IGNORECASE,
)

# Extension -> base risk score in [0, 1]. Unmapped extensions fall back to
# EXTENSION_RISK_DEFAULT. Higher = inherently riskier file type to exfiltrate.
EXTENSION_RISK_MAP: Dict[str, float] = {
    # executables / scripts — highest risk
    ".exe": 0.95, ".bat": 0.9, ".sh": 0.9, ".ps1": 0.9, ".vbs": 0.9,
    ".dll": 0.85, ".msi": 0.85,
    # archives — high risk (common exfiltration container)
    ".zip": 0.8, ".rar": 0.8, ".7z": 0.8, ".tar": 0.75, ".gz": 0.75,
    # documents — medium-high risk
    ".pdf": 0.6, ".doc": 0.65, ".docx": 0.65, ".rtf": 0.55,
    # spreadsheets — medium-high (often financial/HR data)
    ".xls": 0.7, ".xlsx": 0.7, ".csv": 0.65,
    # presentations — medium
    ".ppt": 0.5, ".pptx": 0.5,
    # source code — medium-high (IP)
    ".py": 0.7, ".java": 0.7, ".cpp": 0.7, ".c": 0.7, ".js": 0.65, ".sql": 0.7,
    # plain text / email — medium
    ".txt": 0.4, ".eml": 0.45, ".msg": 0.45,
    # media — low risk
    ".jpg": 0.15, ".jpeg": 0.15, ".png": 0.15, ".gif": 0.1,
    ".mp3": 0.1, ".mp4": 0.15, ".avi": 0.15,
    # web / config — low-medium
    ".html": 0.3, ".xml": 0.35, ".json": 0.35, ".log": 0.3,
}
EXTENSION_RISK_DEFAULT = 0.3

# After-hours window for TemporalImportance (24h clock, local to the
# timestamps as recorded in file.csv — CERT dataset convention).
AFTER_HOURS_START = 19  # 7 PM
AFTER_HOURS_END = 7     # 7 AM

# Final SensitivityScore weighting. Must sum to 1.0; validated at startup.
SCORE_WEIGHTS: Dict[str, float] = {
    "KeywordScore": 0.25,
    "ExtensionRisk": 0.20,
    "AccessFrequency": 0.10,
    "UserDiversity": 0.15,
    "DepartmentDiversity": 0.15,
    "TemporalImportance": 0.15,
}

OUTPUT_COLUMNS = [
    "filename",
    "extension",
    "KeywordScore",
    "ExtensionRisk",
    "AccessFrequency",
    "UserDiversity",
    "DepartmentDiversity",
    "TemporalImportance",
    "SensitivityScore",
]


# =========================================================
# UTILITIES
# =========================================================

def normalize_key(value) -> Optional[str]:
    """Normalize a string key (filename, user, department) to stripped lowercase."""
    if pd.isna(value):
        return None
    s = str(value).strip().lower()
    return s if s else None


def extract_extension(filename: str) -> str:
    """
    Extract a normalized file extension from a filename string.

    Args:
        filename: Raw filename, e.g. "report.PDF" or "archive.tar.gz".

    Returns:
        Lowercase extension including the leading dot (e.g. ".pdf"), or
        ".unknown" if the filename has no extension.
    """
    if not filename or "." not in filename:
        return ".unknown"
    ext = "." + filename.rsplit(".", 1)[-1].strip().lower()
    # Guard against pathological cases like "report." or a bare dot
    if ext == "." or len(ext) > 10:
        return ".unknown"
    return ext


def keyword_score_for_text(text: str) -> float:
    """
    Compute a raw (unnormalized) keyword sensitivity score for one row's
    content string, by counting distinct sensitive keyword matches.

    Args:
        text: Raw `content` field value for one row.

    Returns:
        Float count of distinct sensitive keywords matched (whole-word,
        case-insensitive). Not yet normalized to [0, 1] — normalization
        happens once across the full per-filename aggregate.
    """
    if not text:
        return 0.0
    matches = _KEYWORD_PATTERN.findall(text)
    return float(len(set(m.lower() for m in matches)))


def is_after_hours(dt: pd.Timestamp) -> bool:
    """
    Determine whether a timestamp falls in the after-hours window.

    Args:
        dt: A parsed pandas Timestamp.

    Returns:
        True if the timestamp is on a weekend, or outside
        [AFTER_HOURS_END, AFTER_HOURS_START) on a weekday.
    """
    if dt.weekday() >= 5:  # Saturday=5, Sunday=6
        return True
    hour = dt.hour
    return hour >= AFTER_HOURS_START or hour < AFTER_HOURS_END


def min_max_normalize(values: np.ndarray) -> np.ndarray:
    """
    Min-max normalize an array into [0, 1]. Degenerate (constant) arrays
    are mapped to a constant 0.5 rather than producing NaN or all-zero,
    since "no variation" is a real, distinct outcome from "minimum value".

    Args:
        values: 1-D array of raw scores.

    Returns:
        Array of the same shape, normalized into [0, 1].
    """
    if values.size == 0:
        return values
    v_min = values.min()
    v_max = values.max()
    if v_max <= v_min:
        return np.full_like(values, 0.5, dtype=np.float64)
    return (values - v_min) / (v_max - v_min)


def validate_weights(weights: Dict[str, float]) -> None:
    """
    Validate that SCORE_WEIGHTS sums to 1.0 (within floating-point tolerance).

    Args:
        weights: The weight dict to validate.

    Raises:
        ValueError: if the weights do not sum to 1.0.
    """
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"SCORE_WEIGHTS must sum to 1.0, got {total}.")


# =========================================================
# STEP 1 — LOAD DEPARTMENT MAPPING (LDAP, same convention as build_graph.py)
# =========================================================

def detect_latest_ldap_file(ldap_dir: Path) -> Optional[Path]:
    """
    Return the most recently modified CSV in the LDAP directory, or None
    if the directory does not exist or contains no CSVs.

    Args:
        ldap_dir: Path to the LDAP directory.

    Returns:
        Path to the latest LDAP CSV, or None.
    """
    if not ldap_dir.exists():
        return None
    csv_files = list(ldap_dir.glob("*.csv"))
    if not csv_files:
        return None
    return max(csv_files, key=lambda p: p.stat().st_mtime)


def load_user_to_department(ldap_dir: Path) -> Dict[str, str]:
    """
    Build a user -> department mapping from the latest LDAP CSV, using the
    same flexible column-detection convention as build_graph.py.

    Args:
        ldap_dir: Path to the LDAP directory.

    Returns:
        Dict mapping normalized user id -> department string. Empty dict
        if no LDAP file is found (callers must treat this as "department
        data unavailable", not as "everyone has no department").
    """
    ldap_path = detect_latest_ldap_file(ldap_dir)
    if ldap_path is None:
        log.warning(
            f"No LDAP file found under {ldap_dir}; DepartmentDiversity will "
            f"be 0.0 for every row. This is a real data-availability gap, "
            f"not an approximation — install LDAP data to populate this score."
        )
        return {}

    log.info(f"Loading LDAP department mapping from {ldap_path.name} ...")
    df = pd.read_csv(ldap_path)

    columns_lower = {c.strip().lower().replace(" ", "_"): c for c in df.columns}

    def get_col(candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in columns_lower:
                return columns_lower[c]
        return None

    uid_col = get_col(["user_id", "userid", "id", "user"])
    dept_col = get_col(["department", "dept"])

    if uid_col is None:
        log.warning(
            f"LDAP file {ldap_path.name} has no recognizable user id column "
            f"(tried: user_id, userid, id, user); DepartmentDiversity will be 0.0."
        )
        return {}
    if dept_col is None:
        log.warning(
            f"LDAP file {ldap_path.name} has no recognizable department column "
            f"(tried: department, dept); DepartmentDiversity will be 0.0."
        )
        return {}

    df[uid_col] = df[uid_col].apply(normalize_key)
    df = df.dropna(subset=[uid_col]).drop_duplicates(subset=[uid_col])

    mapping: Dict[str, str] = {}
    for uid, dept in zip(df[uid_col], df[dept_col]):
        dept_clean = normalize_key(dept) or "unknown"
        mapping[uid] = dept_clean

    log.info(f"  Loaded department mapping for {len(mapping)} users.")
    return mapping


# =========================================================
# STEP 2 — PER-FILENAME ACCUMULATOR
# =========================================================

class FileAccumulator:
    """
    Accumulates streaming per-filename statistics across the full
    chunked pass over file.csv, without ever materializing the full
    dataset in memory.

    Tracked per filename:
        access_count          : total row count referencing this filename
        user_set               : set of distinct users who accessed it
        department_set         : set of distinct departments (via user->dept)
        after_hours_count      : count of accesses in the after-hours window
        keyword_score_sum       : running sum of per-row keyword scores
        extension               : the (single, assumed-stable) extension
                                  parsed for this filename
    """

    def __init__(self) -> None:
        self.access_count: Dict[str, int] = defaultdict(int)
        self.user_set: Dict[str, Set[str]] = defaultdict(set)
        self.department_set: Dict[str, Set[str]] = defaultdict(set)
        self.after_hours_count: Dict[str, int] = defaultdict(int)
        self.keyword_score_sum: Dict[str, float] = defaultdict(float)
        self.extension_of: Dict[str, str] = {}

    def update(
        self,
        filename: str,
        user: Optional[str],
        department: Optional[str],
        after_hours: bool,
        keyword_score: float,
        extension: str,
    ) -> None:
        """Fold one row's contribution into the running per-filename stats."""
        self.access_count[filename] += 1
        if user:
            self.user_set[filename].add(user)
        if department:
            self.department_set[filename].add(department)
        if after_hours:
            self.after_hours_count[filename] += 1
        self.keyword_score_sum[filename] += keyword_score
        if filename not in self.extension_of:
            self.extension_of[filename] = extension

    def filenames(self) -> List[str]:
        """Return all distinct filenames observed so far."""
        return list(self.access_count.keys())


# =========================================================
# STEP 3 — STREAM file.csv (single pass)
# =========================================================

def stream_file_csv(
    path: Path,
    user_to_department: Dict[str, str],
    mem_tracker: Optional[PeakMemoryTracker] = None,
) -> Tuple[FileAccumulator, int]:
    """
    Stream file.csv exactly once in chunks, accumulating all statistics
    needed for every output score. Never loads the full file into memory.

    Args:
        path: Path to data/raw/file.csv.
        user_to_department: uid -> department mapping (may be empty).
        mem_tracker: Optional PeakMemoryTracker sampled once per chunk.

    Returns:
        Tuple of (populated FileAccumulator, total row count processed).

    Raises:
        FileNotFoundError: if file.csv does not exist.
        ValueError: if required columns are missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"file.csv not found at {path}.")

    required_cols = ["date", "user", "pc", "filename", "content"]
    accumulator = FileAccumulator()
    total_rows = 0
    chunk_num = 0

    log.info(f"Streaming {path} in chunks of {CHUNK_SIZE:,} rows ...")

    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False):
        chunk_num += 1

        if chunk_num == 1:
            missing = [c for c in required_cols if c not in chunk.columns]
            if missing:
                log.error(f"[file.csv] Missing required columns: {missing}")
                log.error(f"[file.csv] Available columns: {list(chunk.columns)}")
                raise ValueError(
                    f"[file.csv] Missing required columns: {missing}. "
                    f"Available: {list(chunk.columns)}."
                )
            log.info(f"  Columns confirmed: {list(chunk.columns)}")

        log.info(f"Processing chunk {chunk_num} ({len(chunk):,} rows)...")

        # Parse date once per chunk, vectorized
        parsed_dates = pd.to_datetime(chunk["date"], errors="coerce")

        filenames = chunk["filename"].astype(str).str.strip()
        users = chunk["user"].apply(normalize_key)
        contents = chunk["content"].astype(str)

        for i in range(len(chunk)):
            filename = filenames.iat[i].strip()
            if not filename or filename.lower() == "nan":
                continue

            user = users.iat[i]
            content = contents.iat[i]
            dt = parsed_dates.iat[i]

            extension = extract_extension(filename)
            keyword_score = keyword_score_for_text(content)
            after_hours = bool(pd.notna(dt) and is_after_hours(dt))
            department = user_to_department.get(user) if user else None

            accumulator.update(
                filename=filename,
                user=user,
                department=department,
                after_hours=after_hours,
                keyword_score=keyword_score,
                extension=extension,
            )

            total_rows += 1

        if mem_tracker is not None:
            mem_tracker.sample()

        if chunk_num % 5 == 0:
            log.info(f"  Processed {total_rows:,} rows so far, {len(accumulator.filenames()):,} distinct filenames.")

    log.info(f"Streaming complete. Total rows: {total_rows:,}, distinct filenames: {len(accumulator.filenames()):,}")
    return accumulator, total_rows


# =========================================================
# STEP 4 — FINALIZE SCORES
# =========================================================

def finalize_scores(accumulator: FileAccumulator, department_data_available: bool) -> pd.DataFrame:
    """
    Reduce the accumulated per-filename statistics into the final output
    DataFrame, computing all seven score columns plus filename/extension.

    Args:
        accumulator: Populated FileAccumulator from stream_file_csv.
        department_data_available: Whether an LDAP user->department mapping
            was successfully loaded. If False, DepartmentDiversity is set
            to a true 0.0 for every row rather than being min-max
            normalized — normalizing an all-zero array would otherwise
            produce the degenerate-case fallback of 0.5 (see
            min_max_normalize), which would misrepresent "no department
            data available" as "moderate department diversity".

    Returns:
        DataFrame with columns matching OUTPUT_COLUMNS exactly, one row
        per distinct filename.
    """
    filenames = accumulator.filenames()
    n = len(filenames)

    if n == 0:
        log.warning("No filenames were accumulated from file.csv; output will be empty.")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    extensions = np.array([accumulator.extension_of[f] for f in filenames])

    raw_access_count = np.array([accumulator.access_count[f] for f in filenames], dtype=np.float64)
    raw_user_diversity = np.array([len(accumulator.user_set[f]) for f in filenames], dtype=np.float64)
    raw_dept_diversity = np.array([len(accumulator.department_set[f]) for f in filenames], dtype=np.float64)
    raw_keyword_mean = np.array(
        [accumulator.keyword_score_sum[f] / max(accumulator.access_count[f], 1) for f in filenames],
        dtype=np.float64,
    )
    raw_temporal_fraction = np.array(
        [accumulator.after_hours_count[f] / max(accumulator.access_count[f], 1) for f in filenames],
        dtype=np.float64,
    )
    raw_extension_risk = np.array(
        [EXTENSION_RISK_MAP.get(ext, EXTENSION_RISK_DEFAULT) for ext in extensions],
        dtype=np.float64,
    )

    keyword_score = min_max_normalize(raw_keyword_mean)
    extension_risk = raw_extension_risk  # already in [0, 1] by construction; no normalization needed
    access_frequency = min_max_normalize(raw_access_count)
    user_diversity = min_max_normalize(raw_user_diversity)

    if department_data_available:
        department_diversity = min_max_normalize(raw_dept_diversity)
    else:
        # No LDAP mapping was available, so raw_dept_diversity is all
        # zeros by construction. Normalizing an all-zero array would hit
        # min_max_normalize's degenerate-case fallback (0.5), which would
        # misreport "data unavailable" as "moderate diversity". Force a
        # true 0.0 instead.
        department_diversity = np.zeros(n, dtype=np.float64)

    temporal_importance = raw_temporal_fraction  # already a fraction in [0, 1]

    sensitivity_score = (
        SCORE_WEIGHTS["KeywordScore"] * keyword_score
        + SCORE_WEIGHTS["ExtensionRisk"] * extension_risk
        + SCORE_WEIGHTS["AccessFrequency"] * access_frequency
        + SCORE_WEIGHTS["UserDiversity"] * user_diversity
        + SCORE_WEIGHTS["DepartmentDiversity"] * department_diversity
        + SCORE_WEIGHTS["TemporalImportance"] * temporal_importance
    )

    df = pd.DataFrame(
        {
            "filename": filenames,
            "extension": extensions,
            "KeywordScore": keyword_score,
            "ExtensionRisk": extension_risk,
            "AccessFrequency": access_frequency,
            "UserDiversity": user_diversity,
            "DepartmentDiversity": department_diversity,
            "TemporalImportance": temporal_importance,
            "SensitivityScore": sensitivity_score,
        }
    )

    return df[OUTPUT_COLUMNS]


# =========================================================
# STEP 5 — VALIDATION
# =========================================================

def validate_output(df: pd.DataFrame) -> None:
    """
    Defensive validation of the final output DataFrame before saving.

    Checks: required columns present in exact order, no NaN, no Inf,
    all score columns within [0, 1], no duplicate filenames.

    Args:
        df: The finalized output DataFrame.

    Raises:
        ValueError: on any validation failure.
    """
    if list(df.columns) != OUTPUT_COLUMNS:
        raise ValueError(f"Output columns {list(df.columns)} do not match required {OUTPUT_COLUMNS}.")

    score_cols = [
        "KeywordScore", "ExtensionRisk", "AccessFrequency",
        "UserDiversity", "DepartmentDiversity", "TemporalImportance", "SensitivityScore",
    ]

    for col in score_cols:
        if df[col].isna().any():
            raise ValueError(f"NaN values found in column '{col}'.")
        if np.isinf(df[col].to_numpy()).any():
            raise ValueError(f"Inf values found in column '{col}'.")
        if (df[col] < -1e-9).any() or (df[col] > 1 + 1e-9).any():
            raise ValueError(f"Column '{col}' has values outside [0, 1].")

    if df["filename"].duplicated().any():
        raise ValueError("Duplicate filenames found in output — one row per filename is required.")

    log.info(f"  Validation passed — {len(df):,} rows, columns: {list(df.columns)}")


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    start_time = time.time()
    mem_tracker = PeakMemoryTracker()
    mem_tracker.sample()

    if not mem_tracker._psutil_available:
        log.warning("psutil not installed; memory usage will show 0.0 MB. Install with: pip install psutil")

    log_banner("PHASE 4 — FILE SENSITIVITY SCORING (file.csv schema)")

    validate_weights(SCORE_WEIGHTS)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    log_banner("STAGE 1 — LOADING DEPARTMENT MAPPING")
    user_to_department = load_user_to_department(LDAP_DIR)
    mem_tracker.sample()

    log_banner("STAGE 2 — STREAMING file.csv (SINGLE PASS)")
    accumulator, total_rows = stream_file_csv(FILE_CSV_PATH, user_to_department, mem_tracker)
    mem_tracker.sample()

    log_banner("STAGE 3 — FINALIZING SCORES")
    result_df = finalize_scores(accumulator, department_data_available=bool(user_to_department))
    mem_tracker.sample()

    log_banner("STAGE 4 — VALIDATING OUTPUT")
    validate_output(result_df)

    log_banner("STAGE 5 — SAVING OUTPUT")
    result_df.to_csv(OUTPUT_PATH, index=False)
    log.info(f"  Saved {OUTPUT_PATH}")
    mem_tracker.sample()

    elapsed = time.time() - start_time
    peak_mem = mem_tracker.sample()

    log_banner("SUMMARY")
    log.info(f"  Rows processed       : {total_rows:,}")
    log.info(f"  Distinct filenames   : {len(result_df):,}")
    log.info(f"  Department data      : {'available' if user_to_department else 'UNAVAILABLE (DepartmentDiversity=0 for all rows)'}")
    log.info(f"  Output columns       : {list(result_df.columns)}")
    log.info(f"  Execution time       : {elapsed:.2f} seconds")
    log.info(f"  Peak memory usage    : {peak_mem:.2f} MB")
    log_banner("PHASE 4 COMPLETE")


if __name__ == "__main__":
    main()