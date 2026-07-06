"""
fusion/feature_fusion.py
------------------------
Production-grade feature fusion pipeline.

Merges behavior, psychology, LDAP, psychometric, and file-sensitivity
features into a single ML-ready dataset.

Usage
-----
python fusion/feature_fusion.py \
    --behavior   data/processed/behavior_features.csv \
    --psychology data/processed/psychology_features.csv \
    --sensitivity data/processed/file_sensitivity.csv \
    --ldap       data/raw/LDAP.csv \
    --psychometric data/raw/psychometric.csv \
    --output     data/processed/final_features.csv \
    --chunksize 10000 \
    --log-level DEBUG
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _configure_logging(level: str = "INFO") -> logging.Logger:
    """
    Configure root logger with a consistent format and return a module logger.

    Parameters
    ----------
    level:
        One of DEBUG | INFO | WARNING | ERROR | CRITICAL.

    Returns
    -------
    logging.Logger
        Module-level logger named after this file.
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


# Module-level placeholder; replaced inside main() once args are parsed.
log: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Canonical name for the user identifier column after normalisation.
USER_ID_COL: str = "user_id"

# Required columns per source file.
REQUIRED_COLUMNS: dict[str, list[str]] = {
    "behavior":     [USER_ID_COL],
    "psychology":   [USER_ID_COL],
    "ldap":         [USER_ID_COL],
    "psychometric": [USER_ID_COL],
    # file_sensitivity needs a user_id AND a numeric sensitivity score column;
    # the exact sensitivity column name is configurable via CLI.
    "sensitivity":  [USER_ID_COL],
}

# Aggregated column names produced from file_sensitivity.
AGG_AVG_COL   = "AverageSensitivity"
AGG_MAX_COL   = "MaxSensitivity"
AGG_COUNT_COL = "SensitiveFileCount"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _resolve_path(p: str | Path) -> Path:
    """Convert a string to an absolute Path and verify the file exists."""
    path = Path(p).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    return path


def _safe_read_csv(
    path: Path,
    chunksize: Optional[int] = None,
    dtype: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Read a CSV file with robust error handling.

    Reads in chunks when *chunksize* is set so that large files do not
    exhaust memory; concatenates into a single DataFrame before returning
    so that the rest of the pipeline sees a uniform interface.

    Parameters
    ----------
    path:
        Absolute path to the CSV file.
    chunksize:
        Number of rows per chunk.  Pass None to read the whole file at once.
    dtype:
        Optional dtype map forwarded to pandas.

    Returns
    -------
    pd.DataFrame
    """
    log.info("Reading %s …", path.name)
    try:
        if chunksize:
            chunks = pd.read_csv(
                path,
                chunksize=chunksize,
                dtype=dtype,
                low_memory=False,
            )
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(path, dtype=dtype, low_memory=False)
    except pd.errors.EmptyDataError:
        log.warning("File %s is empty — returning empty DataFrame.", path.name)
        return pd.DataFrame()
    except pd.errors.ParserError as exc:
        raise ValueError(f"Failed to parse {path}: {exc}") from exc

    log.debug("  → %d rows × %d cols loaded from %s", *df.shape, path.name)
    return df


# ---------------------------------------------------------------------------
# Validation & normalisation
# ---------------------------------------------------------------------------

def _validate_columns(df: pd.DataFrame, source: str, required: list[str]) -> None:
    """
    Raise ValueError when *required* columns are absent from *df*.

    Parameters
    ----------
    df:
        DataFrame to validate.
    source:
        Human-readable label used in error messages.
    required:
        List of column names that must be present.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{source}] Missing required columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )
    log.debug("[%s] Column validation passed.", source)


def _normalize_user_id(df: pd.DataFrame, alias_map: dict[str, str]) -> pd.DataFrame:
    """
    Rename any known user-ID aliases to the canonical USER_ID_COL name and
    ensure the column is stored as a stripped, lower-cased string so that
    joins are case- and whitespace-insensitive.

    Parameters
    ----------
    df:
        Input DataFrame (modified in-place via copy).
    alias_map:
        Mapping of {alias_col_name: USER_ID_COL}.  Only applied when
        USER_ID_COL is absent from *df*.

    Returns
    -------
    pd.DataFrame
        DataFrame with a normalised USER_ID_COL column.
    """
    df = df.copy()

    # Rename alias → canonical only when the canonical name is absent.
    if USER_ID_COL not in df.columns:
        for alias, canonical in alias_map.items():
            if alias in df.columns:
                log.debug("  Renaming column '%s' → '%s'.", alias, canonical)
                df = df.rename(columns={alias: canonical})
                break

    # Coerce to str, strip whitespace, lower-case.
    df[USER_ID_COL] = (
        df[USER_ID_COL]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    return df


def _drop_duplicate_users(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Remove rows with duplicate USER_ID_COL values, keeping the first
    occurrence.  Logs how many rows were removed.

    Parameters
    ----------
    df:
        DataFrame that must already contain USER_ID_COL.
    source:
        Label used in log messages.

    Returns
    -------
    pd.DataFrame
        Deduplicated DataFrame.
    """
    n_before = len(df)
    df = df.drop_duplicates(subset=[USER_ID_COL], keep="first")
    n_dropped = n_before - len(df)
    if n_dropped:
        log.warning(
            "[%s] Dropped %d duplicate user(s) (kept first occurrence).",
            source,
            n_dropped,
        )
    else:
        log.debug("[%s] No duplicate users found.", source)
    return df


# ---------------------------------------------------------------------------
# File-sensitivity aggregation
# ---------------------------------------------------------------------------

def _aggregate_sensitivity(
    df: pd.DataFrame,
    sensitivity_col: str,
) -> pd.DataFrame:
    """
    Collapse file-level sensitivity rows to one row per user.

    Produces three aggregate columns:
      - AverageSensitivity  : mean sensitivity score per user
      - MaxSensitivity      : maximum sensitivity score per user
      - SensitiveFileCount  : total number of files for that user

    Parameters
    ----------
    df:
        File-level sensitivity DataFrame with USER_ID_COL and
        *sensitivity_col*.
    sensitivity_col:
        Name of the numeric sensitivity score column.

    Returns
    -------
    pd.DataFrame
        User-level aggregated DataFrame indexed by USER_ID_COL.
    """
    if sensitivity_col not in df.columns:
        raise ValueError(
            f"Sensitivity column '{sensitivity_col}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    log.info("Aggregating file-sensitivity to user level …")

    # Coerce score column to numeric; non-parseable values become NaN.
    df = df.copy()
    df[sensitivity_col] = pd.to_numeric(df[sensitivity_col], errors="coerce")

    agg = (
        df.groupby(USER_ID_COL, sort=False)[sensitivity_col]
        .agg(
            **{
                AGG_AVG_COL:   "mean",
                AGG_MAX_COL:   "max",
                AGG_COUNT_COL: "count",
            }
        )
        .reset_index()
    )

    log.debug(
        "  → Sensitivity aggregated: %d unique users, columns %s",
        len(agg),
        agg.columns.tolist(),
    )
    return agg


# ---------------------------------------------------------------------------
# NaN filling
# ---------------------------------------------------------------------------

def _fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values by dtype:
      - Numeric columns  → 0
      - Object/category  → "Unknown"

    Parameters
    ----------
    df:
        Merged DataFrame (modified via copy).

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Exclude user_id from categorical fill to preserve its integrity.
    categorical_cols = [c for c in categorical_cols if c != USER_ID_COL]

    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].fillna(0)
        log.debug("  Filled %d numeric column(s) with 0.", len(numeric_cols))

    if categorical_cols:
        df[categorical_cols] = df[categorical_cols].fillna("Unknown")
        log.debug(
            "  Filled %d categorical column(s) with 'Unknown'.",
            len(categorical_cols),
        )

    return df


# ---------------------------------------------------------------------------
# Diagnostics / reporting
# ---------------------------------------------------------------------------

def _print_summary(df: pd.DataFrame) -> None:
    """
    Print a concise dataset summary to stdout.

    Covers: shape, missing values, duplicate user count, memory usage,
    feature names, and the first 10 rows.
    """
    sep = "─" * 72

    print(f"\n{sep}")
    print("FINAL DATASET SUMMARY")
    print(sep)

    rows, cols = df.shape
    print(f"  Rows              : {rows:,}")
    print(f"  Columns           : {cols:,}")

    # Missing values across all columns.
    total_missing = int(df.isnull().sum().sum())
    print(f"  Missing values    : {total_missing:,}")

    # Duplicate user check on the output dataset.
    dup_users = int(df.duplicated(subset=[USER_ID_COL]).sum())
    print(f"  Duplicate users   : {dup_users:,}")

    # Memory usage (deep to account for object columns).
    mem_mb = df.memory_usage(deep=True).sum() / 1_048_576
    print(f"  Memory usage      : {mem_mb:.2f} MB")

    print("\n  Feature columns:")
    feature_cols = [c for c in df.columns if c != USER_ID_COL]
    for col in feature_cols:
        print(f"    • {col}")

    print(f"\n  Top 10 rows:\n{df.head(10).to_string(index=False)}")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    behavior_path: Path,
    psychology_path: Path,
    sensitivity_path: Path,
    ldap_path: Path,
    psychometric_path: Path,
    output_path: Path,
    sensitivity_col: str,
    chunksize: Optional[int],
    user_id_aliases: list[str],
) -> pd.DataFrame:
    """
    Execute the full feature-fusion pipeline end-to-end.

    Steps
    -----
    1. Read all source CSVs.
    2. Validate required columns.
    3. Normalise user IDs.
    4. Deduplicate users within each source.
    5. Aggregate file-sensitivity to user level. (COMMENTED OUT)
    6. Left-join all sources onto the behavior base. (MODIFIED)
    7. Fill NaN values.
    8. Save output CSV.
    9. Print summary.
    """
    # Build a normalisation alias map: alias → USER_ID_COL.
    alias_map: dict[str, str] = {a: USER_ID_COL for a in user_id_aliases}

    # ------------------------------------------------------------------
    # 1. Read source files
    # ------------------------------------------------------------------
    log.info("=== STEP 1: Reading source files ===")

    df_behavior    = _safe_read_csv(behavior_path,    chunksize=chunksize)
    df_psychology  = _safe_read_csv(psychology_path,  chunksize=chunksize)
    # df_sensitivity = _safe_read_csv(sensitivity_path, chunksize=chunksize)
    df_ldap        = _safe_read_csv(ldap_path,        chunksize=chunksize)
    df_psychometric = _safe_read_csv(psychometric_path, chunksize=chunksize)

    # ------------------------------------------------------------------
    # 2. Validate columns
    # ------------------------------------------------------------------
    log.info("=== STEP 2: Validating columns ===")

    sources = {
        "behavior":     df_behavior,
        "psychology":   df_psychology,
        "ldap":         df_ldap,
        "psychometric": df_psychometric,
        # "sensitivity":  df_sensitivity,
    }
    for name, df in sources.items():
        _validate_columns(df, name, REQUIRED_COLUMNS[name])

    # ------------------------------------------------------------------
    # 3. Normalise user IDs
    # ------------------------------------------------------------------
    log.info("=== STEP 3: Normalising user IDs ===")

    df_behavior     = _normalize_user_id(df_behavior,     alias_map)
    df_psychology   = _normalize_user_id(df_psychology,   alias_map)
    # df_sensitivity  = _normalize_user_id(df_sensitivity,  alias_map)
    df_ldap         = _normalize_user_id(df_ldap,         alias_map)
    df_psychometric = _normalize_user_id(df_psychometric, alias_map)

    # ------------------------------------------------------------------
    # 4. Deduplicate users within each source
    # ------------------------------------------------------------------
    log.info("=== STEP 4: Removing duplicate users ===")

    df_behavior     = _drop_duplicate_users(df_behavior,     "behavior")
    df_psychology   = _drop_duplicate_users(df_psychology,   "psychology")
    df_ldap         = _drop_duplicate_users(df_ldap,         "ldap")
    df_psychometric = _drop_duplicate_users(df_psychometric, "psychometric")

    # ------------------------------------------------------------------
    # 5. Aggregate file sensitivity → user level
    # ------------------------------------------------------------------
    # log.info("=== STEP 5: Aggregating file sensitivity ===")
    # df_sensitivity_agg = _aggregate_sensitivity(df_sensitivity, sensitivity_col)

    # ------------------------------------------------------------------
    # 6. Merge all sources
    # ------------------------------------------------------------------
    log.info("=== STEP 6: Merging feature tables ===")

    merged = df_behavior.copy()

    merge_sources = [
        ("psychology",   df_psychology),
        ("ldap",         df_ldap),
        ("psychometric", df_psychometric),
        # ("sensitivity",  df_sensitivity_agg),
    ]

    for label, df_right in merge_sources:
        n_before = len(merged)

        overlap = [
            c for c in df_right.columns
            if c != USER_ID_COL and c in merged.columns
        ]
        if overlap:
            log.warning(
                "[%s] Overlapping columns will be suffixed: %s",
                label,
                overlap,
            )

        merged = merged.merge(df_right, on=USER_ID_COL, how="left", suffixes=("", f"_{label}"))
        log.info(
            "  Merged [%s]: %d → %d rows (expected no change on left join).",
            label,
            n_before,
            len(merged),
        )

        if len(merged) != n_before:
            log.error(
                "[%s] Row count changed after left join (%d → %d). "
                "Check for duplicate user_ids in the right table.",
                label,
                n_before,
                len(merged),
            )

    log.info("Post-merge shape: %d rows × %d cols", *merged.shape)

    # ------------------------------------------------------------------
    # 7. Fill NaN values
    # ------------------------------------------------------------------
    log.info("=== STEP 7: Filling missing values ===")
    merged = _fill_missing_values(merged)

    # ------------------------------------------------------------------
    # 8. Save output
    # ------------------------------------------------------------------
    log.info("=== STEP 8: Saving output ===")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    log.info("Saved final dataset → %s", output_path)

    # ------------------------------------------------------------------
    # 9. Print summary
    # ------------------------------------------------------------------
    _print_summary(merged)

    return merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="feature_fusion",
        description="Merge all engineered features into one ML dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input paths
    parser.add_argument(
        "--behavior",
        type=Path,
        default=Path("data/processed/behavior_features.csv"),
        help="Path to behavior_features.csv",
    )
    parser.add_argument(
        "--psychology",
        type=Path,
        default=Path("data/processed/psychology_features.csv"),
        help="Path to psychology_features.csv",
    )
    parser.add_argument(
        "--sensitivity",
        type=Path,
        default=Path("data/processed/file_sensitivity.csv"),
        help="Path to file_sensitivity.csv (filename-level granularity)",
    )
    parser.add_argument(
        "--ldap",
        type=Path,
        default=Path("data/raw/LDAP.csv"),
        help="Path to LDAP.csv",
    )
    parser.add_argument(
        "--psychometric",
        type=Path,
        default=Path("data/raw/psychometric.csv"),
        help="Path to psychometric.csv",
    )

    # Output path
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/final_features.csv"),
        help="Destination path for the fused dataset",
    )

    # Sensitivity score column
    parser.add_argument(
        "--sensitivity-col",
        type=str,
        default="sensitivity_score",
        help="Column name of the numeric sensitivity score in file_sensitivity.csv",
    )

    # User-ID aliases
    parser.add_argument(
        "--user-id-aliases",
        nargs="*",
        default=["userId", "UserID", "user", "username", "emp_id"],
        help="Alternative column names that represent user_id across source files",
    )

    # Chunked reading
    parser.add_argument(
        "--chunksize",
        type=int,
        default=None,
        help="Number of rows per CSV chunk (None = read whole file at once)",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Parse CLI arguments, configure logging, resolve & validate paths,
    then hand off to the fusion pipeline.
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Configure logging first so that all subsequent calls can use it.
    global log
    log = _configure_logging(args.log_level)

    log.info("Feature Fusion Pipeline — starting")
    log.debug("Arguments: %s", vars(args))

    try:
        behavior_path    = _resolve_path(args.behavior)
        psychology_path  = _resolve_path(args.psychology)
        sensitivity_path = _resolve_path(args.sensitivity)
        ldap_path        = _resolve_path(args.ldap)
        psychometric_path = _resolve_path(args.psychometric)
    except FileNotFoundError as exc:
        log.error("Path resolution failed: %s", exc)
        sys.exit(1)

    output_path = Path(args.output).expanduser().resolve()

    try:
        run_pipeline(
            behavior_path=behavior_path,
            psychology_path=psychology_path,
            sensitivity_path=sensitivity_path,
            ldap_path=ldap_path,
            psychometric_path=psychometric_path,
            output_path=output_path,
            sensitivity_col=args.sensitivity_col,
            chunksize=args.chunksize,
            user_id_aliases=args.user_id_aliases,
        )
    except (ValueError, KeyError) as exc:
        log.error("Pipeline failed: %s", exc)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        log.exception("Unexpected error: %s", exc)
        sys.exit(2)

    log.info("Feature Fusion Pipeline — complete.")


if __name__ == "__main__":
    main()