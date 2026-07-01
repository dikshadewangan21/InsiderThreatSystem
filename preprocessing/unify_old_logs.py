"""
unify_logs.py
-------------
Merges logon, email, file, device, and http CSVs into a single unified
events dataframe with a canonical 5-column schema.

Columns produced:
    user_id      – identity of the acting user
    timestamp    – UTC-normalised datetime (NaT if unparseable)
    event_type   – one of: logon | email | file | device | http
    resource     – the primary object acted upon (PC, filename, URL, recipient …)
    target_user  – second identity involved (email recipient, cc, etc.)

Usage:
    python unify_logs.py --input-dir <dir> --output <path>

Defaults:
    --input-dir   ./  (current directory)
    --output      data/processed/unified_events.csv
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw"

DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "unified_events.csv"


# ---------------------------------------------------------------------------
# Timestamp normalisation
# ---------------------------------------------------------------------------
_TIMESTAMP_FORMATS = [
    "%m/%d/%Y %H:%M:%S",   # 01/05/2010 07:12:00   (logon, email, device, http)
    "%Y-%m-%d %H:%M:%S",   # 2010-01-06 08:45:33
    "%Y-%m-%dT%H:%M:%S",   # 2010-01-06T10:30:00
    "%b %d %Y %H:%M:%S",   # Jan 5 2010 07:45:00   (file)
    "%Y-%m-%d",            # date-only fallback
]


def _parse_timestamp(series: pd.Series) -> pd.Series:
    """
    Try each known format in order; fall back to pandas' infer_datetime_format
    for anything that still hasn't parsed.  Rows that remain unparseable become
    NaT and are logged as warnings.
    """
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    remaining_mask = series.notna()

    for fmt in _TIMESTAMP_FORMATS:
        if not remaining_mask.any():
            break
        subset = series[remaining_mask]
        parsed = pd.to_datetime(subset, format=fmt, errors="coerce")
        # ok is index-aligned to the subset; reindex to full index for masking
        ok_full = parsed.notna().reindex(series.index, fill_value=False)
        result[ok_full] = parsed[parsed.notna()].values
        remaining_mask = remaining_mask & ~ok_full

    # Last-chance: let pandas infer
    if remaining_mask.any():
        subset = series[remaining_mask]
        parsed = pd.to_datetime(subset, errors="coerce", infer_datetime_format=True)
        ok_full = parsed.notna().reindex(series.index, fill_value=False)
        result[ok_full] = parsed[parsed.notna()].values
        remaining_mask = remaining_mask & ~ok_full

    unparseable_count = remaining_mask.sum()
    if unparseable_count:
        log.warning(
            "%d timestamp(s) could not be parsed and were set to NaT.", unparseable_count
        )

    return result


# ---------------------------------------------------------------------------
# Per-source loaders  →  each returns a DataFrame with the canonical 5 cols
# ---------------------------------------------------------------------------

def _read_raw(path: Path, source_name: str) -> pd.DataFrame:
    """Read a CSV and attach source metadata for diagnostics."""
    if not path.exists():
        raise FileNotFoundError(f"Expected file not found: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=["", "NA", "N/A", "nan", "NaN", "null"])
    log.info("%-8s – loaded %d rows, columns: %s", source_name, len(df), list(df.columns))
    return df


def _canonical_columns() -> list[str]:
    return ["user_id", "timestamp", "event_type", "resource", "target_user"]


def _load_logon(path: Path) -> pd.DataFrame:
    """
    Expected columns: id, date, user, pc, activity
    resource    = pc
    target_user = n/a (logon events are self-referential)
    """
    df = _read_raw(path, "logon")
    out = pd.DataFrame(index=df.index)
    out["user_id"]     = df.get("user")
    out["timestamp"]   = _parse_timestamp(df.get("date", pd.Series(dtype=str)))
    out["event_type"]  = "logon"
    out["resource"]    = df.get("pc")
    out["target_user"] = pd.NA
    return out


def _load_email(path: Path) -> pd.DataFrame:
    """
    Expected columns: id, date, user, pc, to, cc, bcc, from, size, attachments, content
    resource    = 'to' field (primary recipient)
    target_user = 'from' field (sender address, useful when != user)
    """
    df = _read_raw(path, "email")
    out = pd.DataFrame(index=df.index)
    out["user_id"]     = df.get("user")
    out["timestamp"]   = _parse_timestamp(df.get("date", pd.Series(dtype=str)))
    out["event_type"]  = "email"
    out["resource"]    = df.get("to")
    out["target_user"] = df.get("from")
    return out


def _load_file(path: Path) -> pd.DataFrame:
    """
    Expected columns: id, date, user, pc, filename, content
    resource    = filename
    target_user = n/a
    """
    df = _read_raw(path, "file")
    out = pd.DataFrame(index=df.index)
    out["user_id"]     = df.get("user")
    out["timestamp"]   = _parse_timestamp(df.get("date", pd.Series(dtype=str)))
    out["event_type"]  = "file"
    out["resource"]    = df.get("filename")
    out["target_user"] = pd.NA
    return out


def _load_device(path: Path) -> pd.DataFrame:
    """
    Expected columns: id, date, user, pc, activity
    resource    = pc
    target_user = n/a
    """
    df = _read_raw(path, "device")
    out = pd.DataFrame(index=df.index)
    out["user_id"]     = df.get("user")
    out["timestamp"]   = _parse_timestamp(df.get("date", pd.Series(dtype=str)))
    out["event_type"]  = "device"
    out["resource"]    = df.get("pc")
    out["target_user"] = pd.NA
    return out


def _load_http(path: Path) -> pd.DataFrame:
    """
    Expected columns: id, date, user, pc, url, content
    resource    = url
    target_user = n/a
    """
    df = _read_raw(path, "http")
    out = pd.DataFrame(index=df.index)
    out["user_id"]     = df.get("user")
    out["timestamp"]   = _parse_timestamp(df.get("date", pd.Series(dtype=str)))
    out["event_type"]  = "http"
    out["resource"]    = df.get("url")
    out["target_user"] = pd.NA
    return out


# ---------------------------------------------------------------------------
# Source registry – extend here to add new log types without touching logic
# ---------------------------------------------------------------------------
_SOURCE_LOADERS: dict[str, callable] = {
    "logon":  _load_logon,
    "email":  _load_email,
    "file":   _load_file,
    "device": _load_device,
    "http":   _load_http,
}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_unified_events(input_dir: Path) -> pd.DataFrame:
    """
    Load every registered source, concatenate, and apply post-processing:
      1. Deduplicate exact rows (all 5 columns identical).
      2. Normalise empty strings → NaN across text columns.
      3. Sort by timestamp ascending (NaT rows go to end).
    """
    frames: list[pd.DataFrame] = []

    for source_name, loader in _SOURCE_LOADERS.items():
        csv_path = input_dir / f"{source_name}.csv"
        try:
            df = loader(csv_path)
            frames.append(df)
        except FileNotFoundError as exc:
            log.error("%s", exc)
            sys.exit(1)
        except Exception as exc:                      # noqa: BLE001
            log.error("Failed to load %s: %s", source_name, exc)
            sys.exit(1)

    unified = pd.concat(frames, ignore_index=True)

    # ── Normalise ──────────────────────────────────────────────────────────
    # Strip whitespace and coerce blank strings to NaN for all object columns
    str_cols = unified.select_dtypes(include=["object", "str"]).columns
    unified[str_cols] = (
        unified[str_cols]
        .apply(lambda col: col.str.strip())
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    )

    # ── Deduplicate ────────────────────────────────────────────────────────
    before = len(unified)
    unified = unified.drop_duplicates(subset=_canonical_columns())
    after  = len(unified)
    if before != after:
        log.info("Dropped %d duplicate row(s).", before - after)

    # ── Sort ───────────────────────────────────────────────────────────────
    unified = unified.sort_values("timestamp", na_position="last").reset_index(drop=True)

    # ── Summary ────────────────────────────────────────────────────────────
    log.info(
        "Unified dataframe: %d rows | NaT timestamps: %d | null user_id: %d",
        len(unified),
        unified["timestamp"].isna().sum(),
        unified["user_id"].isna().sum(),
    )

    return unified


def save(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info("Saved → %s  (%d rows)", output_path, len(df))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Folder containing CERT csv files",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output unified csv",
    )

    return parser.parse_args()


def main():

    args = _parse_args()

    input_dir = args.input_dir.resolve()
    output_path = args.output.resolve()

    log.info("=" * 60)
    log.info("CERT Unify Logs")
    log.info("=" * 60)

    log.info("Project Root : %s", PROJECT_ROOT)
    log.info("Input Dir    : %s", input_dir)
    log.info("Output File  : %s", output_path)

    if not input_dir.exists():
        log.error("Input directory does not exist.")
        log.error("%s", input_dir)
        sys.exit(1)

    required = [
        "logon.csv",
        "email.csv",
        "file.csv",
        "device.csv",
        "http.csv",
    ]

    missing = []

    for f in required:
        if not (input_dir / f).exists():
            missing.append(f)

    if missing:
        log.error("Missing files:")

        for f in missing:
            log.error("  %s", f)

        sys.exit(1)

    unified = build_unified_events(input_dir)

    save(unified, output_path)

    log.info("=" * 60)
    log.info("Completed Successfully")
    log.info("=" * 60)

if __name__ == "__main__":
    main()