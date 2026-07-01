"""
unify_logs.py
=============
Streaming ETL pipeline for the CERT Insider Threat Dataset.

Reads logon, email, file, device, and http CSVs one at a time,
processes each in fixed-size chunks, and appends rows directly to the
output CSV — never holding more than one chunk in memory at once.

Output schema
-------------
  user_id      str   – acting user
  timestamp    str   – ISO-8601 datetime (NaT-as-empty if unparseable)
  event_type   str   – logon | email | file | device | http
  resource     str   – primary object (PC, filename, URL, recipient …)
  target_user  str   – second identity involved (email sender, etc.)

Memory contract
---------------
* Each source is read in chunks; chunks are discarded after writing.
* gc.collect() is called after every chunk.
* Peak RSS should stay well under 1 GB even on the full CERT dataset.

Usage
-----
  python unify_logs.py [--input-dir DIR] [--output PATH]
                       [--chunk-size N] [--log-level LEVEL]

Defaults
--------
  --input-dir   .
  --output      data/processed/unified_events.csv
  --chunk-size  100000
  --log-level   INFO
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
from pathlib import Path
from typing import Iterator

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw"

DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "unified_events.csv"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_COLUMNS = ["user_id", "timestamp", "event_type", "resource", "target_user"]

# Tried in order; first hit wins.  NaT is set for anything that survives all formats.
_TIMESTAMP_FORMATS = [
    "%m/%d/%Y %H:%M:%S",   # 01/05/2010 07:12:00  — logon, email, device, http
    "%Y-%m-%d %H:%M:%S",   # 2010-01-06 08:45:33
    "%Y-%m-%dT%H:%M:%S",   # 2010-01-06T10:30:00
    "%b %d %Y %H:%M:%S",   # Jan 5 2010 07:45:00  — file
    "%Y-%m-%d",            # date-only fallback
]

# Sentinel values read_csv treats as NaN for all string columns.
_NA_VALUES = {"", "NA", "N/A", "nan", "NaN", "null", "NULL", "None", "none"}


# ---------------------------------------------------------------------------
# Timestamp normalisation  (operates on a single Series, never the full file)
# ---------------------------------------------------------------------------

def _normalise_timestamps(series: pd.Series) -> pd.Series:
    """
    Parse a mixed-format datetime Series without loading more than `series`
    into memory.

    Strategy:
      1. Walk candidate formats; assign successes, keep a "still-unparsed" mask.
      2. One pandas infer pass for stragglers.
      3. Anything still NaT is left as NaT (stored as empty string in CSV).

    Returns a Series of dtype datetime64[ns].
    """
    n = len(series)
    result   = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    unparsed = series.notna()  # boolean mask over the full chunk index

    for fmt in _TIMESTAMP_FORMATS:
        if not unparsed.any():
            break

        subset  = series[unparsed]                                   # only unsolved rows
        parsed  = pd.to_datetime(subset, format=fmt, errors="coerce")
        hit     = parsed.notna().reindex(series.index, fill_value=False)  # realign to chunk index
        result[hit]  = parsed[parsed.notna()].values
        unparsed     = unparsed & ~hit                               # narrow the remaining set

    # Last-chance inference pass
    if unparsed.any():
        subset = series[unparsed]
        parsed = pd.to_datetime(subset, errors="coerce", infer_datetime_format=True)
        hit    = parsed.notna().reindex(series.index, fill_value=False)
        result[hit]  = parsed[parsed.notna()].values
        unparsed     = unparsed & ~hit

    still_nat = unparsed.sum()
    if still_nat:
        log.warning("  %d value(s) could not be parsed → NaT", still_nat)

    return result


# ---------------------------------------------------------------------------
# Per-source chunk transformers
# Each receives ONE raw chunk DataFrame, returns ONE canonical DataFrame.
# They must not retain any reference to the input after returning.
# ---------------------------------------------------------------------------

def _transform_logon(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Columns used: date → timestamp, user → user_id, pc → resource
    event_type  = 'logon'
    target_user = <empty>
    """
    out = pd.DataFrame(index=chunk.index)
    out["user_id"]     = chunk.get("user")
    out["timestamp"]   = _normalise_timestamps(chunk.get("date", pd.Series(dtype=str, index=chunk.index)))
    out["event_type"]  = "logon"
    out["resource"]    = chunk.get("pc")
    out["target_user"] = pd.NA
    return out[OUTPUT_COLUMNS]


def _transform_email(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Columns used: date, user, to (→ resource), from (→ target_user)
    event_type = 'email'

    'from' captures the sender address — distinct from 'user' when mail is
    sent on behalf of another account (shared mailbox, forwarding rule, etc.).
    """
    out = pd.DataFrame(index=chunk.index)
    out["user_id"]     = chunk.get("user")
    out["timestamp"]   = _normalise_timestamps(chunk.get("date", pd.Series(dtype=str, index=chunk.index)))
    out["event_type"]  = "email"
    out["resource"]    = chunk.get("to")
    out["target_user"] = chunk.get("from")
    return out[OUTPUT_COLUMNS]


def _transform_file(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Columns used: date, user, filename (→ resource)
    event_type  = 'file'
    target_user = <empty>
    """
    out = pd.DataFrame(index=chunk.index)
    out["user_id"]     = chunk.get("user")
    out["timestamp"]   = _normalise_timestamps(chunk.get("date", pd.Series(dtype=str, index=chunk.index)))
    out["event_type"]  = "file"
    out["resource"]    = chunk.get("filename")
    out["target_user"] = pd.NA
    return out[OUTPUT_COLUMNS]


def _transform_device(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Columns used: date, user, pc (→ resource)
    event_type  = 'device'
    target_user = <empty>
    """
    out = pd.DataFrame(index=chunk.index)
    out["user_id"]     = chunk.get("user")
    out["timestamp"]   = _normalise_timestamps(chunk.get("date", pd.Series(dtype=str, index=chunk.index)))
    out["event_type"]  = "device"
    out["resource"]    = chunk.get("pc")
    out["target_user"] = pd.NA
    return out[OUTPUT_COLUMNS]


def _transform_http(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Columns used: date, user, url (→ resource)
    event_type  = 'http'
    target_user = <empty>
    """
    out = pd.DataFrame(index=chunk.index)
    out["user_id"]     = chunk.get("user")
    out["timestamp"]   = _normalise_timestamps(chunk.get("date", pd.Series(dtype=str, index=chunk.index)))
    out["event_type"]  = "http"
    out["resource"]    = chunk.get("url")
    out["target_user"] = pd.NA
    return out[OUTPUT_COLUMNS]


# ---------------------------------------------------------------------------
# Source registry
# Each entry: (filename_stem, transformer_fn, use_chunked_reader)
#
# Large files (email, http) use chunked reading.
# Smaller files (logon, file, device) are read in a single pass but still
# processed as a single "chunk" through the same pipeline so the code path
# is identical — makes adding chunk support to any source trivial later.
# ---------------------------------------------------------------------------

_SOURCE_REGISTRY: list[tuple[str, callable, bool]] = [
    ("logon",  _transform_logon,  False),
    ("email",  _transform_email,  True),
    ("file",   _transform_file,   False),
    ("device", _transform_device, False),
    ("http",   _transform_http,   True),
]


# ---------------------------------------------------------------------------
# Chunk reader — yields raw DataFrames one at a time
# ---------------------------------------------------------------------------

def _iter_chunks(
    path: Path,
    chunk_size: int,
    chunked: bool,
) -> Iterator[pd.DataFrame]:
    """
    Yield raw DataFrames.

    * chunked=True  → TextFileReader with chunksize; each yield is ≤ chunk_size rows.
    * chunked=False → single read, yielded once as a one-chunk iterator.

    In both cases read_csv is called with dtype=str and the full NA value set
    so every column arrives as strings with explicit NaN — no silent type
    coercion before the transformer runs.
    """
    read_kwargs = dict(
        dtype=str,
        keep_default_na=False,
        na_values=list(_NA_VALUES),
        low_memory=False,
    )

    if chunked:
        reader = pd.read_csv(path, chunksize=chunk_size, **read_kwargs)
        try:
            for chunk in reader:
                yield chunk
        finally:
            # Ensure the underlying file handle is closed on any exit path.
            reader.close()
    else:
        df = pd.read_csv(path, **read_kwargs)
        yield df
        del df


# ---------------------------------------------------------------------------
# Chunk post-processing — applied after transformation, before writing
# ---------------------------------------------------------------------------

def _clean_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise a transformed canonical chunk in-place.

    1. Strip whitespace from all string columns.
    2. Replace blank strings / sentinel values with pd.NA.
    3. Drop exact-duplicate rows within the chunk.
       (Cross-chunk deduplication would require memory; not done here.)
    4. Convert timestamp to ISO-8601 string for CSV storage.
    """
    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].str.strip()

    df.replace(
        to_replace={"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA, "null": pd.NA},
        inplace=True,
    )

    before = len(df)
    df.drop_duplicates(inplace=True)
    dupes = before - len(df)
    if dupes:
        log.debug("  Dropped %d intra-chunk duplicate(s).", dupes)

    # Serialise timestamps as ISO strings; NaT becomes empty string in CSV.
    if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    return df


# ---------------------------------------------------------------------------
# Writer — append-mode, header written once via a flag file
# ---------------------------------------------------------------------------

class _CSVWriter:
    """
    Thin wrapper that keeps track of whether the header has been written.
    Writes each chunk with mode='a' after the first.

    Using a flag file rather than an in-memory bool makes the class safe
    if the process is restarted and the output already exists — it will
    detect the existing header and skip writing it again.
    """

    def __init__(self, output_path: Path) -> None:
        self.path         = output_path
        self._header_done = output_path.exists() and output_path.stat().st_size > 0

    def write(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        df.to_csv(
            self.path,
            mode="a",
            header=not self._header_done,
            index=False,
        )
        if not self._header_done:
            log.info("Header written to %s", self.path)
            self._header_done = True


# ---------------------------------------------------------------------------
# Row-count helper (fast, no full load)
# ---------------------------------------------------------------------------

def _count_rows(path: Path) -> int:
    """Return approximate row count by counting newlines (fast, no pandas)."""
    count = 0
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):   # 1 MB blocks
            count += block.count(b"\n")
    return max(0, count - 1)  # subtract header line


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(input_dir: Path, output_path: Path, chunk_size: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = _CSVWriter(output_path)

    total_rows_written = 0

    for source_name, transformer, chunked in _SOURCE_REGISTRY:
        csv_path = input_dir / f"{source_name}.csv"

        if not csv_path.exists():
            log.error("Missing source file: %s — aborting.", csv_path)
            sys.exit(1)

        approx_rows  = _count_rows(csv_path)
        approx_chunks = max(1, -(-approx_rows // chunk_size))   # ceiling division

        log.info(
            "Processing %-8s  (~%d rows, ~%d chunk(s), chunked=%s)",
            f"{source_name}.csv",
            approx_rows,
            approx_chunks,
            chunked,
        )

        source_rows = 0
        chunk_index = 0

        with tqdm(
            total=approx_rows,
            desc=f"  {source_name:<8}",
            unit="row",
            unit_scale=True,
            dynamic_ncols=True,
            leave=True,
        ) as bar:
            for raw_chunk in _iter_chunks(csv_path, chunk_size, chunked):
                chunk_index += 1
                chunk_len    = len(raw_chunk)

                log.debug(
                    "  %s chunk %d: %d raw rows", source_name, chunk_index, chunk_len
                )

                # --- Transform ---
                canonical = transformer(raw_chunk)
                del raw_chunk        # drop source columns immediately
                gc.collect()

                # --- Clean ---
                canonical = _clean_chunk(canonical)

                # --- Write ---
                writer.write(canonical)

                rows_out      = len(canonical)
                source_rows  += rows_out
                total_rows_written += rows_out

                del canonical        # explicit drop before next iteration
                gc.collect()

                bar.update(chunk_len)

        log.info("  → %d rows written from %s", source_rows, source_name)

    log.info("Done. Total rows written: %d → %s", total_rows_written, output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing CERT CSV files",
    )

    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Output unified CSV",
    )

    p.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Rows per chunk",
    )

    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    return p


def main():

    args = _build_parser().parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    input_dir = args.input_dir.resolve()
    output_file = args.output.resolve()

    log.info("=" * 60)
    log.info("CERT Unify Logs")
    log.info("=" * 60)
    log.info("Project Root : %s", PROJECT_ROOT)
    log.info("Input Dir    : %s", input_dir)
    log.info("Output File  : %s", output_file)
    log.info("Chunk Size   : %d", args.chunk_size)

    run(
        input_dir=input_dir,
        output_path=output_file,
        chunk_size=args.chunk_size,
    )

    log.info("=" * 60)
    log.info("Completed Successfully")
    log.info("=" * 60)

if __name__ == "__main__":
    main()