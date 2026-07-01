"""
===============================================================================
PHASE 4
Behavior Feature Engineering

CERT Insider Threat Detection System

Input:
    data/processed/unified_events.csv

Output:
    data/processed/behavior_features.csv

Generates per-user behavioural features for ML and GNN.

Author:
B.Tech Project
===============================================================================
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import re
import sys
import time

from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from tqdm import tqdm


###############################################################################
# CONFIGURATION
###############################################################################

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "unified_events.csv"

DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "behavior_features.csv"

DEFAULT_CHUNK_SIZE = 100000

BUSINESS_START = 8
BUSINESS_END = 18

NIGHT_END = 6


###############################################################################
# OUTPUT FEATURES
###############################################################################

OUTPUT_COLUMNS = [
    "user_id",

    # Counts
    "LoginCount",
    "EmailCount",
    "FileCount",
    "HttpCount",
    "DeviceCount",
    "TotalEvents",

    # Time Behaviour
    "AfterHoursCount",
    "AfterHoursRatio",

    "NightEvents",
    "NightActivityRatio",

    "WeekendEvents",
    "WeekendRatio",

    "BusinessHourEvents",
    "BusinessHourRatio",

    # Login
    "FailedLoginCount",
    "FailedLoginRatio",

    # Devices
    "UniquePCs",
    "DeviceSwitchCount",

    # Email
    "UniqueRecipients",
    "ExternalEmailCount",
    "ExternalEmailRatio",

    # Web
    "UniqueDomainsVisited",

    # File
    "UniqueFilesTouched",
    "FileReadCount",
    "FileWriteCount",
    "FileCopyCount",
    "FileDeleteCount",

    # USB
    "USBInsertions",
    "USBRemovals",

    # Daily Behaviour
    "ActiveDays",
    "IdleDays",
    "AvgEventsPerDay",
    "StdEventsPerDay",

    # Statistical
    "PeakActivityHour",
    "ActivityBurstScore",
    "BehaviorEntropy"
]


###############################################################################
# LOGGER
###############################################################################

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout
)

log = logging.getLogger("behavior")


###############################################################################
# HELPERS
###############################################################################

EMAIL_REGEX = re.compile(r"@(.+)$")


def extract_domain(email):
    if pd.isna(email):
        return None

    m = EMAIL_REGEX.search(str(email).lower())
    if m:
        return m.group(1)

    return None


def shannon_entropy(counter):
    total = sum(counter.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for c in counter.values():
        p = c / total
        entropy -= p * math.log2(p)

    return entropy


def count_rows(path: Path):
    rows = 0
    if not path.exists():
        return 0
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            rows += block.count(b"\n")
    return max(rows - 1, 0)


def summary_report(df: pd.DataFrame):
    """Generates a brief structural report of the extracted behavioral features."""
    log.info("Behavioral Feature Dataframe summary matrix built.")
    log.info("Shape: %s | Unique Users: %d", df.shape, df["user_id"].nunique())


###############################################################################
# USER STATISTICS
###############################################################################

class UserStatistics:

    def __init__(self):
        # Event Counts
        self.login = defaultdict(int)
        self.email = defaultdict(int)
        self.file = defaultdict(int)
        self.http = defaultdict(int)
        self.device = defaultdict(int)
        self.total = defaultdict(int)

        # Time
        self.after_hours = defaultdict(int)
        self.weekend = defaultdict(int)
        self.night = defaultdict(int)
        self.business = defaultdict(int)

        # Failed login
        self.failed_login = defaultdict(int)

        # USB
        self.usb_insert = defaultdict(int)
        self.usb_remove = defaultdict(int)

        # File Ops
        self.file_read = defaultdict(int)
        self.file_write = defaultdict(int)
        self.file_copy = defaultdict(int)
        self.file_delete = defaultdict(int)

        # Sets
        self.pcs = defaultdict(set)
        self.files = defaultdict(set)
        self.domains = defaultdict(set)
        self.recipients = defaultdict(set)
        self.days = defaultdict(set)

        # Counters
        self.hour_counter = defaultdict(Counter)
        self.daily_counter = defaultdict(Counter)
        self.device_sequence = defaultdict(list)
        self.external_email = defaultdict(int)

    def update(self, chunk: pd.DataFrame):
        # Parse timestamps
        chunk["timestamp"] = pd.to_datetime(
            chunk["timestamp"],
            errors="coerce"
        )
        chunk = chunk.dropna(subset=["timestamp"]).copy()

        chunk["hour"] = chunk["timestamp"].dt.hour
        chunk["weekday"] = chunk["timestamp"].dt.weekday
        chunk["date"] = chunk["timestamp"].dt.date

        for row in chunk.itertuples(index=False):
            user = str(row.user_id)
            event = str(row.event_type).lower()
            self.total[user] += 1

            ##########################################################
            # Event Counts
            ##########################################################
            if event == "logon":
                self.login[user] += 1
            elif event == "email":
                self.email[user] += 1
            elif event == "file":
                self.file[user] += 1
            elif event == "http":
                self.http[user] += 1
            elif event == "device":
                self.device[user] += 1

            ##########################################################
            # Time Features
            ##########################################################
            hour = row.hour
            weekday = row.weekday

            if hour < BUSINESS_START or hour >= BUSINESS_END:
                self.after_hours[user] += 1
            else:
                self.business[user] += 1

            if hour < NIGHT_END:
                self.night[user] += 1

            if weekday >= 5:
                self.weekend[user] += 1

            ##########################################################
            # Daily Statistics
            ##########################################################
            self.days[user].add(row.date)
            self.daily_counter[user][row.date] += 1
            self.hour_counter[user][hour] += 1

            ##########################################################
            # Resource / PC
            ##########################################################
            if hasattr(row, "resource"):
                if pd.notna(row.resource) and str(row.resource).strip() != "":
                    pc = str(row.resource)
                    self.pcs[user].add(pc)
                    self.device_sequence[user].append(pc)

            ##########################################################
            # Filename
            ##########################################################
            if hasattr(row, "filename"):
                if pd.notna(row.filename) and str(row.filename).strip() != "":
                    self.files[user].add(str(row.filename))

            ##########################################################
            # Email
            ##########################################################
            if hasattr(row, "to"):
                if pd.notna(row.to) and str(row.to).strip() != "":
                    recipients = str(row.to).split(";")
                    for r in recipients:
                        r = r.strip()
                        if not r:
                            continue
                        self.recipients[user].add(r)
                        domain = extract_domain(r)
                        if domain:
                            self.domains[user].add(domain)
                            if not domain.endswith("dtaa.com"):
                                self.external_email[user] += 1

            ##########################################################
            # HTTP Domains
            ##########################################################
            if hasattr(row, "url"):
                if pd.notna(row.url) and str(row.url).strip() != "":
                    url = str(row.url).lower()
                    url = url.replace("http://", "").replace("https://", "")
                    domain = url.split("/")[0]
                    if domain:
                        self.domains[user].add(domain)

            ##########################################################
            # File Operations
            ##########################################################
            if hasattr(row, "activity"):
                act = str(row.activity).lower()
                if "read" in act:
                    self.file_read[user] += 1
                elif "write" in act:
                    self.file_write[user] += 1
                elif "copy" in act:
                    self.file_copy[user] += 1
                elif "delete" in act:
                    self.file_delete[user] += 1

            ##########################################################
            # Device Events
            ##########################################################
            if hasattr(row, "activity"):
                act = str(row.activity).lower()
                if "connect" in act:
                    self.usb_insert[user] += 1
                elif "disconnect" in act:
                    self.usb_remove[user] += 1

            ##########################################################
            # Failed Login
            ##########################################################
            if hasattr(row, "status"):
                status = str(row.status).lower()
                if status in ["fail", "failed", "failure"]:
                    self.failed_login[user] += 1


###############################################################################
# WRITE OUTPUT
###############################################################################

def write_output(stats: UserStatistics, output_file: Path):
    log.info("=" * 70)
    log.info("Generating Behavioral Features")
    log.info("=" * 70)

    rows = []
    TOTAL_DAYS = 495      # CERT r4.2 duration
    users = sorted(stats.total.keys())

    for user in users:
        total = stats.total[user]
        if total == 0:
            continue

        ############################################################
        # Basic Counts
        ############################################################
        login = stats.login[user]
        email = stats.email[user]
        file = stats.file[user]
        http = stats.http[user]
        device = stats.device[user]

        ############################################################
        # Time Features
        ############################################################
        after = stats.after_hours[user]
        weekend = stats.weekend[user]
        night = stats.night[user]
        business = stats.business[user]

        ############################################################
        # Failed Login
        ############################################################
        failed = stats.failed_login[user]

        ############################################################
        # Diversity
        ############################################################
        unique_pcs = len(stats.pcs[user])
        unique_files = len(stats.files[user])
        unique_domains = len(stats.domains[user])
        unique_recipients = len(stats.recipients[user])

        ############################################################
        # Days
        ############################################################
        active_days = len(stats.days[user])
        idle_days = TOTAL_DAYS - active_days

        ############################################################
        # Events Per Day
        ############################################################
        daily_events = list(stats.daily_counter[user].values())
        if len(daily_events):
            avg_events = np.mean(daily_events)
            std_events = np.std(daily_events)
            burst = np.max(daily_events)
        else:
            avg_events = 0
            std_events = 0
            burst = 0

        ############################################################
        # Peak Hour
        ############################################################
        if len(stats.hour_counter[user]):
            peak_hour = max(
                stats.hour_counter[user],
                key=stats.hour_counter[user].get
            )
        else:
            peak_hour = 0

        ############################################################
        # Device Switching
        ############################################################
        switches = 0
        seq = stats.device_sequence[user]
        for i in range(1, len(seq)):
            if seq[i] != seq[i - 1]:
                switches += 1

        ############################################################
        # Entropy
        ############################################################
        entropy = shannon_entropy(stats.hour_counter[user])

        ############################################################
        # Ratios
        ############################################################
        after_ratio = after / total
        weekend_ratio = weekend / total
        night_ratio = night / total
        business_ratio = business / total
        failed_ratio = failed / login if login else 0
        external_ratio = stats.external_email[user] / email if email else 0

        ############################################################
        # Final Row
        ############################################################
        rows.append({
            "user_id": user,
            "LoginCount": login,
            "EmailCount": email,
            "FileCount": file,
            "HttpCount": http,
            "DeviceCount": device,
            "TotalEvents": total,
            "AfterHoursCount": after,
            "AfterHoursRatio": round(after_ratio, 4),
            "NightEvents": night,
            "NightActivityRatio": round(night_ratio, 4),
            "WeekendEvents": weekend,
            "WeekendRatio": round(weekend_ratio, 4),
            "BusinessHourEvents": business,
            "BusinessHourRatio": round(business_ratio, 4),
            "FailedLoginCount": failed,
            "FailedLoginRatio": round(failed_ratio, 4),
            "UniquePCs": unique_pcs,
            "DeviceSwitchCount": switches,
            "UniqueRecipients": unique_recipients,
            "ExternalEmailCount": stats.external_email[user],
            "ExternalEmailRatio": round(external_ratio, 4),
            "UniqueDomainsVisited": unique_domains,
            "UniqueFilesTouched": unique_files,
            "FileReadCount": stats.file_read[user],
            "FileWriteCount": stats.file_write[user],
            "FileCopyCount": stats.file_copy[user],
            "FileDeleteCount": stats.file_delete[user],
            "USBInsertions": stats.usb_insert[user],
            "USBRemovals": stats.usb_remove[user],
            "ActiveDays": active_days,
            "IdleDays": idle_days,
            "AvgEventsPerDay": round(avg_events, 2),
            "StdEventsPerDay": round(std_events, 2),
            "PeakActivityHour": peak_hour,
            "ActivityBurstScore": burst,
            "BehaviorEntropy": round(entropy, 4)
        })

    ###################################################################
    # Save CSV
    ###################################################################
    if not rows:
        log.warning("No behavior metrics calculated. Output dataframe is empty.")
        return

    df = pd.DataFrame(rows)
    df = df[OUTPUT_COLUMNS]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    
    log.info("Saved %d users to %s", len(df), output_file)
    summary_report(df)


###############################################################################
# MAIN PROCESSING
###############################################################################

def process_file(input_file: Path, output_file: Path, chunk_size: int):
    start = time.time()
    total_rows = count_rows(input_file)

    log.info("=" * 70)
    log.info("Phase 4 : Behavior Feature Engineering")
    log.info("=" * 70)
    log.info("Input      : %s", input_file)
    log.info("Output     : %s", output_file)
    log.info("Chunk Size : %d", chunk_size)
    log.info("Rows       : %s", f"{total_rows:,}")

    stats = UserStatistics()
    processed = 0

    if total_rows == 0:
        log.error("Input file empty or does not exist.")
        return

    reader = pd.read_csv(
        input_file,
        chunksize=chunk_size,
        low_memory=False,
        dtype=str
    )

    with tqdm(
        total=total_rows,
        desc="Processing",
        unit="rows",
        dynamic_ncols=True
    ) as pbar:
        for chunk in reader:
            # Required columns
            required = ["user_id", "timestamp", "event_type"]
            for col in required:
                if col not in chunk.columns:
                    raise ValueError(f"Missing required column: {col}")

            # Optional columns
            optional_columns = [
                "activity",
                "status",
                "to",
                "url",
                "filename",
                "resource"
            ]

            # Add missing optional columns cleanly
            for col in optional_columns:
                if col not in chunk.columns:
                    chunk[col] = ""

            stats.update(chunk)
            processed += len(chunk)
            pbar.update(len(chunk))

            del chunk
            gc.collect()

    elapsed = time.time() - start
    log.info("Rows Processed : %s", f"{processed:,}")
    log.info("Elapsed        : %.2f sec", elapsed)
    log.info("Rows/sec       : %.0f", processed / elapsed if elapsed > 0 else 0)

    write_output(stats, output_file)

    log.info("=" * 70)
    log.info("Pipeline Completed Successfully")
    log.info("=" * 70)
    log.info("Users Processed : %d", len(stats.total))


###############################################################################
# ARGUMENTS
###############################################################################

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        type=Path
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        type=Path
    )
    parser.add_argument(
        "--chunk-size",
        default=DEFAULT_CHUNK_SIZE,
        type=int
    )
    return parser.parse_args()


###############################################################################
# MAIN
###############################################################################

def main():
    args = parse_args()
    process_file(
        args.input,
        args.output,
        args.chunk_size
    )


if __name__ == "__main__":
    main()