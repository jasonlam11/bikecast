"""Normalize both Bluebikes CSV schemas into one and apply documented cleaning rules.

Legacy schema (through 202303): tripduration, starttime, "start station id", usertype, ...
New schema (from 202304): ride_id, rideable_type, started_at, start_station_id, member_casual, ...
"""

import logging
import re
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

UNIFIED_COLUMNS = [
    "ride_id",
    "rideable_type",
    "started_at",
    "ended_at",
    "start_station_id",
    "start_station_name",
    "start_lat",
    "start_lng",
    "end_station_id",
    "end_station_name",
    "end_lat",
    "end_lng",
    "member_type",
]

NEW_RENAME = {"member_casual": "member_type"}

LEGACY_RENAME = {
    "starttime": "started_at",
    "stoptime": "ended_at",
    "start station id": "start_station_id",
    "start station name": "start_station_name",
    "start station latitude": "start_lat",
    "start station longitude": "start_lng",
    "end station id": "end_station_id",
    "end station name": "end_station_name",
    "end station latitude": "end_lat",
    "end station longitude": "end_lng",
    "usertype": "member_type",
}
LEGACY_MEMBER_MAP = {"Subscriber": "member", "Customer": "casual"}

MIN_DURATION_S = 60
MAX_DURATION_S = 24 * 3600
# Test, depot, and staff stations, matched on start and end. IDs catch renames the names miss:
# X32999 was "18 Dorrance Warehouse", later renamed "440 Rutherford Ave Depot".
TEST_STATION_PATTERN = re.compile(
    r"\btest\b|warehouse|maintenance|\bbcu\b|mobile temporary|\bdepot\b", re.IGNORECASE
)
TEST_STATION_IDS = {"X32999"}


def detect_schema(columns: list[str]) -> str:
    cols = {c.strip().lower() for c in columns}
    if {"ride_id", "started_at", "start_station_id"} <= cols:
        return "new"
    if {"tripduration", "starttime", "start station id"} <= cols:
        return "legacy"
    raise ValueError(f"Unrecognized trip schema: {sorted(cols)}")


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Map either schema to UNIFIED_COLUMNS with consistent dtypes. No rows are dropped here."""
    df = raw.rename(columns=lambda c: c.strip().lower())
    schema = detect_schema(list(df.columns))
    if schema == "new":
        df = df.rename(columns=NEW_RENAME)
    else:
        df = df.rename(columns=LEGACY_RENAME)
        df["member_type"] = df["member_type"].map(LEGACY_MEMBER_MAP)
        # Legacy files have no ride ID or bike type.
        df["ride_id"] = pd.NA
        df["rideable_type"] = pd.NA

    df = df[UNIFIED_COLUMNS].copy()
    for col in ["started_at", "ended_at"]:
        df[col] = pd.to_datetime(df[col], format="ISO8601", errors="coerce").astype(
            "datetime64[ms]"
        )
    for col in ["start_lat", "start_lng", "end_lat", "end_lng"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    text_cols = [c for c in UNIFIED_COLUMNS if c not in {"started_at", "ended_at"}]
    text_cols = [c for c in text_cols if not c.endswith(("_lat", "_lng"))]
    for col in text_cols:
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA)
    return df


def _is_test_station(ids: pd.Series, names: pd.Series) -> pd.Series:
    return ids.isin(TEST_STATION_IDS) | names.fillna("").str.contains(TEST_STATION_PATTERN)


def apply_rules(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply cleaning rules in order. Returns kept rows and rows removed by each rule."""
    duration = (df["ended_at"] - df["started_at"]).dt.total_seconds()
    dup_key = ["ride_id"] if df["ride_id"].notna().all() else UNIFIED_COLUMNS
    rules = {
        "duplicate": lambda d: d.duplicated(subset=dup_key),
        "unparseable_time": lambda d: d["started_at"].isna() | d["ended_at"].isna(),
        "duration_nonpositive": lambda d: duration[d.index] <= 0,
        "duration_under_60s": lambda d: duration[d.index] < MIN_DURATION_S,
        "duration_over_24h": lambda d: duration[d.index] > MAX_DURATION_S,
        "test_station": lambda d: (
            _is_test_station(d["start_station_id"], d["start_station_name"])
            | _is_test_station(d["end_station_id"], d["end_station_name"])
        ),
        "no_station": lambda d: d["start_station_id"].isna() & d["end_station_id"].isna(),
    }
    removed = {}
    for name, rule in rules.items():
        mask = rule(df).fillna(False).astype(bool)
        removed[name] = int(mask.sum())
        df = df[~mask]
    return df.reset_index(drop=True), removed


def clean_csv(csv_path: Path, out_path: Path) -> dict[str, int]:
    """Clean one monthly CSV to Parquet. Returns counts for the cleaning log."""
    raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    df = normalize(raw)
    kept, removed = apply_rules(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    kept.to_parquet(tmp, index=False)
    tmp.rename(out_path)  # only a complete file ever has the final name
    counts = {"rows_in": len(raw), **removed, "rows_out": len(kept)}
    log.info("%s: %s", csv_path.name, counts)
    return counts
