"""Month-at-a-time data pipeline: download, clean to Parquet, delete the raw CSV.

Only one month of raw data is on disk at any time. Months that already have Parquet are skipped,
so the pipeline can be stopped and restarted safely.
"""

import argparse
import logging

import pandas as pd

from bikecast import config
from bikecast.data.clean import clean_csv
from bikecast.data.download import download_month, fetch_month_keys

log = logging.getLogger(__name__)

CLEANING_LOG = config.INTERIM / "cleaning_log.csv"


def interim_path(month: str):
    return config.INTERIM / f"trips_{month}.parquet"


def record_counts(month: str, counts: dict[str, int]) -> None:
    """Write this month's counts to the cleaning log, replacing any earlier entry."""
    row = pd.DataFrame([{"month": month, **counts}])
    if CLEANING_LOG.exists():
        existing = pd.read_csv(CLEANING_LOG, dtype={"month": str})
        row = pd.concat([existing[existing["month"] != month], row])
    row.sort_values("month").to_csv(CLEANING_LOG, index=False)


def process_month(month: str, key: str) -> None:
    out = interim_path(month)
    if out.exists():
        log.info("%s: Parquet exists, skipping", month)
        return
    csv_path = download_month(month, key)
    counts = clean_csv(csv_path, out)
    record_counts(month, counts)
    csv_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="*", help="YYYYMM months (default: all available)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    keys = fetch_month_keys()
    log.info("Found %d months: %s to %s", len(keys), min(keys), max(keys))
    for month, key in keys.items():
        if not args.months or month in args.months:
            process_month(month, key)


if __name__ == "__main__":
    main()
