"""Find and download monthly Bluebikes trip files from the public S3 bucket.

The bucket mixes file formats (.zip, .csv.zip, bare .csv), so keys are discovered from the
bucket listing instead of building URLs by hand.
"""

import argparse
import logging
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile
from datetime import date
from pathlib import Path

import requests

from bikecast import config

log = logging.getLogger(__name__)

KEY_PATTERN = re.compile(r"^(\d{6})-bluebikes-tripdata\.(zip|csv\.zip|csv)$")
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def parse_listing(xml_text: str) -> list[str]:
    """Return every object key in an S3 ListObjectsV2 response."""
    root = ET.fromstring(xml_text)
    return [el.text for el in root.iter(f"{S3_NS}Key") if el.text]


def month_keys(keys: list[str], first: str, last: str) -> dict[str, str]:
    """Map YYYYMM to bucket key for trip files in [first, last]."""
    out = {}
    for key in keys:
        m = KEY_PATTERN.match(key)
        if m and first <= m.group(1) <= last:
            if m.group(1) in out:
                raise ValueError(f"Two files for month {m.group(1)}: {out[m.group(1)]}, {key}")
            out[m.group(1)] = key
    return dict(sorted(out.items()))


def latest_full_month(today: date) -> str:
    """The month before today's month, as YYYYMM."""
    year, month = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
    return f"{year}{month:02d}"


def fetch_month_keys(first: str = config.FIRST_MONTH, last: str | None = None) -> dict[str, str]:
    last = last or latest_full_month(date.today())
    resp = requests.get(config.BUCKET_URL, params={"list-type": "2"}, timeout=60)
    resp.raise_for_status()
    if "<IsTruncated>true" in resp.text:
        raise RuntimeError("Bucket listing is truncated; pagination is needed")
    return month_keys(parse_listing(resp.text), first, last)


def download_month(month: str, key: str, raw_dir: Path = config.RAW) -> Path:
    """Download one month and return the path to its CSV. Zips are extracted and removed."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / f"{month}.csv"
    if csv_path.exists():
        log.info("%s: raw CSV already present, skipping download", month)
        return csv_path

    dest = raw_dir / key
    with requests.get(f"{config.BUCKET_URL}/{key}", stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            shutil.copyfileobj(resp.raw, f, length=1 << 20)

    if key.endswith(".zip"):
        extract_csv(dest, csv_path)
        dest.unlink()
    else:
        dest.rename(csv_path)
    log.info("%s: downloaded %s (%.0f MB)", month, key, csv_path.stat().st_size / 1e6)
    return csv_path


def extract_csv(zip_path: Path, csv_path: Path) -> None:
    """Extract the single trip CSV, ignoring macOS metadata entries."""
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX/")]
        if len(names) != 1:
            raise ValueError(f"Expected one CSV in {zip_path.name}, found {names}")
        # Write to our own filename so archive paths are never used on disk.
        with zf.open(names[0]) as src, open(csv_path, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="*", help="YYYYMM months (default: all available)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    keys = fetch_month_keys()
    for month, key in keys.items():
        if not args.months or month in args.months:
            download_month(month, key)


if __name__ == "__main__":
    main()
