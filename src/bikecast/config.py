"""Project paths and fixed settings. Test weeks live here so nothing tunes on them by accident."""

from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

BUCKET_URL = "https://s3.amazonaws.com/hubway-data"
FIRST_MONTH = "202401"

# Boston (Open-Meteo snaps to the nearest grid cell)
LATITUDE = 42.36
LONGITUDE = -71.06
TIMEZONE = "America/New_York"

N_TOP_STATIONS = 50
# Stations are ranked on data before the first test week only.
STATION_RANK_START = date(2024, 1, 1)
STATION_RANK_END = date(2025, 1, 1)  # exclusive

# Monday starts of the rolling-origin test weeks (each runs Mon to Sun).
TEST_WEEKS = [
    date(2025, 1, 13),
    date(2025, 4, 14),
    date(2025, 7, 14),
    date(2025, 10, 13),
    date(2026, 1, 12),
    date(2026, 3, 9),
    date(2026, 5, 11),
    date(2026, 6, 15),
    date(2026, 7, 13),
    date(2026, 9, 14),
]
