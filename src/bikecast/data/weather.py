"""Hourly Boston weather from Open-Meteo: archived forecasts (default) and actuals.

The Historical Forecast API stores what weather models predicted at the time, which is closer to
what an operator would know the night before than actual observations. It is stitched from the
first hours of each model run, so it is still somewhat more accurate than a true 24-hour-ahead
forecast. Actuals (archive API) are kept for a sensitivity check.

We request UTC and convert to naive Boston wall-clock time to match the trip grid. At fall-back the
two UTC hours that map to local 1am are averaged; at spring-forward the missing local 2am is
interpolated.
"""

import argparse
import logging
from datetime import datetime

import pandas as pd
import requests

from bikecast import config
from bikecast.data.aggregate import data_range

log = logging.getLogger(__name__)

SOURCES = {
    "forecast": "https://historical-forecast-api.open-meteo.com/v1/forecast",
    "actual": "https://archive-api.open-meteo.com/v1/archive",
}
VARIABLES = {
    "temperature_2m": "temp_c",
    "precipitation": "precip_mm",
    "wind_speed_10m": "wind_kmh",
}


def output_path(source: str):
    return config.PROCESSED / f"weather_{source}.parquet"


def fetch(source: str, start: datetime, end: datetime) -> dict:
    """Raw JSON for [start, end) local days. Pads a day each side so every local hour is covered."""
    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "start_date": (start - pd.Timedelta(days=1)).date().isoformat(),
        "end_date": end.date().isoformat(),
        "hourly": ",".join(VARIABLES),
        "timezone": "GMT",
    }
    resp = requests.get(SOURCES[source], params=params, timeout=120)
    resp.raise_for_status()
    return resp.json()


def to_local_grid(payload: dict, start: datetime, end: datetime) -> pd.DataFrame:
    """Convert an Open-Meteo UTC hourly payload to the naive local hourly grid [start, end)."""
    hourly = payload["hourly"]
    df = pd.DataFrame({new: hourly[old] for old, new in VARIABLES.items()})
    utc = pd.to_datetime(pd.Series(hourly["time"]), format="ISO8601").dt.tz_localize("UTC")
    df["ts"] = utc.dt.tz_convert(config.TIMEZONE).dt.tz_localize(None)
    df = df.groupby("ts").mean()  # fall-back: two UTC hours share one local hour
    grid = pd.date_range(start, end, freq="h", inclusive="left", name="ts")
    df = df.reindex(df.index.union(grid)).interpolate(method="time", limit=2).reindex(grid)
    if df.isna().any().any():
        missing = df[df.isna().any(axis=1)].index
        raise ValueError(f"Weather gaps after conversion: {len(missing)} hours, first {missing[0]}")
    return df.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Re-download even if cached")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    start, end = data_range()
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    for source in SOURCES:
        path = output_path(source)
        if path.exists() and not args.refresh:
            cached = pd.read_parquet(path, columns=["ts"])
            if cached["ts"].min() <= pd.Timestamp(start) and cached["ts"].max() >= pd.Timestamp(
                end
            ) - pd.Timedelta(hours=1):
                log.info("%s: cached, covers range, skipping", source)
                continue
        df = to_local_grid(fetch(source, start, end), start, end)
        df.to_parquet(path, index=False)
        log.info(
            "%s: %d hours, temp %.1f to %.1f C, %.0f%% of hours with precipitation",
            source,
            len(df),
            df["temp_c"].min(),
            df["temp_c"].max(),
            100 * (df["precip_mm"] > 0).mean(),
        )


if __name__ == "__main__":
    main()
