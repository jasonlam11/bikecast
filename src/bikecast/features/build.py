"""Model-ready feature table: one row per (station, hour) with lags, calendar, and weather.

Day-ahead rule: features for any hour of day D may use demand only from before D (midnight).
- Lags of 24, 48, and 168 hours. For hour h of day D, lag 24 is hour h of D-1, always before D.
- Same-hour mean over the previous 7 days (lags 24, 48, ..., 168).
- Previous-day mean: the mean of all 24 hours of D-1, identical for every hour of D.
Calendar features are known in advance. Weather is the archived forecast for the hour.
"""

import logging

import numpy as np
import pandas as pd

from bikecast import config
from bikecast.models.baseline import TARGETS

log = logging.getLogger(__name__)

LAGS = [24, 48, 168]
PRECIP_CLIP_MM = 30.0
CALENDAR_COLS = ["hour", "dow", "month", "is_weekend", "is_holiday", "is_semester"]


def check_complete_grid(hourly: pd.DataFrame) -> None:
    """Row shifts equal time shifts only if every station has every hour, in order."""
    steps = hourly.groupby("station_id")["ts"].diff().dropna()
    if not (steps == pd.Timedelta(hours=1)).all():
        raise ValueError("Hourly data is not a complete, sorted grid; run aggregate.py")


def demand_features(hourly: pd.DataFrame) -> pd.DataFrame:
    df = hourly.sort_values(["station_id", "ts"]).reset_index(drop=True)
    check_complete_grid(df)
    by_station = df.groupby("station_id")
    for target in TARGETS:
        s = by_station[target]
        for lag in LAGS:
            df[f"{target}_lag{lag}"] = s.shift(lag)
        same_hour = [s.shift(24 * k) for k in range(1, 8)]
        df[f"{target}_same_hour_mean_7d"] = pd.concat(same_hour, axis=1).mean(axis=1, skipna=False)

    df["date"] = df["ts"].dt.normalize()
    daily = df.groupby(["station_id", "date"])[TARGETS].mean()
    prev = daily.groupby(level="station_id").shift(1).add_suffix("_prev_day_mean")
    return df.join(prev, on=["station_id", "date"]).drop(columns="date")


def weather_features(weather: pd.DataFrame) -> pd.DataFrame:
    w = weather[["ts", "temp_c", "wind_kmh"]].copy()
    # A few forecast hours have model artifacts (119.5 mm in one hour); clip and log-transform.
    w["precip_log"] = np.log1p(weather["precip_mm"].clip(0, PRECIP_CLIP_MM))
    return w


def build_feature_table(
    hourly: pd.DataFrame, weather: pd.DataFrame, calendar: pd.DataFrame
) -> pd.DataFrame:
    df = demand_features(hourly)
    df = df.merge(calendar[["ts", *CALENDAR_COLS]], on="ts", how="left")
    df = df.merge(weather_features(weather), on="ts", how="left")
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    P = config.PROCESSED
    weather = pd.read_parquet(P / "weather_forecast.parquet")
    calendar = pd.read_parquet(P / "calendar.parquet")
    for name in ["hourly", "hourly_system"]:
        table = build_feature_table(pd.read_parquet(P / f"{name}.parquet"), weather, calendar)
        out = P / f"features_{name.removeprefix('hourly').strip('_') or 'stations'}.parquet"
        table.to_parquet(out, index=False)
        log.info("%s: %d rows, %d columns -> %s", name, len(table), table.shape[1], out.name)


if __name__ == "__main__":
    main()
