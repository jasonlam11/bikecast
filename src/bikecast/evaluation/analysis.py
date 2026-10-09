"""Shared error-analysis helpers used by the error-analysis notebook, rebalancing.py, and the app.

Functions take prediction frames (long format from backtest.py) so the same numbers come out
whether they are computed from data/predictions or from the committed app snapshot.
"""

import pandas as pd

from bikecast import config
from bikecast.evaluation.metrics import coverage, mae
from bikecast.evaluation.report import STRONG_BASELINE, add_groups, load_predictions

MORNING_HOURS = (7, 10)  # 7am up to 10am: hours 7, 8, 9


def enrich(preds: pd.DataFrame, weather_actual: pd.DataFrame, calendar: pd.DataFrame):
    """Add season, hour band, rain flag, holiday, and date columns."""
    p = add_groups(preds, weather_actual)
    cal = calendar[["ts", "is_holiday", "holiday_name"]]
    p = p.merge(cal, on="ts", how="left")
    p["date"] = p["ts"].dt.normalize()
    p["hour"] = p["ts"].dt.hour
    return p


def load_enriched_station_predictions() -> pd.DataFrame:
    P = config.PROCESSED
    return enrich(
        load_predictions("stations"),
        pd.read_parquet(P / "weather_actual.parquet"),
        pd.read_parquet(P / "calendar.parquet"),
    )


def _mae(d: pd.DataFrame) -> float:
    return mae(d["y"], d["yhat"])


def station_scores(preds: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """MAE per station and model, skill vs the historical average, and station volume."""
    t = preds.groupby(["station_id", "model"]).apply(_mae, include_groups=False).unstack("model")
    out = pd.DataFrame(index=t.index)
    for m in t.columns:
        out[f"mae_{m}"] = t[m]
        out[f"skill_{m}"] = 1 - t[m] / t[STRONG_BASELINE]
    actual = preds[preds["model"] == STRONG_BASELINE].groupby("station_id")["y"].mean()
    out["mean_actual"] = actual
    info = stations.set_index("canonical_id")[["name", "rank_volume"]]
    return out.join(info).sort_values("rank_volume", ascending=False)


def bias_by_hour(preds: pd.DataFrame) -> pd.DataFrame:
    """Mean of forecast minus actual by hour. Positive means the model forecasts too many trips."""
    err = preds.assign(err=preds["yhat"] - preds["y"])
    return err.groupby(["hour", "model"])["err"].mean().unstack("model")


def daily_errors(preds: pd.DataFrame, weather_actual: pd.DataFrame) -> pd.DataFrame:
    """MAE per model and date with that day's weather and holiday context."""
    t = preds.groupby(["date", "model"]).apply(_mae, include_groups=False).rename("mae")
    t = t.reset_index()
    day = preds.groupby("date").agg(
        is_holiday=("is_holiday", "first"),
        holiday_name=("holiday_name", "first"),
        actual_trips=("y", "sum"),
    )
    w = weather_actual.set_index("ts")["precip_mm"].resample("D").sum().rename("precip_mm")
    t = t.join(day, on="date").join(w, on="date")
    t["weekday"] = t["date"].dt.day_name()
    return t


def worst_days(daily: pd.DataFrame, model: str, n: int = 10) -> pd.DataFrame:
    return daily[daily["model"] == model].nlargest(n, "mae").reset_index(drop=True)


def coverage_by_hour(preds: pd.DataFrame) -> pd.Series:
    p = preds[preds["yhat_lower"].notna()] if "yhat_lower" in preds else preds.iloc[0:0]
    return p.groupby("hour").apply(
        lambda d: coverage(d["y"], d["yhat_lower"], d["yhat_upper"]), include_groups=False
    )


def morning_net_flow(preds: pd.DataFrame, model: str, hours=MORNING_HOURS) -> pd.DataFrame:
    """Forecast and actual net flow (arrivals minus departures) per station and weekday morning.

    Only non-holiday weekdays where the station was in service for every morning hour, for both
    targets, are kept, so a partial closure cannot look like a drain.
    """
    lo, hi = hours
    p = preds[
        (preds["model"] == model)
        & preds["hour"].between(lo, hi - 1)
        & (preds["ts"].dt.dayofweek < 5)
        & ~preds["is_holiday"].fillna(False).astype(bool)
    ]
    sign = p["target"].map({"arrivals": 1, "departures": -1})
    p = p.assign(forecast=p["yhat"] * sign, actual=p["y"] * sign)
    g = p.groupby(["station_id", "date"])
    out = g[["forecast", "actual"]].sum()
    out["rows"] = g.size()
    complete = out["rows"] == 2 * (hi - lo)
    return out[complete].drop(columns="rows").reset_index()
