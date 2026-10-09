"""Data functions behind the Streamlit app. Kept out of the app file so they can be unit tested.

Everything reads the committed snapshot in reports/app_data/; nothing here trains a model or
touches raw data.
"""

from pathlib import Path

import pandas as pd

from bikecast import config
from bikecast.evaluation.report import model_order, scored_table

APP_DATA = config.REPORTS / "app_data"
WINDOWS = {"Morning, 7 to 10am": (7, 10), "Evening, 4 to 7pm": (16, 19)}
MODEL_LABELS = {
    "seasonal_naive": "Seasonal naive",
    "historical_average": "Historical average",
    "prophet": "Prophet",
    "torch_mlp": "Global MLP (PyTorch)",
}


def load_snapshot(app_data: Path = APP_DATA) -> dict[str, pd.DataFrame]:
    out = {}
    for level in ["stations", "system"]:
        p = pd.read_parquet(app_data / f"predictions_{level}.parquet")
        for c in ["model", "station_id", "target", "weather", "holiday_name"]:
            if c in p:
                p[c] = p[c].astype(str).replace({"<NA>": None, "nan": None})
        p["date"] = p["ts"].dt.normalize()
        p["hour"] = p["ts"].dt.hour
        out[level] = p
    out["stations_info"] = pd.read_parquet(app_data / "stations.parquet")
    return out


def window_net_flow(preds: pd.DataFrame, model: str, day, hours=(7, 10)) -> pd.DataFrame:
    """Forecast and actual net flow per station for one day and time window.

    Stations with any closed hour in the window are left out.
    """
    lo, hi = hours
    p = preds[
        (preds["model"] == model) & (preds["date"] == pd.Timestamp(day))
        & preds["hour"].between(lo, hi - 1)
    ]  # fmt: skip
    sign = p["target"].map({"arrivals": 1, "departures": -1})
    p = p.assign(forecast=p["yhat"] * sign, actual=p["y"] * sign)
    g = p.groupby("station_id")
    out = g[["forecast", "actual"]].sum()
    out = out[g.size() == 2 * (hi - lo)]
    return out.reset_index()


def rebalancing_list(flow: pd.DataFrame, info: pd.DataFrame, n: int = 10):
    """Stations with the largest forecast deficit (bring bikes) and surplus (free docks)."""
    t = flow.merge(info[["station_id", "name"]], on="station_id")
    t = t[["name", "station_id", "forecast", "actual"]].round(1)
    drains = t.nsmallest(n, "forecast")
    fills = t.nlargest(n, "forecast")
    return drains[drains["forecast"] < 0], fills[fills["forecast"] > 0]


def comparison_table(preds: pd.DataFrame) -> pd.DataFrame:
    """MAE, RMSE, WAPE and skill vs both baselines, departures and arrivals pooled."""
    t = scored_table(preds.assign(level="all"), ["level"]).drop(columns="level")
    t["model"] = t["model"].astype(str)
    t = t.set_index("model").loc[model_order(t["model"])]
    return t[["MAE", "RMSE", "WAPE", "skill", "skill vs hist avg"]]


def backtest_days(preds: pd.DataFrame) -> list[pd.Timestamp]:
    return sorted(preds["date"].unique())


def week_of(preds: pd.DataFrame, day) -> pd.Timestamp:
    return preds.loc[preds["date"] == pd.Timestamp(day), "test_week"].iloc[0]
