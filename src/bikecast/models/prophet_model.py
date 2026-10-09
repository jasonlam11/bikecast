"""Prophet: one model per station per target.

Design choices (see DECISIONS.md):
- Two conditional daily curves, one for working days and one for weekends and holidays. A single
  daily curve would average the weekday commute peaks with the weekend midday hump.
- Weekly and yearly seasonality, US + Massachusetts holidays, and extra regressors: forecast
  temperature, precipitation (clipped, log1p), wind, and the semester flag.
- Raw counts as the target. Forecasts and the 80% interval are clipped at 0.

Prophet uses no recent lags, so one fit per test week forecasts all 7 days of that week. `fit`
does the fitting and forecasting for the week after the training data ends; `predict` returns the
slice for one day. The regressors for the forecast days are the archived weather forecast and the
calendar, both known the night before.
"""

import logging
import os
import zlib
from concurrent.futures import ProcessPoolExecutor

import holidays
import numpy as np
import pandas as pd

from bikecast.features.build import weather_features
from bikecast.models.baseline import TARGETS

log = logging.getLogger(__name__)

REGRESSORS = ["temp_c", "precip_log", "wind_kmh", "semester"]


def regressor_frame(calendar: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Hourly regressors and seasonality conditions, keyed by ds."""
    cal = calendar[["ts", "is_weekend", "is_holiday", "is_semester"]]
    df = cal.merge(weather_features(weather), on="ts", how="inner").rename(columns={"ts": "ds"})
    df["workday"] = ~df["is_weekend"] & ~df["is_holiday"]
    df["offday"] = ~df["workday"]
    df["semester"] = df["is_semester"].astype(float)
    return df[["ds", "workday", "offday", *REGRESSORS]]


def holiday_frame(years) -> pd.DataFrame:
    ma = holidays.country_holidays("US", subdiv="MA", years=list(years))
    return pd.DataFrame(
        {"holiday": list(ma.values()), "ds": pd.to_datetime(list(ma.keys()))}
    ).sort_values("ds")


def _fit_and_forecast(job: dict) -> pd.DataFrame:
    """Runs in a worker process: fit one Prophet and forecast the horizon."""
    from prophet import Prophet  # imported here so worker start-up stays light

    logging.getLogger("cmdstanpy").disabled = True
    m = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=job["yearly"],
        holidays=job["holidays"],
        interval_width=job["interval_width"],
        uncertainty_samples=job["uncertainty_samples"],
    )
    m.add_seasonality("daily_workday", period=1, fourier_order=12, condition_name="workday")
    m.add_seasonality("daily_offday", period=1, fourier_order=12, condition_name="offday")
    for r in REGRESSORS:
        m.add_regressor(r)
    m.fit(job["train"])
    # Prophet draws its interval by simulation with numpy's global random state. Seed it per
    # station and target so intervals are identical on every run and in any worker order.
    np.random.seed(zlib.crc32(f"{job['station_id']}|{job['target']}".encode()))
    fc = m.predict(job["future"])[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    for c in ["yhat", "yhat_lower", "yhat_upper"]:
        fc[c] = fc[c].clip(lower=0)
    fc["station_id"], fc["target"] = job["station_id"], job["target"]
    return fc


class ProphetModel:
    name = "prophet"

    def __init__(
        self,
        calendar: pd.DataFrame,
        weather: pd.DataFrame,
        horizon_days: int = 7,
        workers: int | None = None,
        interval_width: float = 0.8,
        uncertainty_samples: int = 300,
        yearly: bool = True,
    ):
        self.regressors = regressor_frame(calendar, weather)
        self.horizon_days = horizon_days
        self.workers = workers if workers is not None else min(6, os.cpu_count() or 1)
        self.interval_width = interval_width
        self.uncertainty_samples = uncertainty_samples
        self.yearly = yearly
        self.forecast: pd.DataFrame | None = None

    def _jobs(self, train: pd.DataFrame) -> list[dict]:
        start = train["ts"].max() + pd.Timedelta(hours=1)
        horizon = pd.date_range(start, periods=self.horizon_days * 24, freq="h")
        future = self.regressors[self.regressors["ds"].isin(horizon)].reset_index(drop=True)
        if len(future) != len(horizon):
            raise ValueError("Missing weather or calendar rows for the forecast horizon")
        hol = holiday_frame(range(train["ts"].dt.year.min(), horizon[-1].year + 1))
        visible = train[train["in_service"]]
        jobs = []
        for station_id, g in visible.groupby("station_id"):
            for target in TARGETS:
                df = g[["ts", target]].rename(columns={"ts": "ds", target: "y"})
                df = df.merge(self.regressors, on="ds", how="inner")
                jobs.append(
                    {
                        "station_id": station_id,
                        "target": target,
                        "train": df,
                        "future": future,
                        "holidays": hol,
                        "yearly": self.yearly,
                        "interval_width": self.interval_width,
                        "uncertainty_samples": self.uncertainty_samples,
                    }
                )
        return jobs

    def fit(self, train: pd.DataFrame) -> "ProphetModel":
        jobs = self._jobs(train)
        if self.workers > 1:
            with ProcessPoolExecutor(max_workers=self.workers) as pool:
                results = list(pool.map(_fit_and_forecast, jobs))
        else:
            results = [_fit_and_forecast(j) for j in jobs]
        self.forecast = pd.concat(results, ignore_index=True)
        log.info("prophet: %d models fit through %s", len(jobs), train["ts"].max())
        return self

    def predict(self, history: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
        if self.forecast is None:
            raise RuntimeError("fit() must run before predict()")
        fc = self.forecast[
            (self.forecast["ds"] >= day) & (self.forecast["ds"] < day + pd.Timedelta(days=1))
        ]
        if fc.empty:
            raise ValueError(f"{day.date()} is outside the fitted forecast horizon")
        wide = fc.pivot_table(
            index=["station_id", "ds"],
            columns="target",
            values=["yhat", "yhat_lower", "yhat_upper"],
        )
        out = pd.DataFrame(index=wide.index)
        for target in TARGETS:
            out[target] = wide[("yhat", target)]
            out[f"{target}_lower"] = wide[("yhat_lower", target)]
            out[f"{target}_upper"] = wide[("yhat_upper", target)]
        return out.reset_index().rename(columns={"ds": "ts"})
