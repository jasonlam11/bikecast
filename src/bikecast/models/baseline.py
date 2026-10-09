"""Baseline forecasts. Every other model must beat these to matter.

Model interface used by the backtest:
    fit(train)            train: hourly rows with ts before the test week
    predict(history, day) history: hourly rows with ts before `day`; returns one row per
                          (station_id, hour of day) with predicted departures and arrivals

`history` has an `in_service` column computed on history alone, so it never reflects later data.
"""

import pandas as pd

TARGETS = ["departures", "arrivals"]


def day_grid(station_ids, day: pd.Timestamp) -> pd.DataFrame:
    hours = pd.date_range(day, periods=24, freq="h")
    return pd.MultiIndex.from_product(
        [sorted(station_ids), hours], names=["station_id", "ts"]
    ).to_frame(index=False)


class HistoricalAverage:
    """Mean of the same station, weekday, and hour over the previous `weeks` weeks.

    Out-of-service hours are skipped. A station with no in-service history gets 0.
    """

    name = "historical_average"

    def __init__(self, weeks: int = 8):
        self.weeks = weeks

    def fit(self, train: pd.DataFrame) -> "HistoricalAverage":
        return self

    def predict(self, history: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
        recent = history[
            (history["ts"] >= day - pd.Timedelta(weeks=self.weeks))
            & (history["ts"].dt.dayofweek == day.dayofweek)
            & history["in_service"]
        ]
        means = recent.groupby(["station_id", recent["ts"].dt.hour])[TARGETS].mean()
        grid = day_grid(history["station_id"].unique(), day)
        keys = pd.MultiIndex.from_arrays([grid["station_id"], grid["ts"].dt.hour])
        out = means.reindex(keys).fillna(0.0).reset_index(drop=True)
        return pd.concat([grid, out], axis=1)


class SeasonalNaive:
    """Same station, same hour, one week earlier.

    If that hour was out of service (station closed a week ago), we fall back to the historical
    average, since copying a closure forward would forecast 0 for a station that is open again.
    """

    name = "seasonal_naive"

    def __init__(self):
        self.fallback = HistoricalAverage()

    def fit(self, train: pd.DataFrame) -> "SeasonalNaive":
        return self

    def predict(self, history: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
        week_ago = history[
            (history["ts"] >= day - pd.Timedelta(days=7))
            & (history["ts"] < day - pd.Timedelta(days=6))
        ].copy()
        week_ago["ts"] = week_ago["ts"] + pd.Timedelta(days=7)
        out = self.fallback.predict(history, day).set_index(["station_id", "ts"])
        lag = week_ago[week_ago["in_service"]].set_index(["station_id", "ts"])[TARGETS]
        out.update(lag.astype(float))
        return out.reset_index()
