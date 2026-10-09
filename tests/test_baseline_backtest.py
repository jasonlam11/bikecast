import numpy as np
import pandas as pd
import pytest

from bikecast.evaluation.backtest import forecast_days, run_backtest, visible_slice
from bikecast.models.baseline import HistoricalAverage, SeasonalNaive

START = pd.Timestamp("2024-01-01")  # a Monday


def synthetic(days: int = 70, stations=("A", "B")) -> pd.DataFrame:
    """Complete hourly grid. departures = day index, arrivals = hour, so values are traceable."""
    ts = pd.date_range(START, periods=days * 24, freq="h")
    frames = []
    for i, s in enumerate(stations):
        frames.append(
            pd.DataFrame(
                {
                    "station_id": s,
                    "ts": ts,
                    "departures": ((ts - START).days + 100 * i).astype(float),
                    "arrivals": ts.hour.astype(float),
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    df["in_service"] = True
    return df


def test_seasonal_naive_copies_same_hour_one_week_earlier():
    data = synthetic()
    day = START + pd.Timedelta(days=30)
    pred = SeasonalNaive().predict(visible_slice(data, day), day).set_index(["station_id", "ts"])
    assert len(pred) == 2 * 24
    assert (pred.loc["A", "departures"] == 23).all()  # day 30 - 7
    assert (pred.loc["B", "departures"] == 123).all()
    assert (pred.loc["A", "arrivals"].to_numpy() == np.arange(24)).all()


def test_historical_average_uses_last_8_same_weekdays():
    data = synthetic()
    day = START + pd.Timedelta(days=63)  # Monday; previous Mondays are days 7, 14, ..., 56
    pred = HistoricalAverage(weeks=8).predict(visible_slice(data, day), day)
    a = pred[pred["station_id"] == "A"]
    assert (a["departures"] == np.mean([7, 14, 21, 28, 35, 42, 49, 56])).all()


def test_seasonal_naive_falls_back_when_last_week_was_closed():
    data = synthetic()
    day = START + pd.Timedelta(days=30)
    # Station A closed for 3 days covering last week's same day: idle for 72 hours.
    closed = (data["station_id"] == "A") & data["ts"].between(
        day - pd.Timedelta(days=8), day - pd.Timedelta(days=5)
    )
    data.loc[closed, ["departures", "arrivals"]] = 0
    pred = SeasonalNaive().predict(visible_slice(data, day), day).set_index(["station_id", "ts"])
    # Not 0 (the closure), but the historical average of earlier in-service Tuesdays.
    expected = HistoricalAverage().predict(visible_slice(data, day), day)
    expected = expected.set_index(["station_id", "ts"])
    assert (pred.loc["A", "departures"] > 0).all()
    assert np.allclose(pred.loc["A", "departures"], expected.loc["A", "departures"])


class Spy:
    """Records the latest timestamp it was ever shown."""

    name = "spy"

    def __init__(self):
        self.fit_max, self.predict_max = [], []

    def fit(self, train):
        self.fit_max.append(train["ts"].max())
        return self

    def predict(self, history, day):
        self.predict_max.append((history["ts"].max(), day))
        return SeasonalNaive().predict(history, day)


def test_backtest_never_shows_models_the_future():
    data = synthetic()
    weeks = [START + pd.Timedelta(days=28), START + pd.Timedelta(days=49)]
    spy = Spy()
    preds = run_backtest(data, [spy], weeks=weeks)
    assert spy.fit_max == [w - pd.Timedelta(hours=1) for w in weeks]
    assert all(seen == day - pd.Timedelta(hours=1) for seen, day in spy.predict_max)
    # 2 weeks x 7 days x 24 hours x 2 stations x 2 targets
    assert len(preds) == 2 * 7 * 24 * 2 * 2
    assert preds.groupby("test_week")["ts"].nunique().tolist() == [168, 168]


def test_forecast_days_are_the_seven_days_of_the_week():
    days = forecast_days(START)
    assert days[0] == START and days[-1] == START + pd.Timedelta(days=6)


def test_backtest_rejects_predictions_outside_the_day():
    class Bad:
        name = "bad"

        def fit(self, train):
            return self

        def predict(self, history, day):
            p = SeasonalNaive().predict(history, day)
            p["ts"] = p["ts"] + pd.Timedelta(days=1)
            return p

    with pytest.raises(ValueError, match="outside"):
        run_backtest(synthetic(), [Bad()], weeks=[START + pd.Timedelta(days=28)])
