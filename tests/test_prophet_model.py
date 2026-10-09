import numpy as np
import pandas as pd
import pytest

from bikecast.data.calendar import calendar_features
from bikecast.evaluation.backtest import run_backtest
from bikecast.models.prophet_model import ProphetModel

START = pd.Timestamp("2024-02-05")  # Monday, no holidays in the window
DAYS = 42


def commute_data(seed=0):
    """One station: weekdays peak at 8am and 5pm, weekends have a midday hump."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(START, periods=DAYS * 24, freq="h")
    h = ts.hour.to_numpy()
    weekday = 2 + 10 * np.exp(-((h - 8) ** 2) / 2) + 12 * np.exp(-((h - 17) ** 2) / 3)
    weekend = 2 + 8 * np.exp(-((h - 14) ** 2) / 10)
    mean = np.where(ts.dayofweek < 5, weekday, weekend)
    return pd.DataFrame(
        {
            "station_id": "A",
            "ts": ts,
            "departures": rng.poisson(mean).astype(float),
            "arrivals": rng.poisson(mean).astype(float),
            "in_service": True,
        }
    )


@pytest.fixture(scope="module")
def setup():
    data = commute_data()
    ts = pd.Series(pd.date_range(START, periods=(DAYS + 7) * 24, freq="h"))
    weather = pd.DataFrame({"ts": ts, "temp_c": 10.0, "wind_kmh": 10.0, "precip_mm": 0.0})
    weather["temp_c"] += np.random.default_rng(1).normal(0, 1, len(ts))  # avoid a constant column
    return data, calendar_features(ts), weather


def fast_model(cal, weather):
    return ProphetModel(cal, weather, workers=1, uncertainty_samples=100, yearly=False)


@pytest.fixture(scope="module")
def fitted(setup):
    data, cal, weather = setup
    train = data[data["ts"] < START + pd.Timedelta(days=35)]
    return fast_model(cal, weather).fit(train)


def test_prophet_learns_weekday_and_weekend_shapes(fitted):
    monday = fitted.predict(None, START + pd.Timedelta(days=35)).set_index("ts")
    saturday = fitted.predict(None, START + pd.Timedelta(days=40)).set_index("ts")
    assert monday["departures"].iloc[8] > 2 * monday["departures"].iloc[12]  # commute peak
    assert saturday["departures"].iloc[14] > 2 * saturday["departures"].iloc[8]  # midday hump


def test_prophet_intervals_are_ordered_and_non_negative(fitted):
    p = fitted.predict(None, START + pd.Timedelta(days=36))
    for t in ["departures", "arrivals"]:
        assert (p[f"{t}_lower"] >= 0).all()
        assert (p[f"{t}_lower"] <= p[t]).all() and (p[t] <= p[f"{t}_upper"]).all()


def test_prophet_refuses_days_outside_its_horizon(fitted):
    with pytest.raises(ValueError, match="horizon"):
        fitted.predict(None, START + pd.Timedelta(days=50))


def test_prophet_runs_in_backtest_with_intervals(setup):
    data, cal, weather = setup
    preds = run_backtest(data, [fast_model(cal, weather)], weeks=[START + pd.Timedelta(days=35)])
    assert len(preds) == 7 * 24 * 2
    assert {"yhat_lower", "yhat_upper"} <= set(preds.columns)
    assert preds[["yhat", "yhat_lower", "yhat_upper"]].notna().all().all()
