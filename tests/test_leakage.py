"""No-leakage tests: anything computed for day D must not change when data from D onward changes.

Method: build features or forecasts for D, then corrupt every demand value at or after D (set to
huge numbers) and build them again. If any feature or forecast for D moves, it used the future.

Weather is the exception by design: the archived forecast for D is a legitimate input, because a
forecast for D exists the night before. So we corrupt weather only after D.
"""

import numpy as np
import pandas as pd
import pytest

from bikecast.data.calendar import calendar_features
from bikecast.evaluation.backtest import run_backtest, visible_slice
from bikecast.features.build import build_feature_table, demand_features
from bikecast.models.baseline import TARGETS, HistoricalAverage, SeasonalNaive

START = pd.Timestamp("2024-01-01")
DAYS = 70
HUGE = 1e6


def realistic_hourly(seed: int = 0) -> pd.DataFrame:
    """Random Poisson counts with a daily shape and a closure, on a complete grid."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(START, periods=DAYS * 24, freq="h")
    shape = 1 + 3 * np.exp(-((ts.hour - 8) ** 2) / 4) + 4 * np.exp(-((ts.hour - 17) ** 2) / 6)
    frames = []
    for i, s in enumerate(["A", "B", "C"]):
        df = pd.DataFrame(
            {
                "station_id": s,
                "ts": ts,
                "departures": rng.poisson(shape * (1 + i)).astype(float),
                "arrivals": rng.poisson(shape * (1 + i)).astype(float),
            }
        )
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    # Station C closes on day 40 and stays closed for 6 days, straddling some cutoffs.
    closed = (df["station_id"] == "C") & df["ts"].between(
        START + pd.Timedelta(days=40), START + pd.Timedelta(days=46)
    )
    df.loc[closed, TARGETS] = 0.0
    df["in_service"] = True
    return df


def weather_for(ts: pd.Series, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "ts": ts,
            "temp_c": rng.normal(10, 5, len(ts)),
            "wind_kmh": rng.uniform(0, 30, len(ts)),
            "precip_mm": rng.exponential(0.3, len(ts)),
        }
    )


def corrupt_demand_from(df: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    out = df.copy()
    out.loc[out["ts"] >= day, TARGETS] = HUGE
    return out


def corrupt_weather_after(weather: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    out = weather.copy()
    after = out["ts"] >= day + pd.Timedelta(days=1)
    out.loc[after, ["temp_c", "wind_kmh", "precip_mm"]] = HUGE
    return out


CUTOFFS = [START + pd.Timedelta(days=d) for d in (8, 30, 42, 44, 69)]  # 42, 44: mid-closure


@pytest.fixture(scope="module")
def data():
    return realistic_hourly()


@pytest.mark.parametrize("day", CUTOFFS, ids=lambda d: str(d.date()))
def test_features_for_day_d_ignore_day_d_and_later(data, day):
    unique_ts = pd.Series(data["ts"].unique())
    weather, cal = weather_for(unique_ts), calendar_features(unique_ts)
    clean = build_feature_table(data, weather, cal)
    dirty = build_feature_table(
        corrupt_demand_from(data, day), corrupt_weather_after(weather, day), cal
    )
    feature_cols = [c for c in clean.columns if c not in {*TARGETS, "in_service"}]
    in_day = (clean["ts"] >= day) & (clean["ts"] < day + pd.Timedelta(days=1))
    pd.testing.assert_frame_equal(
        clean.loc[in_day, feature_cols].reset_index(drop=True),
        dirty.loc[in_day, feature_cols].reset_index(drop=True),
    )


def test_leakage_check_catches_a_leaky_feature(data):
    """Control: the same corruption must expose a feature that peeks at the current hour."""
    day = CUTOFFS[1]

    def with_leak(df):
        f = demand_features(df)
        f["leaky"] = f.groupby("station_id")["departures"].shift(1)  # 1 hour back: inside D
        return f

    clean, dirty = with_leak(data), with_leak(corrupt_demand_from(data, day))
    in_day = (clean["ts"] >= day) & (clean["ts"] < day + pd.Timedelta(days=1))
    assert not clean.loc[in_day, "leaky"].equals(dirty.loc[in_day, "leaky"])


@pytest.mark.parametrize("model_cls", [SeasonalNaive, HistoricalAverage])
@pytest.mark.parametrize("day", CUTOFFS[1:], ids=lambda d: str(d.date()))
def test_baseline_forecast_for_day_d_ignores_day_d_and_later(data, model_cls, day):
    clean = model_cls().predict(visible_slice(data, day), day)
    dirty_data = corrupt_demand_from(data, day)
    dirty = model_cls().predict(visible_slice(dirty_data, day), day)
    pd.testing.assert_frame_equal(clean, dirty)
    assert (clean[TARGETS] < HUGE).all().all()


def test_in_service_flag_on_a_slice_ignores_the_future(data):
    """A closure still running at the cutoff is judged only on the hours seen so far."""
    day = START + pd.Timedelta(days=42)  # station C has been closed for 2 days
    clean = visible_slice(data, day)
    # Make the future look like the closure ended immediately, or never happened.
    reopened = data.copy()
    reopened.loc[(reopened["station_id"] == "C") & (reopened["ts"] >= day), TARGETS] = 5.0
    pd.testing.assert_series_equal(clean["in_service"], visible_slice(reopened, day)["in_service"])


def test_full_backtest_predictions_ignore_corrupted_test_weeks(data):
    """End to end: corrupt the test week itself; every prediction stays the same."""
    week = START + pd.Timedelta(days=42)
    models = [SeasonalNaive(), HistoricalAverage()]
    clean = run_backtest(data, models, weeks=[week])
    # Corrupt only the last day of the week: earlier days' forecasts must not move,
    # and that day's forecast must not move either (it is made at its own midnight).
    dirty = run_backtest(
        corrupt_demand_from(data, week + pd.Timedelta(days=6)), models, weeks=[week]
    )
    pd.testing.assert_series_equal(clean["yhat"], dirty["yhat"])
    # The actuals did change, so the test really corrupted something.
    assert not clean["y"].equals(dirty["y"])


def test_prophet_backtest_predictions_ignore_corrupted_test_days(data):
    from bikecast.models.prophet_model import ProphetModel

    ts = pd.Series(pd.date_range(START, periods=(DAYS + 7) * 24, freq="h"))
    weather, cal = weather_for(ts), calendar_features(ts)
    week = START + pd.Timedelta(days=49)

    def model():
        return ProphetModel(cal, weather, workers=1, uncertainty_samples=50, yearly=False)

    clean = run_backtest(data, [model()], weeks=[week])
    # Corrupt from the second test day on: the week's single fit uses only data before the week.
    dirty = run_backtest(corrupt_demand_from(data, week + pd.Timedelta(days=1)), [model()], [week])
    pd.testing.assert_series_equal(clean["yhat"], dirty["yhat"])
    assert not clean["y"].equals(dirty["y"])
