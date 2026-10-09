import numpy as np
import pandas as pd
import pytest

from bikecast.data.calendar import calendar_features
from bikecast.features.build import build_feature_table, demand_features

START = pd.Timestamp("2024-01-01")


def hourly(days=10):
    ts = pd.date_range(START, periods=days * 24, freq="h")
    return pd.DataFrame(
        {
            "station_id": "A",
            "ts": ts,
            "departures": np.arange(len(ts), dtype=float),  # value = hours since start
            "arrivals": 1.0,
            "in_service": True,
        }
    )


def test_lags_point_to_the_right_hours():
    f = demand_features(hourly()).set_index("ts")
    t = pd.Timestamp("2024-01-09 05:00")  # hour index 8*24 + 5 = 197
    assert f.loc[t, "departures_lag24"] == 197 - 24
    assert f.loc[t, "departures_lag48"] == 197 - 48
    assert f.loc[t, "departures_lag168"] == 197 - 168
    assert f.loc[t, "departures_same_hour_mean_7d"] == np.mean([197 - 24 * k for k in range(1, 8)])


def test_prev_day_mean_is_the_whole_previous_day():
    f = demand_features(hourly()).set_index("ts")
    day = f.loc["2024-01-09"]
    expected = np.arange(7 * 24, 8 * 24).mean()  # all of Jan 8
    assert (day["departures_prev_day_mean"] == expected).all()


def test_incomplete_grid_is_rejected():
    with pytest.raises(ValueError, match="complete"):
        demand_features(hourly().drop(index=5))


def test_feature_table_joins_calendar_and_clips_precip():
    h = hourly()
    weather = pd.DataFrame({"ts": h["ts"], "temp_c": 5.0, "wind_kmh": 10.0, "precip_mm": 0.0})
    weather.loc[3, "precip_mm"] = 119.5
    cal = calendar_features(h["ts"])
    f = build_feature_table(h, weather, cal)
    assert len(f) == len(h)
    assert f.loc[3, "precip_log"] == pytest.approx(np.log1p(30.0))
    assert f.loc[0, "is_holiday"]  # New Year's Day
