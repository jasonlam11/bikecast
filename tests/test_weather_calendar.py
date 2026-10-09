from datetime import datetime

import pandas as pd
import pytest

from bikecast.data.calendar import calendar_features
from bikecast.data.weather import to_local_grid


def utc_payload(start_utc: str, hours: int) -> dict:
    """A fake Open-Meteo payload whose temperature equals the UTC hour index."""
    times = pd.date_range(start_utc, periods=hours, freq="h")
    return {
        "hourly": {
            "time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
            "temperature_2m": [float(i) for i in range(hours)],
            "precipitation": [0.0] * hours,
            "wind_speed_10m": [10.0] * hours,
        }
    }


def test_regular_day_has_24_hours_shifted_by_offset():
    # 2024-01-15 local midnight is 05:00 UTC (EST, UTC-5).
    payload = utc_payload("2024-01-15 00:00", 48)
    df = to_local_grid(payload, datetime(2024, 1, 15), datetime(2024, 1, 16))
    assert len(df) == 24
    assert df.loc[0, "temp_c"] == 5.0  # UTC index 5


def test_spring_forward_fills_missing_2am():
    # 2024-03-10: local 2am does not exist. UTC 07:00 is local 3am EDT.
    payload = utc_payload("2024-03-10 00:00", 48)
    df = to_local_grid(payload, datetime(2024, 3, 10), datetime(2024, 3, 11)).set_index("ts")
    assert len(df) == 24
    assert df.loc[pd.Timestamp("2024-03-10 01:00"), "temp_c"] == 6.0  # 06:00 UTC
    assert df.loc[pd.Timestamp("2024-03-10 03:00"), "temp_c"] == 7.0  # 07:00 UTC
    assert df.loc[pd.Timestamp("2024-03-10 02:00"), "temp_c"] == 6.5  # interpolated


def test_fall_back_averages_repeated_1am():
    # 2024-11-03: local 1am happens at 05:00 UTC (EDT) and again at 06:00 UTC (EST).
    payload = utc_payload("2024-11-03 00:00", 48)
    df = to_local_grid(payload, datetime(2024, 11, 3), datetime(2024, 11, 4)).set_index("ts")
    assert len(df) == 24
    assert df.loc[pd.Timestamp("2024-11-03 01:00"), "temp_c"] == 5.5


def test_gap_in_source_data_raises():
    payload = utc_payload("2024-01-15 00:00", 48)
    payload["hourly"]["temperature_2m"][10:20] = [None] * 10
    with pytest.raises(ValueError, match="gaps"):
        to_local_grid(payload, datetime(2024, 1, 15), datetime(2024, 1, 16))


@pytest.fixture
def cal():
    ts = pd.Series(pd.date_range("2025-01-01", "2026-01-01", freq="h", inclusive="left"))
    return calendar_features(ts).set_index("ts")


def test_calendar_basic_fields(cal):
    row = cal.loc[pd.Timestamp("2025-07-04 17:00")]
    assert row["hour"] == 17
    assert row["dow"] == 4  # Friday
    assert row["month"] == 7
    assert not row["is_weekend"]
    assert row["is_holiday"]
    assert row["holiday_name"] == "Independence Day"


def test_patriots_day_is_a_massachusetts_holiday(cal):
    assert cal.loc[pd.Timestamp("2025-04-21 09:00"), "is_holiday"]
    assert not cal.loc[pd.Timestamp("2025-04-22 09:00"), "is_holiday"]


def test_weekend_and_semester_flags(cal):
    assert cal.loc[pd.Timestamp("2025-01-11 12:00"), "is_weekend"]  # Saturday
    assert cal.loc[pd.Timestamp("2025-10-01 12:00"), "is_semester"]
    assert not cal.loc[pd.Timestamp("2025-07-15 12:00"), "is_semester"]


def test_every_day_has_24_rows(cal):
    assert (cal.groupby(cal.index.date).size() == 24).all()
