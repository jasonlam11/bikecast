from datetime import datetime

import duckdb
import pandas as pd
import pytest

from bikecast.data.aggregate import aggregate_hourly, aggregate_system, data_range, flag_in_service
from bikecast.data.trips import register_trips

START = datetime(2024, 1, 1)
END = datetime(2024, 1, 2)  # one day: 24 hours


def trip(rid, start, end, s_from, s_to):
    return {
        "ride_id": rid,
        "started_at": pd.Timestamp(start),
        "ended_at": pd.Timestamp(end),
        "start_station_id": s_from,
        "end_station_id": s_to,
    }


@pytest.fixture
def con(tmp_path):
    trips = pd.DataFrame(
        [
            trip("t1", "2024-01-01 08:05", "2024-01-01 08:20", "A", "B"),
            trip("t2", "2024-01-01 08:40", "2024-01-01 09:10", "A", "B"),
            # Starts at the old ID of station B (merged), ends at C (not a top station).
            trip("t3", "2024-01-01 17:30", "2024-01-01 17:45", "B_OLD", "C"),
            # Missing end station: still a departure from A.
            trip("t4", "2024-01-01 12:00", "2024-01-01 12:30", "A", None),
            # Ends after the range: departure counts, arrival is outside the grid.
            trip("t5", "2024-01-01 23:50", "2024-01-02 00:10", "B", "A"),
            # Entirely before the range.
            trip("t6", "2023-12-31 22:00", "2023-12-31 22:30", "A", "B"),
        ]
    )
    trips.to_parquet(tmp_path / "trips_202401.parquet")
    c = duckdb.connect()
    register_trips(c, str(tmp_path / "trips_*.parquet"))
    c.register(
        "crosswalk",
        pd.DataFrame(
            {"station_id": ["A", "B", "B_OLD", "C"], "canonical_id": ["A", "B", "B", "C"]}
        ),
    )
    return c


@pytest.fixture
def hourly(con):
    return aggregate_hourly(con, ["A", "B"], START, END)


def test_zero_fill_gives_every_station_every_hour(hourly):
    assert len(hourly) == 2 * 24
    assert hourly.groupby("station_id")["ts"].nunique().tolist() == [24, 24]
    assert hourly[["departures", "arrivals", "net_flow"]].notna().all().all()


def test_empty_hours_are_zero_not_missing(hourly):
    a = hourly[hourly["station_id"] == "A"].set_index("ts")
    assert a.loc[pd.Timestamp("2024-01-01 03:00"), "departures"] == 0
    assert a.loc[pd.Timestamp("2024-01-01 03:00"), "arrivals"] == 0
    # A has departures only in hours 8 (t1, t2) and 12 (t4).
    assert (a["departures"] > 0).sum() == 2


def test_counts_land_in_the_right_hour(hourly):
    h = hourly.set_index(["station_id", "ts"])
    assert h.loc[("A", pd.Timestamp("2024-01-01 08:00")), "departures"] == 2
    # t2 left A at 8:40 but arrived at B at 9:10, so the arrival lands in hour 9.
    assert h.loc[("B", pd.Timestamp("2024-01-01 08:00")), "arrivals"] == 1
    assert h.loc[("B", pd.Timestamp("2024-01-01 09:00")), "arrivals"] == 1


def test_merged_ids_count_under_canonical_station(hourly):
    h = hourly.set_index(["station_id", "ts"])
    assert h.loc[("B", pd.Timestamp("2024-01-01 17:00")), "departures"] == 1


def test_totals_match_trips_inside_range(hourly):
    # Departures from A or B in range: t1, t2, t4 (A); t3, t5 (B). t6 is before the range.
    assert hourly["departures"].sum() == 5
    # Arrivals at A or B in range: t1, t2 (B). t5 ends after the range, t3 ends at C.
    assert hourly["arrivals"].sum() == 2
    assert (hourly["net_flow"] == hourly["arrivals"] - hourly["departures"]).all()


def test_system_totals_are_zero_filled(con):
    system = aggregate_system(con, START, END)
    assert len(system) == 24
    assert system["departures"].sum() == 5
    # t4 has no end station; t5 ends after the range. t1, t2, t3 arrive inside it.
    assert system["arrivals"].sum() == 3


def test_data_range_covers_whole_months(tmp_path):
    for m in ["202401", "202402", "202609"]:
        (tmp_path / f"trips_{m}.parquet").touch()
    assert data_range(tmp_path) == (datetime(2024, 1, 1), datetime(2026, 10, 1))


def test_flag_in_service_masks_long_closures_only():
    ts = pd.date_range("2024-01-01", periods=100, freq="h")
    busy = [1] * 100
    # Station X: quiet for 10 hours (overnight), then closed for 50 hours.
    x = [1] * 20 + [0] * 10 + [1] * 10 + [0] * 50 + [1] * 10
    df = pd.DataFrame(
        {
            "station_id": ["X"] * 100 + ["Y"] * 100,
            "ts": list(ts) * 2,
            "departures": x + busy,
            "arrivals": x + busy,
        }
    ).sample(frac=1, random_state=0)  # order should not matter
    flag = flag_in_service(df, min_hours=48)
    x_flag = flag[df["station_id"] == "X"].loc[sorted(df.index[df["station_id"] == "X"])]
    assert x_flag.iloc[20:30].all()  # 10 quiet hours stay in service
    assert not x_flag.iloc[40:90].any()  # the 50-hour closure is masked
    assert flag[df["station_id"] == "Y"].all()


def test_flag_in_service_does_not_join_runs_across_stations():
    ts = pd.date_range("2024-01-01", periods=30, freq="h")
    # Each station is idle for 30 hours: below the threshold alone, above it if joined.
    df = pd.DataFrame(
        {"station_id": ["A"] * 30 + ["B"] * 30, "ts": list(ts) * 2, "departures": 0, "arrivals": 0}
    )
    assert flag_in_service(df, min_hours=48).all()
