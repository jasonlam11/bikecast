from datetime import date, datetime

import duckdb
import pandas as pd

from bikecast.data.stations import build_crosswalk, find_merges, haversine_m, neighbor_changes
from bikecast.data.trips import register_trips

# Offsets in degrees latitude: 0.0001 deg is about 11 m.
BASE_LAT, BASE_LNG = 42.36, -71.06


def station(sid, name, dlat, first, last, n=1000):
    return {
        "station_id": sid,
        "name": name,
        "n_names": 1,
        "lat": BASE_LAT + dlat,
        "lng": BASE_LNG,
        "first_seen": first,
        "last_seen": last,
        "n_trip_ends": n,
    }


def make_stats():
    return pd.DataFrame(
        [
            # Replacement: old ID ends the day the new one starts, same spot.
            station(
                "OLD1", "Damrell St at Old Colony Ave", 0.0, date(2024, 1, 1), date(2024, 9, 18)
            ),
            station("NEW1", "Damrell St at Old Colony", 0.0, date(2024, 9, 18), date(2026, 9, 30)),
            # Ghost: one-day ID on top of a busy station.
            station("BIG2", "Granby St", 0.01, date(2024, 1, 1), date(2026, 9, 30), n=90000),
            station("GHO2", "Granby St", 0.01, date(2025, 2, 25), date(2025, 2, 25), n=16),
            # Neighbors: 16 m apart, both active for years. Must stay separate.
            station("NB3A", "Lafayette Square", 0.02, date(2024, 1, 1), date(2026, 9, 30)),
            station(
                "NB3B", "Mass Ave/Lafayette Square", 0.02015, date(2024, 1, 1), date(2026, 9, 30)
            ),
            # Same name 145 m apart, no overlap: merged by the same-name rule.
            station("FL4A", "Flint St at Bridge St", 0.03, date(2024, 1, 6), date(2025, 11, 12)),
            station(
                "FL4B", "Flint St. at Bridge St.", 0.0313, date(2025, 11, 17), date(2026, 9, 30)
            ),
            # Same name but far away: separate.
            station("FAR5", "Main St", 0.04, date(2024, 1, 1), date(2024, 6, 1)),
            station("FAR6", "Main St", 0.05, date(2024, 7, 1), date(2026, 9, 30)),
        ]
    )


def test_haversine_scale():
    assert abs(haversine_m(42.36, -71.06, 42.3601, -71.06) - 11.1) < 0.2


def test_find_merges_applies_each_rule():
    merges = find_merges(make_stats())
    got = {(r.id_a, r.id_b): r.reason for r in merges.itertuples()}
    assert got == {
        ("OLD1", "NEW1"): "replacement",
        ("BIG2", "GHO2"): "ghost",
        ("FL4A", "FL4B"): "replacement",
    }


def test_crosswalk_points_to_most_recent_station():
    stats = make_stats()
    cw = build_crosswalk(stats, find_merges(stats)).set_index("station_id")["canonical_id"]
    assert cw["OLD1"] == cw["NEW1"] == "NEW1"
    assert cw["GHO2"] == cw["BIG2"] == "BIG2"
    assert cw["FL4A"] == "FL4B"
    assert cw["NB3A"] == "NB3A" and cw["NB3B"] == "NB3B"
    assert cw["FAR5"] == "FAR5"
    assert len(cw) == len(stats)


def test_register_trips_dedupes_repeats_but_keeps_reused_ids(tmp_path):
    t = datetime(2024, 1, 31, 23, 50)
    jan = pd.DataFrame({"ride_id": ["A", "B"], "started_at": [t, t]})
    # A repeats with the same start time (true duplicate). B is reused for a different trip.
    feb = pd.DataFrame({"ride_id": ["A", "B"], "started_at": [t, datetime(2024, 2, 5, 9, 0)]})
    jan.to_parquet(tmp_path / "trips_202401.parquet")
    feb.to_parquet(tmp_path / "trips_202402.parquet")
    con = duckdb.connect()
    register_trips(con, str(tmp_path / "trips_*.parquet"))
    assert con.sql("SELECT count(*) FROM trips").fetchone()[0] == 3


def test_neighbor_changes_flags_new_dock_next_to_top_station():
    stats = make_stats()
    late = station("NB3C", "Lafayette Annex", 0.0201, date(2025, 6, 1), date(2026, 9, 30))
    stats = pd.concat([stats, pd.DataFrame([late])], ignore_index=True)
    out = neighbor_changes(stats, {"NB3A"})
    # NB3B ran the whole time, so only the newly opened NB3C is flagged.
    assert out["neighbor"].tolist() == ["NB3C"]
