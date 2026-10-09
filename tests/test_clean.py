from pathlib import Path

import pandas as pd
import pytest

from bikecast.data.clean import UNIFIED_COLUMNS, apply_rules, clean_csv, detect_schema, normalize

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(FIXTURES / name, dtype=str, keep_default_na=False)


@pytest.mark.parametrize(
    ("name", "schema"), [("new_sample.csv", "new"), ("legacy_sample.csv", "legacy")]
)
def test_detect_schema(name, schema):
    assert detect_schema(list(load(name).columns)) == schema


def test_detect_schema_rejects_unknown():
    with pytest.raises(ValueError):
        detect_schema(["foo", "bar"])


def test_both_schemas_normalize_to_identical_columns_and_dtypes():
    new, legacy = normalize(load("new_sample.csv")), normalize(load("legacy_sample.csv"))
    assert list(new.columns) == list(legacy.columns) == UNIFIED_COLUMNS
    assert new.dtypes.to_dict() == legacy.dtypes.to_dict()


def test_normalize_parses_both_timestamp_precisions():
    df = normalize(load("new_sample.csv"))
    assert df.loc[0, "started_at"] == pd.Timestamp("2024-01-31 12:16:49")
    assert df.loc[1, "started_at"] == pd.Timestamp("2025-11-16 18:20:41.509")


def test_legacy_user_types_map_to_member_casual():
    df = normalize(load("legacy_sample.csv"))
    assert df["member_type"].tolist()[:2] == ["member", "casual"]
    assert df["start_station_id"].iloc[0] == "386"


def test_whitespace_and_empty_strings_become_clean_values():
    df = normalize(load("new_sample.csv"))
    row = df[df["ride_id"] == "R10"].iloc[0]
    assert row["start_station_id"] == "M32037"
    assert pd.isna(row["end_station_id"])


def test_new_schema_rules_remove_exactly_the_bad_rows():
    kept, removed = apply_rules(normalize(load("new_sample.csv")))
    assert removed == {
        "duplicate": 1,
        "unparseable_time": 1,
        "duration_nonpositive": 1,
        "duration_under_60s": 1,
        "duration_over_24h": 1,
        "test_station": 2,
        "no_station": 1,
    }
    # R10 has no end station but still counts as a departure, so it is kept.
    assert sorted(kept["ride_id"]) == ["R01", "R02", "R10"]


def test_legacy_rules_dedupe_on_full_row():
    kept, removed = apply_rules(normalize(load("legacy_sample.csv")))
    assert removed["duplicate"] == 1
    assert removed["duration_under_60s"] == 1
    assert removed["test_station"] == 1
    assert len(kept) == 2


def test_test_station_pattern_does_not_hit_real_names():
    df = normalize(load("new_sample.csv"))
    df.loc[:, "start_station_name"] = "Main Street/Albany Street/Technology Square"
    df.loc[:, "end_station_name"] = "515 Somerville Ave (Temp. Winter Location)"
    _, removed = apply_rules(df)
    assert removed["test_station"] == 0


def test_clean_csv_writes_parquet_and_counts_add_up(tmp_path):
    out = tmp_path / "trips.parquet"
    counts = clean_csv(FIXTURES / "new_sample.csv", out)
    removed = sum(v for k, v in counts.items() if k not in {"rows_in", "rows_out"})
    assert counts["rows_in"] - removed == counts["rows_out"] == len(pd.read_parquet(out))
    assert not out.with_suffix(".tmp").exists()
