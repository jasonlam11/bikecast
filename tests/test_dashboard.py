from pathlib import Path

import pandas as pd
import pytest

from bikecast.dashboard import APP_DATA, comparison_table, load_snapshot, window_net_flow

APP = Path(__file__).parents[1] / "app" / "streamlit_app.py"
DAY = pd.Timestamp("2025-01-14")


def toy():
    rows = []
    for model, err in [("seasonal_naive", 2.0), ("historical_average", 1.0), ("torch_mlp", 0.5)]:
        for h in (7, 8, 9):
            for target, y in [("arrivals", 10.0), ("departures", 4.0)]:
                rows.append({"model": model, "station_id": "A", "ts": DAY + pd.Timedelta(hours=h),
                             "date": DAY, "hour": h, "target": target, "y": y,
                             "yhat": y + err, "test_week": DAY})  # fmt: skip
    return pd.DataFrame(rows)


def test_window_net_flow():
    out = window_net_flow(toy(), "torch_mlp", DAY, (7, 10))
    assert out.loc[0, "actual"] == 3 * (10 - 4)
    assert out.loc[0, "forecast"] == 3 * (10.5 - 4.5)


def test_window_net_flow_drops_incomplete_windows():
    t = toy()
    t = t[~((t["hour"] == 9) & (t["target"] == "arrivals"))]
    assert window_net_flow(t, "torch_mlp", DAY, (7, 10)).empty


def test_comparison_table_orders_models_and_scores_skill():
    t = comparison_table(toy())
    assert list(t.index) == ["seasonal_naive", "historical_average", "torch_mlp"]
    assert t.loc["torch_mlp", "skill"] == pytest.approx(0.75)
    assert t.loc["torch_mlp", "skill vs hist avg"] == pytest.approx(0.5)


@pytest.mark.skipif(not (APP_DATA / "predictions_stations.parquet").exists(), reason="no snapshot")
def test_snapshot_loads_and_has_all_models():
    snap = load_snapshot()
    assert set(snap["stations"]["model"]) == {
        "seasonal_naive", "historical_average", "prophet", "torch_mlp"
    }  # fmt: skip
    assert len(snap["stations_info"]) == 50


@pytest.mark.skipif(not (APP_DATA / "predictions_stations.parquet").exists(), reason="no snapshot")
def test_app_runs_headless():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value.startswith("BikeCast")
