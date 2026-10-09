import pandas as pd
import pytest

from bikecast.evaluation.report import check_identical_rows, coverage_section


def preds(model, n=4, with_interval=False):
    df = pd.DataFrame(
        {
            "model": model,
            "test_week": pd.Timestamp("2025-01-13"),
            "station_id": "A",
            "ts": pd.date_range("2025-01-13", periods=n, freq="h"),
            "target": "departures",
            "y": [0.0, 2.0, 5.0, 1.0][:n],
            "yhat": 1.0,
            "season": "winter",
            "hour_band": "night 0-5",
        }
    )
    if with_interval:
        df["yhat_lower"], df["yhat_upper"] = 0.0, 2.0
    return df


def test_identical_rows_pass_and_fail():
    check_identical_rows(pd.concat([preds("seasonal_naive"), preds("prophet")]))
    with pytest.raises(ValueError, match="different rows"):
        check_identical_rows(pd.concat([preds("seasonal_naive"), preds("prophet", n=3)]))


def test_coverage_section_reports_only_models_with_intervals():
    lines = coverage_section(
        pd.concat([preds("seasonal_naive"), preds("prophet", with_interval=True)])
    )
    rows = [line for line in lines if line.startswith("| prophet | all")]
    assert rows == ["| prophet | all | 75.0% | 2.00 | 4 |"]  # 0, 2, 1 inside [0, 2]; 5 outside
    assert not any(line.startswith("| seasonal_naive") for line in lines)
