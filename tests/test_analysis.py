import pandas as pd
import pytest

from bikecast.evaluation.analysis import (
    bias_by_hour,
    coverage_by_hour,
    morning_net_flow,
    station_scores,
)

MON = pd.Timestamp("2025-01-13")


def rows(model, station, day, hour, target, y, yhat, holiday=False, **extra):
    ts = day + pd.Timedelta(hours=hour)
    return {"model": model, "station_id": station, "ts": ts, "date": day, "hour": hour,
            "target": target, "y": y, "yhat": yhat, "is_holiday": holiday, **extra}  # fmt: skip


def test_morning_net_flow_sums_arrivals_minus_departures():
    data = []
    for h in (7, 8, 9):
        data.append(rows("m", "A", MON, h, "arrivals", 1, 2))
        data.append(rows("m", "A", MON, h, "departures", 5, 6))
    out = morning_net_flow(pd.DataFrame(data), "m")
    assert out.loc[0, "forecast"] == 3 * (2 - 6)
    assert out.loc[0, "actual"] == 3 * (1 - 5)


def test_morning_net_flow_skips_weekends_holidays_and_partial_mornings():
    data = []
    sat, tue, wed = (
        MON + pd.Timedelta(days=5),
        MON + pd.Timedelta(days=1),
        MON + pd.Timedelta(days=2),
    )
    for h in (7, 8, 9):
        for t in ("arrivals", "departures"):
            data.append(rows("m", "A", sat, h, t, 1, 1))
            data.append(rows("m", "A", tue, h, t, 1, 1, holiday=True))
            if h != 9:  # Wednesday is missing an hour (closed)
                data.append(rows("m", "A", wed, h, t, 1, 1))
    assert morning_net_flow(pd.DataFrame(data), "m").empty


def test_station_scores_skill_vs_historical_average():
    data = [
        rows("historical_average", "A", MON, 8, "arrivals", 4, 2),  # error 2
        rows("torch_mlp", "A", MON, 8, "arrivals", 4, 3),  # error 1
    ]
    stations = pd.DataFrame({"canonical_id": ["A"], "name": ["Station A"], "rank_volume": [10]})
    out = station_scores(pd.DataFrame(data), stations)
    assert out.loc["A", "skill_torch_mlp"] == pytest.approx(0.5)
    assert out.loc["A", "name"] == "Station A"


def test_bias_and_coverage_by_hour():
    data = pd.DataFrame(
        [
            rows("p", "A", MON, 8, "arrivals", 4, 6, yhat_lower=3.0, yhat_upper=5.0),
            rows("p", "B", MON, 8, "arrivals", 2, 2, yhat_lower=1.0, yhat_upper=3.0),
        ]
    )
    assert bias_by_hour(data).loc[8, "p"] == pytest.approx(1.0)
    assert coverage_by_hour(data).loc[8] == pytest.approx(1.0)
