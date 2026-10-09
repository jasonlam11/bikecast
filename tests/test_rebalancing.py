import pandas as pd
import pytest

from bikecast.evaluation.rebalancing import classify, summarize

DAYS = pd.date_range("2025-01-13", periods=25, freq="D")


def flows(station, forecast, actual):
    return pd.DataFrame(
        {"station_id": station, "date": DAYS, "forecast": forecast, "actual": actual}
    )


@pytest.fixture
def data():
    return pd.concat(
        [
            flows("DRAIN", -10.0, [-8.0] * 20 + [2.0] * 5),  # actually drains on 20 of 25
            flows("FILL", 12.0, 14.0),
            # Drains below -5 on 19 of 25 mornings (76%): just under the 80% bar.
            flows("MIXED", [-6.0] * 19 + [1.0] * 6, -6.0),
            flows("SHORT", -10.0, -10.0).head(10),  # open too few days
        ],
        ignore_index=True,
    )


def test_classify_labels(data):
    t = classify(data)
    assert t.loc["DRAIN", "label"] == "drains"
    assert t.loc["FILL", "label"] == "fills"
    assert t.loc["MIXED", "label"] == "mixed"
    assert t.loc["SHORT", "label"] == "too few days"


def test_summary_totals_and_hit_rate(data):
    t = classify(data)
    s = summarize(data, t)
    assert s["drains"]["stations"] == 1
    assert s["drains"]["bikes_forecast"] == 10.0
    assert s["drains"]["bikes_actual"] == 8.0  # median actual
    assert s["drains"]["hit_rate"] == pytest.approx(20 / 25)
    assert s["fills"]["bikes_forecast"] == 12.0
    assert s["fills"]["hit_rate"] == 1.0
