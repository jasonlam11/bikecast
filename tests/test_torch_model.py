import numpy as np
import pandas as pd
import pytest
import torch

from bikecast.data.calendar import calendar_features
from bikecast.evaluation.backtest import run_backtest, visible_slice
from bikecast.models.torch_model import HPARAMS, Net, TorchMLP

START = pd.Timestamp("2024-02-05")
DAYS = 60
FAST = {"hidden": [16], "max_epochs": 6, "patience": 2, "batch_size": 32}


def data(seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range(START, periods=DAYS * 24, freq="h")
    h = ts.hour.to_numpy()
    frames = []
    for i, s in enumerate(["A", "B", "C"]):
        mean = (1 + i) * (1 + 5 * np.exp(-((h - 8) ** 2) / 3)) * np.where(ts.dayofweek < 5, 1, 0.5)
        frames.append(
            pd.DataFrame(
                {
                    "station_id": s,
                    "ts": ts,
                    "departures": rng.poisson(mean).astype(float),
                    "arrivals": rng.poisson(mean[::-1]).astype(float),
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    df["in_service"] = True
    return df


@pytest.fixture(scope="module")
def inputs():
    ts = pd.Series(pd.date_range(START, periods=(DAYS + 7) * 24, freq="h"))
    rng = np.random.default_rng(1)
    weather = pd.DataFrame(
        {
            "ts": ts,
            "temp_c": rng.normal(10, 5, len(ts)),
            "wind_kmh": rng.uniform(0, 20, len(ts)),
            "precip_mm": rng.exponential(0.2, len(ts)),
        }
    )
    return data(), calendar_features(ts), weather


def fitted(inputs, cutoff_day=45):
    d, cal, weather = inputs
    model = TorchMLP(cal, weather, FAST)
    model.fit(visible_slice(d, START + pd.Timedelta(days=cutoff_day)))
    return model


def test_predict_shape_and_non_negative(inputs):
    model = fitted(inputs)
    day = START + pd.Timedelta(days=45)
    pred = model.predict(visible_slice(inputs[0], day), day)
    assert len(pred) == 3 * 24
    assert set(pred["station_id"]) == {"A", "B", "C"}
    assert (pred[["departures", "arrivals"]] >= 0).all().all()
    assert pred["ts"].min() == day and pred["ts"].max() == day + pd.Timedelta(hours=23)


def test_same_seed_gives_identical_forecasts(inputs):
    day = START + pd.Timedelta(days=45)
    a = fitted(inputs).predict(visible_slice(inputs[0], day), day)
    b = fitted(inputs).predict(visible_slice(inputs[0], day), day)
    pd.testing.assert_frame_equal(a, b)


def test_validation_window_is_the_14_days_before_the_cutoff(inputs):
    entry = fitted(inputs).training_log[-1]
    assert entry["val_samples"] == 3 * HPARAMS["val_days"]
    # Lags need 7 days of history (lag 168), so days 0 to 6 produce no training samples.
    assert entry["train_samples"] == 3 * (45 - HPARAMS["val_days"] - 7)
    assert entry["chosen_loss"] in {"poisson", "l1"}
    assert {r["loss"] for r in entry["runs"]} == {"poisson", "l1"}


@pytest.mark.parametrize("loss", ["poisson", "l1"])
def test_loss_ignores_masked_hours(loss):
    torch.manual_seed(0)
    net = Net(n_stations=2, n_inputs=5, hp={**HPARAMS, **FAST}, loss=loss)
    out, y = torch.randn(4, 48), torch.rand(4, 48) * 5
    mask = torch.ones(4, 48)
    mask[:, :10] = 0
    y_changed = y.clone()
    y_changed[:, :10] = 1000.0  # only masked entries change
    assert net.loss_fn(out, y, mask).item() == pytest.approx(
        net.loss_fn(out, y_changed, mask).item()
    )


def test_torch_backtest_ignores_corrupted_future(inputs):
    d, cal, weather = inputs
    week = START + pd.Timedelta(days=49)
    clean = run_backtest(d, [TorchMLP(cal, weather, FAST)], weeks=[week])
    dirty_data = d.copy()
    dirty_data.loc[dirty_data["ts"] >= week + pd.Timedelta(days=6), ["departures", "arrivals"]] = (
        1e6
    )
    dirty = run_backtest(dirty_data, [TorchMLP(cal, weather, FAST)], weeks=[week])
    pd.testing.assert_series_equal(clean["yhat"], dirty["yhat"])
    assert not clean["y"].equals(dirty["y"])
