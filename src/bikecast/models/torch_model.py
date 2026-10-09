"""One global MLP across all stations, with a learned station embedding.

One sample is one station-day. Inputs:
- for each of the 24 hours of day D: demand lags at 24, 48, and 168 hours and the same-hour mean of
  the previous 7 days, for both targets (log1p), plus forecast weather (standardized)
- day-level: previous-day mean for both targets (log1p), weekday one-hot, month as sin/cos, holiday,
  semester, and the share of hours a station was closed on D-1 and D-7
- a learned embedding of the station ID
Output: 48 values, departures and arrivals for each hour of D.

Training: all station-days before the validation window (the 14 days before the test week).
Early stopping on validation loss. Two losses are trained, Poisson NLL and L1, and the one with
lower validation MAE is kept; the test week is never used for any choice. Closed hours get weight 0
in the loss. Seeds are fixed and algorithms deterministic, so a rerun gives identical forecasts.
"""

import logging
import random

import numpy as np
import pandas as pd
import torch
from torch import nn

from bikecast.features.build import build_feature_table
from bikecast.models.baseline import TARGETS

log = logging.getLogger(__name__)

HOURLY_DEMAND = [f"{t}_{f}" for t in TARGETS for f in ["lag24", "lag48", "lag168",
                                                        "same_hour_mean_7d"]]  # fmt: skip
HOURLY_WEATHER = ["temp_c", "wind_kmh", "precip_log"]
DAY_DEMAND = [f"{t}_prev_day_mean" for t in TARGETS]

HPARAMS = {
    "embedding_dim": 8,
    "hidden": [128, 64],
    "dropout": 0.1,
    "lr": 1e-3,
    "batch_size": 256,
    "max_epochs": 200,
    "patience": 10,
    "val_days": 14,
    "losses": ["poisson", "l1"],
    "seed": 42,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def closed_share(table: pd.DataFrame) -> pd.DataFrame:
    """Share of out-of-service hours on D-1 and D-7, per station and date."""
    daily = (~table["in_service"]).groupby([table["station_id"], table["date"]]).mean()
    by_station = daily.groupby(level="station_id")
    return pd.DataFrame({"closed_d1": by_station.shift(1), "closed_d7": by_station.shift(7)})


class Encoder:
    """Turns the hourly feature table into model arrays. Fit scaling on training days only."""

    def __init__(self, stations: list[str]):
        self.station_index = {s: i for i, s in enumerate(sorted(stations))}
        self.weather_mean = self.weather_std = None

    def fit(self, table: pd.DataFrame) -> "Encoder":
        w = table[HOURLY_WEATHER]
        self.weather_mean, self.weather_std = w.mean(), w.std().replace(0, 1)
        return self

    def encode(self, table: pd.DataFrame):
        """Arrays for every complete station-day in `table`, sorted by station then date."""
        t = table.sort_values(["station_id", "ts"])
        t = t[t["station_id"].isin(self.station_index)]
        day = t.groupby(["station_id", "date"], sort=True).first()
        if len(t) != 24 * len(day):
            raise ValueError("Every station-day must have exactly 24 hourly rows")
        hourly = np.concatenate(
            [
                np.log1p(t[HOURLY_DEMAND].to_numpy(dtype=float)),
                ((t[HOURLY_WEATHER] - self.weather_mean) / self.weather_std).to_numpy(float),
            ],
            axis=1,
        ).reshape(len(day), 24, -1)
        dow = np.eye(7)[day["dow"].to_numpy()]
        month = day["month"].to_numpy()
        day_feats = np.column_stack(
            [
                np.log1p(day[DAY_DEMAND].to_numpy(dtype=float)),
                dow,
                np.sin(2 * np.pi * month / 12),
                np.cos(2 * np.pi * month / 12),
                day["is_holiday"].to_numpy(float),
                day["is_semester"].to_numpy(float),
                day[["closed_d1", "closed_d7"]].to_numpy(float),
            ]
        )
        x = np.concatenate([hourly.reshape(len(day), -1), day_feats], axis=1)
        y = t[TARGETS].to_numpy(dtype=float).reshape(len(day), 24, 2).transpose(0, 2, 1)
        mask = t["in_service"].to_numpy(dtype=float).reshape(len(day), 24)
        mask = np.repeat(mask[:, None, :], 2, axis=1)
        keys = day.index.to_frame(index=False)
        station = keys["station_id"].map(self.station_index).to_numpy()
        complete = ~np.isnan(x).any(axis=1)
        return {
            "x": x[complete].astype(np.float32),
            "station": station[complete],
            "y": y.reshape(len(day), 48)[complete].astype(np.float32),
            "mask": mask.reshape(len(day), 48)[complete].astype(np.float32),
            "keys": keys[complete].reset_index(drop=True),
        }


class Net(nn.Module):
    def __init__(self, n_stations: int, n_inputs: int, hp: dict, loss: str):
        super().__init__()
        self.loss = loss
        self.embedding = nn.Embedding(n_stations, hp["embedding_dim"])
        layers, width = [], n_inputs + hp["embedding_dim"]
        for h in hp["hidden"]:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(hp["dropout"])]
            width = h
        layers.append(nn.Linear(width, 48))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, station):
        """Raw output: log-rate for Poisson, pre-softplus value for L1."""
        return self.mlp(torch.cat([x, self.embedding(station)], dim=1))

    def predict_counts(self, out):
        return torch.exp(out) if self.loss == "poisson" else nn.functional.softplus(out)

    def loss_fn(self, out, y, mask):
        if self.loss == "poisson":
            per = nn.functional.poisson_nll_loss(out, y, log_input=True, reduction="none")
        else:
            per = (self.predict_counts(out) - y).abs()
        return (per * mask).sum() / mask.sum().clamp(min=1)


def _tensors(arrs):
    return (
        torch.from_numpy(arrs["x"]),
        torch.from_numpy(arrs["station"]).long(),
        torch.from_numpy(arrs["y"]),
        torch.from_numpy(arrs["mask"]),
    )


def train_net(train, val, n_stations, hp, loss):
    set_seed(hp["seed"])
    net = Net(n_stations, train["x"].shape[1], hp, loss)
    opt = torch.optim.Adam(net.parameters(), lr=hp["lr"])
    xt, st, yt, mt = _tensors(train)
    xv, sv, yv, mv = _tensors(val)
    gen = torch.Generator().manual_seed(hp["seed"])
    best = {"val_loss": float("inf"), "epoch": 0, "state": None}
    curve = []
    for epoch in range(1, hp["max_epochs"] + 1):
        net.train()
        for idx in torch.randperm(len(xt), generator=gen).split(hp["batch_size"]):
            opt.zero_grad()
            net.loss_fn(net(xt[idx], st[idx]), yt[idx], mt[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            out = net(xv, sv)
            val_loss = net.loss_fn(out, yv, mv).item()
        curve.append(round(val_loss, 5))
        if val_loss < best["val_loss"] - 1e-6:
            best = {"val_loss": val_loss, "epoch": epoch,
                    "state": {k: v.clone() for k, v in net.state_dict().items()}}  # fmt: skip
        elif epoch - best["epoch"] >= hp["patience"]:
            break
    net.load_state_dict(best["state"])
    net.eval()
    with torch.no_grad():
        pred = net.predict_counts(net(xv, sv))
        val_mae = ((pred - yv).abs() * mv).sum().item() / mv.sum().item()
    return net, {"loss": loss, "best_epoch": best["epoch"], "epochs_run": epoch,
                 "val_loss": best["val_loss"], "val_mae": val_mae, "val_curve": curve}  # fmt: skip


class TorchMLP:
    name = "torch_mlp"

    def __init__(self, calendar: pd.DataFrame, weather: pd.DataFrame, hparams: dict | None = None):
        self.calendar, self.weather = calendar, weather
        self.hp = {**HPARAMS, **(hparams or {})}
        self.training_log: list[dict] = []
        self.net = self.encoder = None

    def _table(self, hourly: pd.DataFrame) -> pd.DataFrame:
        t = build_feature_table(hourly, self.weather, self.calendar)
        t["date"] = t["ts"].dt.normalize()
        # Computed on the whole table so D-1 and D-7 are visible before any filtering to day D.
        return t.join(closed_share(t), on=["station_id", "date"])

    def fit(self, train: pd.DataFrame) -> "TorchMLP":
        cutoff = train["ts"].max() + pd.Timedelta(hours=1)
        val_start = cutoff - pd.Timedelta(days=self.hp["val_days"])
        table = self._table(train)
        self.encoder = Encoder(train["station_id"].unique().tolist())
        self.encoder.fit(table[table["ts"] < val_start])
        arrs = self.encoder.encode(table)
        is_val = (arrs["keys"]["date"] >= val_start).to_numpy()
        split = {
            name: {
                k: (v[sel] if k != "keys" else v[sel].reset_index(drop=True))
                for k, v in arrs.items()
            }  # fmt: skip
            for name, sel in [("train", ~is_val), ("val", is_val)]
        }
        runs = [
            train_net(split["train"], split["val"], len(self.encoder.station_index), self.hp, loss)
            for loss in self.hp["losses"]
        ]
        self.net, chosen = min(runs, key=lambda r: r[1]["val_mae"])
        entry = {
            "cutoff": str(cutoff),
            "train_samples": int((~is_val).sum()),
            "val_samples": int(is_val.sum()),
            "chosen_loss": chosen["loss"],
            "runs": [r[1] for r in runs],
            "hparams": self.hp,
        }
        self.training_log.append(entry)
        log.info(
            "torch_mlp: cutoff %s, %d train / %d val samples, chose %s (val MAE %.3f, epoch %d)",
            cutoff.date(),
            entry["train_samples"],
            entry["val_samples"],
            chosen["loss"],
            chosen["val_mae"],
            chosen["best_epoch"],
        )
        return self

    def predict(self, history: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
        if self.net is None:
            raise RuntimeError("fit() must run before predict()")
        # Only the last 8 days are needed for lags; day D's targets are unknown, so they are NaN.
        recent = history[history["ts"] >= day - pd.Timedelta(days=8)]
        stations = sorted(recent["station_id"].unique())
        hours = pd.date_range(day, periods=24, freq="h")
        blank = pd.MultiIndex.from_product([stations, hours], names=["station_id", "ts"])
        blank = blank.to_frame(index=False).assign(in_service=True)
        table = self._table(pd.concat([recent, blank], ignore_index=True))
        arrs = self.encoder.encode(table[table["date"] == day].assign(departures=0.0, arrivals=0.0))
        if len(arrs["keys"]) != len(stations):
            raise ValueError(f"torch_mlp: incomplete inputs for {day.date()}")
        with torch.no_grad():
            x, s = torch.from_numpy(arrs["x"]), torch.from_numpy(arrs["station"]).long()
            counts = self.net.predict_counts(self.net(x, s)).numpy().reshape(-1, 2, 24)
        rows = []
        for i, station in enumerate(arrs["keys"]["station_id"]):
            rows.append(
                pd.DataFrame(
                    {
                        "station_id": station,
                        "ts": hours,
                        "departures": counts[i, 0],
                        "arrivals": counts[i, 1],
                    }
                )
            )
        return pd.concat(rows, ignore_index=True)
