"""Rolling-origin backtest.

For each test week: fit on data before the week starts. Then for each day D in the week, at
midnight, predict all 24 hours of D using only data before D, and keep the actuals for scoring.
Models never receive rows at or after the time they are forecasting from.

`in_service` is recomputed on every slice a model sees, so a closure that is still going on at the
cut is judged only on what was known then. Scoring uses the full-data flag (known after the fact)
to leave out hours when a station was closed.
"""

import argparse
import logging
import time

import pandas as pd

from bikecast import config
from bikecast.data.aggregate import flag_in_service
from bikecast.models.baseline import TARGETS, HistoricalAverage, SeasonalNaive

log = logging.getLogger(__name__)

PREDICTIONS_DIR = config.DATA / "predictions"


def forecast_days(week_start) -> list[pd.Timestamp]:
    start = pd.Timestamp(week_start)
    return [start + pd.Timedelta(days=i) for i in range(7)]


def visible_slice(data: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Rows strictly before `cutoff`, with the out-of-service flag recomputed on them alone."""
    past = data[data["ts"] < cutoff].copy()
    past["in_service"] = flag_in_service(past)
    return past


def run_backtest(data: pd.DataFrame, models: list, weeks=config.TEST_WEEKS) -> pd.DataFrame:
    """Long-format predictions: one row per model, week, station, hour, and target."""
    rows = []
    for week in weeks:
        week_start = pd.Timestamp(week)
        train = visible_slice(data, week_start)
        for model in models:
            model.fit(train)
        for day in forecast_days(week):
            history = visible_slice(data, day)
            actual = data[(data["ts"] >= day) & (data["ts"] < day + pd.Timedelta(days=1))]
            actual = actual[["station_id", "ts", "in_service", *TARGETS]]
            for model in models:
                pred = model.predict(history, day)
                if not pred["ts"].between(day, day + pd.Timedelta(hours=23)).all():
                    raise ValueError(f"{model.name} predicted outside {day.date()}")
                interval_cols = [
                    f"{t}_{b}" for t in TARGETS for b in ("lower", "upper") if f"{t}_{b}" in pred
                ]
                merged = actual.merge(
                    pred[["station_id", "ts", *TARGETS, *interval_cols]],
                    on=["station_id", "ts"],
                    how="left",
                    suffixes=("", "_pred"),
                    validate="one_to_one",
                )
                if merged[[f"{t}_pred" for t in TARGETS]].isna().any().any():
                    raise ValueError(f"{model.name} is missing predictions for {day.date()}")
                for target in TARGETS:
                    frame = pd.DataFrame(
                        {
                            "model": model.name,
                            "test_week": week_start,
                            "station_id": merged["station_id"],
                            "ts": merged["ts"],
                            "target": target,
                            "y": merged[target],
                            "yhat": merged[f"{target}_pred"],
                            "in_service": merged["in_service"],
                        }
                    )
                    if f"{target}_lower" in merged:
                        frame["yhat_lower"] = merged[f"{target}_lower"]
                        frame["yhat_upper"] = merged[f"{target}_upper"]
                    rows.append(frame)
        log.info("week %s done", week_start.date())
    return pd.concat(rows, ignore_index=True)


MODELS = ["seasonal_naive", "historical_average", "prophet", "torch_mlp"]
# The global neural model is a station-level model; the system total is a single series.
STATION_ONLY = {"torch_mlp"}


def make_model(name: str):
    if name == "seasonal_naive":
        return SeasonalNaive()
    if name == "historical_average":
        return HistoricalAverage()
    calendar = pd.read_parquet(config.PROCESSED / "calendar.parquet")
    weather = pd.read_parquet(config.PROCESSED / "weather_forecast.parquet")
    if name == "prophet":
        from bikecast.models.prophet_model import ProphetModel

        return ProphetModel(calendar, weather)
    if name == "torch_mlp":
        from bikecast.models.torch_model import TorchMLP

        return TorchMLP(calendar, weather)
    raise ValueError(f"Unknown model {name}; choose from {MODELS}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=MODELS, choices=MODELS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    for level, file in [("stations", "hourly"), ("system", "hourly_system")]:
        data = pd.read_parquet(config.PROCESSED / f"{file}.parquet")
        if "in_service" not in data:
            data["in_service"] = True  # system-wide totals are never closed
        for name in args.models:
            if level == "system" and name in STATION_ONLY:
                continue
            t0 = time.time()
            preds = run_backtest(data, [make_model(name)])
            preds.to_parquet(PREDICTIONS_DIR / f"{level}_{name}.parquet", index=False)
            log.info(
                "%s %s: %d prediction rows across %d weeks in %.0fs",
                level,
                name,
                len(preds),
                preds["test_week"].nunique(),
                time.time() - t0,
            )


if __name__ == "__main__":
    main()
