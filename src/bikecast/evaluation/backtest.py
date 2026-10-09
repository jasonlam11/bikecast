"""Rolling-origin backtest.

For each test week: fit on data before the week starts. Then for each day D in the week, at
midnight, predict all 24 hours of D using only data before D, and keep the actuals for scoring.
Models never receive rows at or after the time they are forecasting from.

`in_service` is recomputed on every slice a model sees, so a closure that is still going on at the
cut is judged only on what was known then. Scoring uses the full-data flag (known after the fact)
to leave out hours when a station was closed.
"""

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
                merged = actual.merge(
                    pred[["station_id", "ts", *TARGETS]],
                    on=["station_id", "ts"],
                    how="left",
                    suffixes=("", "_pred"),
                    validate="one_to_one",
                )
                if merged[[f"{t}_pred" for t in TARGETS]].isna().any().any():
                    raise ValueError(f"{model.name} is missing predictions for {day.date()}")
                for target in TARGETS:
                    rows.append(
                        pd.DataFrame(
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
                    )
        log.info("week %s done", week_start.date())
    return pd.concat(rows, ignore_index=True)


def baseline_models() -> list:
    return [SeasonalNaive(), HistoricalAverage()]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    for level, file in [("stations", "hourly"), ("system", "hourly_system")]:
        data = pd.read_parquet(config.PROCESSED / f"{file}.parquet")
        if "in_service" not in data:
            data["in_service"] = True  # system-wide totals are never closed
        t0 = time.time()
        preds = run_backtest(data, baseline_models())
        for name, df in preds.groupby("model"):
            df.to_parquet(PREDICTIONS_DIR / f"{level}_{name}.parquet", index=False)
        log.info(
            "%s: %d prediction rows across %d weeks in %.0fs",
            level,
            len(preds),
            preds["test_week"].nunique(),
            time.time() - t0,
        )


if __name__ == "__main__":
    main()
