"""Write the small, committed results snapshot the dashboard reads (reports/app_data/).

Model outputs only: backtest predictions for the top stations, station names and coordinates, and
system-level predictions. No trips and no processed hourly data, so the repo still never stores the
source data. Regenerate with `make snapshot` after `make backtest`.
"""

import logging

import pandas as pd

from bikecast import config
from bikecast.evaluation.analysis import enrich
from bikecast.evaluation.report import load_predictions

log = logging.getLogger(__name__)

APP_DATA = config.REPORTS / "app_data"
PRED_COLS = ["model", "test_week", "station_id", "ts", "target", "y", "yhat",
             "yhat_lower", "yhat_upper", "weather", "is_holiday", "holiday_name"]  # fmt: skip
CATEGORIES = ["model", "station_id", "target", "weather", "holiday_name"]
FLOATS = ["y", "yhat", "yhat_lower", "yhat_upper"]


def compact(preds: pd.DataFrame) -> pd.DataFrame:
    out = preds[[c for c in PRED_COLS if c in preds]].copy()
    for c in CATEGORIES:
        if c in out:
            out[c] = out[c].astype("category")
    for c in FLOATS:
        if c in out:
            out[c] = out[c].astype("float32")
    return out.reset_index(drop=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    P = config.PROCESSED
    weather, calendar = (
        pd.read_parquet(P / "weather_actual.parquet"),
        pd.read_parquet(P / "calendar.parquet"),
    )
    APP_DATA.mkdir(parents=True, exist_ok=True)
    for level in ["stations", "system"]:
        preds = compact(enrich(load_predictions(level), weather, calendar))
        path = APP_DATA / f"predictions_{level}.parquet"
        preds.to_parquet(path, compression="zstd", index=False)
        log.info("%s: %d rows, %.1f MB", path.name, len(preds), path.stat().st_size / 1e6)
    stations = pd.read_parquet(P / "stations.parquet")
    top = stations.loc[stations["is_top"], ["canonical_id", "name", "lat", "lng", "rank_volume"]]
    top.rename(columns={"canonical_id": "station_id"}).to_parquet(
        APP_DATA / "stations.parquet", index=False
    )


if __name__ == "__main__":
    main()
