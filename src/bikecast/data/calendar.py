"""Calendar features for the hourly grid. Everything here is known years in advance."""

import logging

import holidays
import pandas as pd

from bikecast import config
from bikecast.data.aggregate import data_range

log = logging.getLogger(__name__)

# Approximate Boston-area university terms (MIT, Harvard, BU, Northeastern overlap closely).
# Month-day ranges, inclusive, reused every year. Rough by a few days at each edge.
SEMESTERS = [("01-20", "05-15"), ("09-01", "12-20")]


def is_semester(dates: pd.Series) -> pd.Series:
    md = dates.dt.strftime("%m-%d")
    out = pd.Series(False, index=dates.index)
    for start, end in SEMESTERS:
        out |= (md >= start) & (md <= end)
    return out


def calendar_features(ts: pd.Series) -> pd.DataFrame:
    """Hour, weekday, month, weekend, US + Massachusetts holidays, and semester flag."""
    years = range(ts.dt.year.min(), ts.dt.year.max() + 1)
    ma_holidays = holidays.country_holidays("US", subdiv="MA", years=years)
    day = ts.dt.normalize()
    names = day.dt.date.map(ma_holidays.get)
    return pd.DataFrame(
        {
            "ts": ts,
            "hour": ts.dt.hour.astype("int8"),
            "dow": ts.dt.dayofweek.astype("int8"),  # Monday = 0
            "month": ts.dt.month.astype("int8"),
            "is_weekend": ts.dt.dayofweek >= 5,
            "is_holiday": names.notna(),
            "holiday_name": names.astype("string"),
            "is_semester": is_semester(day),
        }
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    start, end = data_range()
    ts = pd.Series(pd.date_range(start, end, freq="h", inclusive="left"))
    df = calendar_features(ts)
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.PROCESSED / "calendar.parquet", index=False)
    log.info(
        "%d hours, %d holiday days, %.0f%% of hours in semester",
        len(df),
        df.loc[df["is_holiday"], "ts"].dt.date.nunique(),
        100 * df["is_semester"].mean(),
    )


if __name__ == "__main__":
    main()
