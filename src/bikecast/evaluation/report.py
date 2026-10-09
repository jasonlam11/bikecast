"""Write reports/results.md and result figures from saved backtest predictions.

Every number in the README comes from this file's output. Never edit results.md by hand.
"""

import logging

import matplotlib.pyplot as plt
import pandas as pd

from bikecast import config, viz
from bikecast.evaluation.backtest import PREDICTIONS_DIR
from bikecast.evaluation.metrics import mae, rmse, skill, wape

log = logging.getLogger(__name__)

BASELINE = "seasonal_naive"
STRONG_BASELINE = "historical_average"
RESULTS = config.REPORTS / "results.md"
SEASONS = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
           6: "summer", 7: "summer", 8: "summer", 9: "fall", 10: "fall", 11: "fall"}  # fmt: skip
HOUR_BANDS = [(0, 5, "night 0-5"), (6, 9, "am peak 6-9"), (10, 15, "midday 10-15"),
              (16, 19, "pm peak 16-19"), (20, 23, "evening 20-23")]  # fmt: skip
RAIN_MM = 1.0


def load_predictions(level: str) -> pd.DataFrame:
    files = sorted(PREDICTIONS_DIR.glob(f"{level}_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No {level} predictions in {PREDICTIONS_DIR}; run backtest.py")
    preds = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return preds[preds["in_service"]].copy()


def add_groups(preds: pd.DataFrame, weather_actual: pd.DataFrame) -> pd.DataFrame:
    p = preds
    p["day_type"] = p["ts"].dt.dayofweek.map(lambda d: "weekend" if d >= 5 else "weekday")
    p["season"] = p["test_week"].dt.month.map(SEASONS)
    hour = p["ts"].dt.hour
    p["hour_band"] = ""
    for lo, hi, label in HOUR_BANDS:
        p.loc[hour.between(lo, hi), "hour_band"] = label
    rain = weather_actual.set_index("ts")["precip_mm"].resample("D").sum() >= RAIN_MM
    p["weather"] = p["ts"].dt.normalize().map(rain).map({True: "rainy", False: "dry"})
    return p


def model_order(models) -> list[str]:
    return [BASELINE, *sorted(m for m in models if m != BASELINE)]


def score(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "MAE": mae(df["y"], df["yhat"]),
            "RMSE": rmse(df["y"], df["yhat"]),
            "WAPE": wape(df["y"], df["yhat"]),
            "n": len(df),
        }  # fmt: skip
    )


def scored_table(preds: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Metrics per model and group, with skill against the seasonal naive in the same group."""
    t = preds.groupby(["model", *by]).apply(score, include_groups=False).reset_index()
    base = t.loc[t["model"] == BASELINE, [*by, "MAE"]].rename(columns={"MAE": "baseline_mae"})
    t = t.merge(base, on=by)
    t["skill"] = skill(t["MAE"], t["baseline_mae"])
    strong = t.loc[t["model"] == STRONG_BASELINE, [*by, "MAE"]].rename(
        columns={"MAE": "strong_mae"}
    )
    t = t.merge(strong, on=by)
    t["skill vs hist avg"] = skill(t["MAE"], t["strong_mae"])
    t["model"] = pd.Categorical(t["model"], model_order(t["model"].unique()))
    return t.sort_values([*by, "model"]).reset_index(drop=True)


def fmt_table(t: pd.DataFrame, cols: list[str]) -> str:
    out = t[cols].copy()
    for c in cols:
        if c in {"MAE", "RMSE"}:
            out[c] = out[c].map("{:.3f}".format)
        elif c == "WAPE":
            out[c] = out[c].map("{:.1%}".format)
        elif c.startswith("skill"):
            out[c] = out[c].map("{:+.1%}".format)
        elif c == "n":
            out[c] = out[c].map("{:,.0f}".format)
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines = ["| " + " | ".join(str(v) for v in row) + " |" for row in out.itertuples(index=False)]
    return "\n".join([header, sep, *lines])


def wide_mae(preds: pd.DataFrame, by: str) -> pd.DataFrame:
    """MAE with groups as rows and models as columns, both targets pooled."""
    t = preds.groupby([by, "model"]).apply(lambda d: mae(d["y"], d["yhat"]), include_groups=False)
    t = t.unstack("model")[model_order(preds["model"].unique())]
    return t


def fmt_wide(t: pd.DataFrame, index_name: str) -> str:
    header = f"| {index_name} | " + " | ".join(t.columns) + " |"
    sep = "|" + "|".join("---" for _ in range(len(t.columns) + 1)) + "|"
    lines = [
        f"| {idx} | " + " | ".join(f"{v:.3f}" for v in row) + " |"
        for idx, row in zip(t.index, t.values, strict=True)
    ]
    return "\n".join([header, sep, *lines])


def plot_mae_by_hour(preds: pd.DataFrame) -> None:
    t = (
        preds.groupby([preds["ts"].dt.hour, "model"])
        .apply(lambda d: mae(d["y"], d["yhat"]), include_groups=False)
        .unstack("model")[model_order(preds["model"].unique())]
    )
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, m in enumerate(t.columns):
        ax.plot(t.index, t[m], color=viz.SERIES[i], marker="o", markersize=4, label=m)
    ax.set_title("Station-level MAE by hour of day, all test weeks")
    ax.set_xlabel("hour of day")
    ax.set_ylabel("MAE (bikes per hour)")
    ax.set_xticks(range(0, 24, 3))
    ax.legend(loc="upper left")
    viz.save(fig, "results_mae_by_hour")
    plt.close(fig)


def plot_mae_by_week(preds: pd.DataFrame) -> None:
    t = wide_mae(preds, "test_week")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    width = 0.8 / len(t.columns)
    x = range(len(t))
    for i, m in enumerate(t.columns):
        ax.bar([xi + (i - (len(t.columns) - 1) / 2) * width for xi in x], t[m], width=width * 0.92,
               color=viz.SERIES[i], label=m)  # fmt: skip
    ax.set_xticks(list(x))
    ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in t.index], rotation=30, ha="right")
    ax.set_title("Station-level MAE by test week")
    ax.set_ylabel("MAE (bikes per hour)")
    ax.legend(loc="upper left")
    viz.save(fig, "results_mae_by_week")
    plt.close(fig)


def build_report() -> str:
    weather_actual = pd.read_parquet(config.PROCESSED / "weather_actual.parquet")
    stations = add_groups(load_predictions("stations"), weather_actual)
    system = add_groups(load_predictions("system"), weather_actual)
    all_station_rows = pd.concat(
        [pd.read_parquet(f) for f in PREDICTIONS_DIR.glob(f"stations_{BASELINE}.parquet")]
    )
    masked = (~all_station_rows["in_service"]).sum()

    weeks = sorted(stations["test_week"].unique())
    cols = ["model", "target", "MAE", "RMSE", "WAPE", "skill", "skill vs hist avg"]
    parts = [
        "# BikeCast results",
        "",
        "_Generated by `src/bikecast/evaluation/report.py`. Do not edit by hand._",
        "",
        "## Setup",
        "",
        f"- Rolling-origin backtest over {len(weeks)} test weeks (Monday to Sunday): "
        + ", ".join(pd.Timestamp(w).strftime("%Y-%m-%d") for w in weeks)
        + ".",
        "- Each day is forecast at midnight for all 24 hours, using only data from before that "
        "day.",
        f"- Station level: the {stations['station_id'].nunique()} busiest stations "
        "(ranked on 2024).",
        f"- Out-of-service station-hours are not scored: {masked:,} of {len(all_station_rows):,} "
        "station-hour-target rows in the test weeks.",
        "- Skill = 1 - MAE / seasonal naive MAE on the same rows. Positive means better than the "
        "baseline. Skill vs hist avg uses the historical average baseline instead, which is the "
        "stronger of the two.",
        "",
        "## Station level (per station-hour)",
        "",
        fmt_table(scored_table(stations, ["target"]), cols),
        "",
        "## System-wide (hourly totals across all stations)",
        "",
        fmt_table(scored_table(system, ["target"]), cols),
        "",
        "## Station-level MAE by test week (departures and arrivals pooled)",
        "",
        fmt_wide(
            wide_mae(stations, "test_week").rename(index=lambda d: d.strftime("%Y-%m-%d")),
            "test week",
        ),  # fmt: skip
        "",
        "![MAE by week](figures/results_mae_by_week.png)",
        "",
        "## Station-level breakdowns (MAE, both targets pooled)",
        "",
    ]
    for by, title in [("season", "Season"), ("day_type", "Weekday vs weekend"),
                      ("weather", f"Rain vs dry (actual daily precipitation >= {RAIN_MM:g} mm)"),
                      ("hour_band", "Hour of day")]:  # fmt: skip
        t = wide_mae(stations, by)
        if by == "hour_band":
            t = t.reindex([b[2] for b in HOUR_BANDS])
        parts += [f"### {title}", "", fmt_wide(t, by), ""]
    parts += ["![MAE by hour](figures/results_mae_by_hour.png)", ""]

    plot_mae_by_hour(stations)
    plot_mae_by_week(stations)
    return "\n".join(parts)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    viz.use_style()
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(build_report())
    log.info("Wrote %s", RESULTS)


if __name__ == "__main__":
    main()
