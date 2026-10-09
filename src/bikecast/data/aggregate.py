"""Hourly departures and arrivals per top station, plus system-wide totals.

Every (station, hour) in the range gets a row, even when no trips happened. Without the zero rows a
quiet hour would silently disappear, and lags or averages computed later would skip over it.

Timestamps are naive Boston wall-clock time, as Bluebikes publishes them. A departure counts in the
hour the trip started, an arrival in the hour it ended. Raw station IDs are mapped to canonical IDs
through the crosswalk first, so merged stations add up.
"""

import logging
import re
from datetime import datetime

import duckdb
import pandas as pd

from bikecast import config
from bikecast.data.trips import register_trips

log = logging.getLogger(__name__)

SYSTEM_ID = "SYSTEM"
# A station with no departures and no arrivals for this many hours in a row is treated as closed
# (winter removal, Marathon street closures), not as having zero demand.
CLOSED_MIN_HOURS = 48


def data_range(interim_dir=config.INTERIM) -> tuple[datetime, datetime]:
    """[start, end) covering every cleaned month: midnight of the first month to midnight after
    the last."""
    months = sorted(
        m.group(1)
        for p in interim_dir.glob("trips_*.parquet")
        if (m := re.fullmatch(r"trips_(\d{6})\.parquet", p.name))
    )
    if not months:
        raise FileNotFoundError(f"No cleaned trip files in {interim_dir}")
    start = datetime.strptime(months[0], "%Y%m")
    end = (
        pd.Timestamp(datetime.strptime(months[-1], "%Y%m")) + pd.offsets.MonthBegin(1)
    ).to_pydatetime()
    return start, end


def aggregate_hourly(
    con: duckdb.DuckDBPyConnection, station_ids: list[str], start: datetime, end: datetime
) -> pd.DataFrame:
    """Zero-filled hourly counts for the given canonical stations.

    Needs a `trips` view and a `crosswalk` table (station_id, canonical_id) on the connection.
    """
    con.register("selected", pd.DataFrame({"station_id": station_ids}))
    params = {"start": start, "end": end}
    return con.execute(
        """
        WITH hours AS (
            SELECT unnest(generate_series($start::TIMESTAMP, $end::TIMESTAMP - INTERVAL 1 HOUR,
                                          INTERVAL 1 HOUR)) AS ts
        ),
        grid AS (SELECT s.station_id, h.ts FROM selected s CROSS JOIN hours h),
        dep AS (
            SELECT c.canonical_id AS station_id, date_trunc('hour', t.started_at) AS ts,
                   count(*) AS n
            FROM trips t JOIN crosswalk c ON t.start_station_id = c.station_id
            WHERE t.started_at >= $start AND t.started_at < $end
            GROUP BY ALL
        ),
        arr AS (
            SELECT c.canonical_id AS station_id, date_trunc('hour', t.ended_at) AS ts,
                   count(*) AS n
            FROM trips t JOIN crosswalk c ON t.end_station_id = c.station_id
            WHERE t.ended_at >= $start AND t.ended_at < $end
            GROUP BY ALL
        )
        SELECT g.station_id, g.ts,
               coalesce(dep.n, 0)::INTEGER AS departures,
               coalesce(arr.n, 0)::INTEGER AS arrivals,
               (coalesce(arr.n, 0) - coalesce(dep.n, 0))::INTEGER AS net_flow
        FROM grid g
        LEFT JOIN dep USING (station_id, ts)
        LEFT JOIN arr USING (station_id, ts)
        ORDER BY g.station_id, g.ts
        """,
        params,
    ).df()


def aggregate_system(
    con: duckdb.DuckDBPyConnection, start: datetime, end: datetime
) -> pd.DataFrame:
    """Zero-filled hourly totals across every station, not only the top ones."""
    params = {"start": start, "end": end}
    return con.execute(
        """
        WITH hours AS (
            SELECT unnest(generate_series($start::TIMESTAMP, $end::TIMESTAMP - INTERVAL 1 HOUR,
                                          INTERVAL 1 HOUR)) AS ts
        ),
        dep AS (
            SELECT date_trunc('hour', started_at) AS ts, count(*) AS n FROM trips
            WHERE start_station_id IS NOT NULL AND started_at >= $start AND started_at < $end
            GROUP BY 1
        ),
        arr AS (
            SELECT date_trunc('hour', ended_at) AS ts, count(*) AS n FROM trips
            WHERE end_station_id IS NOT NULL AND ended_at >= $start AND ended_at < $end
            GROUP BY 1
        )
        SELECT 'SYSTEM' AS station_id, h.ts,
               coalesce(dep.n, 0)::INTEGER AS departures,
               coalesce(arr.n, 0)::INTEGER AS arrivals,
               (coalesce(arr.n, 0) - coalesce(dep.n, 0))::INTEGER AS net_flow
        FROM hours h LEFT JOIN dep USING (ts) LEFT JOIN arr USING (ts)
        ORDER BY h.ts
        """,
        params,
    ).df()


def flag_in_service(hourly: pd.DataFrame, min_hours: int = CLOSED_MIN_HOURS) -> pd.Series:
    """False for station-hours inside an all-zero run of at least `min_hours`.

    This is an evaluation and training mask, never a model input: a run's length is only known
    after it ends. Callers working on a time slice should compute it on that slice only.
    """
    df = hourly.sort_values(["station_id", "ts"])
    idle = (df["departures"] == 0) & (df["arrivals"] == 0)
    # A new run starts whenever station or idle state changes.
    run_id = ((idle != idle.shift()) | (df["station_id"] != df["station_id"].shift())).cumsum()
    run_len = idle.groupby(run_id).transform("size")
    closed = idle & (run_len >= min_hours)
    return (~closed).reindex(hourly.index)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    con = duckdb.connect()
    register_trips(con)
    con.register("crosswalk", pd.read_parquet(config.PROCESSED / "crosswalk.parquet"))
    stations = pd.read_parquet(config.PROCESSED / "stations.parquet")
    top_ids = stations.loc[stations["is_top"], "canonical_id"].tolist()
    start, end = data_range()

    hourly = aggregate_hourly(con, top_ids, start, end)
    hourly["in_service"] = flag_in_service(hourly)
    system = aggregate_system(con, start, end)
    hourly.to_parquet(config.PROCESSED / "hourly.parquet", index=False)
    system.to_parquet(config.PROCESSED / "hourly_system.parquet", index=False)

    n_hours = int((end - start).total_seconds() // 3600)
    n_trips = con.sql("SELECT count(*) FROM trips").fetchone()[0]
    log.info("Range %s to %s: %d hours", start, end, n_hours)
    log.info("Trips in (deduped): %d", n_trips)
    log.info(
        "Top %d stations: %d rows (expected %d), %.1f%% of station-hours have zero departures",
        len(top_ids),
        len(hourly),
        len(top_ids) * n_hours,
        100 * (hourly["departures"] == 0).mean(),
    )
    closed = hourly[~hourly["in_service"]]
    log.info(
        "Out of service: %d station-hours (%.1f%%) across %d stations",
        len(closed),
        100 * len(closed) / len(hourly),
        closed["station_id"].nunique(),
    )
    log.info(
        "System: %d rows, %d departures, %d arrivals",
        len(system),
        system["departures"].sum(),
        system["arrivals"].sum(),
    )


if __name__ == "__main__":
    main()
