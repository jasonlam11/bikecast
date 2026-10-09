"""Station crosswalk and top-station selection.

Bluebikes sometimes gives a station a new ID (dock replacement, relocation by a few meters, seasonal
winter sites). We merge two IDs into one station only when they are physically the same place and
did not run side by side:

- replacement: within MERGE_RADIUS_M (or same name within SAME_NAME_RADIUS_M) and their active
  periods do not overlap by more than OVERLAP_TOLERANCE_DAYS
- ghost: within MERGE_RADIUS_M and the smaller ID was active for at most GHOST_MAX_DAYS

IDs that are close but active at the same time (for example two dock groups on one corner) are
separate stations and stay separate.
"""

import logging
import re

import duckdb
import numpy as np
import pandas as pd

from bikecast import config
from bikecast.data.trips import register_trips

log = logging.getLogger(__name__)

MERGE_RADIUS_M = 50
SAME_NAME_RADIUS_M = 200
OVERLAP_TOLERANCE_DAYS = 1
GHOST_MAX_DAYS = 31

MERGES_REPORT = config.REPORTS / "station_merges.md"


def station_stats(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """One row per raw station ID with latest name, median location, activity window, volume."""
    return con.sql(
        """
        WITH obs AS (
            SELECT start_station_id AS station_id, start_station_name AS name,
                   start_lat AS lat, start_lng AS lng, rideable_type, started_at AS ts
            FROM trips WHERE start_station_id IS NOT NULL
            UNION ALL
            SELECT end_station_id, end_station_name, end_lat, end_lng, rideable_type, ended_at
            FROM trips WHERE end_station_id IS NOT NULL
        )
        SELECT
            station_id,
            arg_max(name, ts) AS name,
            count(DISTINCT name) AS n_names,
            -- e-bike GPS points drift; classic bikes report the dock location
            coalesce(median(lat) FILTER (rideable_type = 'classic_bike'), median(lat)) AS lat,
            coalesce(median(lng) FILTER (rideable_type = 'classic_bike'), median(lng)) AS lng,
            min(ts)::DATE AS first_seen,
            max(ts)::DATE AS last_seen,
            count(*) AS n_trip_ends
        FROM obs GROUP BY station_id ORDER BY station_id
        """
    ).df()


def haversine_m(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    )
    return 6_371_000 * 2 * np.arcsin(np.sqrt(a))


def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_merges(stats: pd.DataFrame) -> pd.DataFrame:
    """Return candidate pairs that should be merged, with the reason and distance."""
    s = stats.reset_index(drop=True)
    i, j = np.triu_indices(len(s), k=1)
    a, b = s.iloc[i].reset_index(drop=True), s.iloc[j].reset_index(drop=True)
    pairs = pd.DataFrame(
        {
            "id_a": a["station_id"],
            "id_b": b["station_id"],
            "name_a": a["name"],
            "name_b": b["name"],
            "dist_m": haversine_m(a["lat"], a["lng"], b["lat"], b["lng"]),
            "same_name": a["name"].map(_norm_name) == b["name"].map(_norm_name),
        }
    )
    start = np.maximum(a["first_seen"], b["first_seen"])
    end = np.minimum(a["last_seen"], b["last_seen"])
    pairs["overlap_days"] = (pd.to_datetime(end) - pd.to_datetime(start)).dt.days.clip(lower=0)
    life_a = (pd.to_datetime(a["last_seen"]) - pd.to_datetime(a["first_seen"])).dt.days + 1
    life_b = (pd.to_datetime(b["last_seen"]) - pd.to_datetime(b["first_seen"])).dt.days + 1
    pairs["shorter_life_days"] = np.minimum(life_a, life_b)

    near = pairs["dist_m"] <= MERGE_RADIUS_M
    near_same_name = pairs["same_name"] & (pairs["dist_m"] <= SAME_NAME_RADIUS_M)
    no_overlap = pairs["overlap_days"] <= OVERLAP_TOLERANCE_DAYS
    ghost = near & (pairs["shorter_life_days"] <= GHOST_MAX_DAYS)

    pairs["reason"] = pd.NA
    pairs.loc[ghost, "reason"] = "ghost"
    pairs.loc[(near | near_same_name) & no_overlap & pairs["reason"].isna(), "reason"] = (
        "replacement"
    )
    return pairs[pairs["reason"].notna()].reset_index(drop=True)


def build_crosswalk(stats: pd.DataFrame, merges: pd.DataFrame) -> pd.DataFrame:
    """Map every raw station ID to a canonical ID (union-find over merge pairs).

    The canonical ID of a group is the one seen most recently, then the busiest, so the
    crosswalk points at the station that exists today.
    """
    parent = {sid: sid for sid in stats["station_id"]}

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(merges["id_a"], merges["id_b"], strict=True):
        parent[root(a)] = root(b)

    groups = stats.assign(group=stats["station_id"].map(root))
    canonical = (
        groups.sort_values(["last_seen", "n_trip_ends"], ascending=False)
        .groupby("group")["station_id"]
        .first()
    )
    return pd.DataFrame(
        {"station_id": groups["station_id"], "canonical_id": groups["group"].map(canonical)}
    )


def rank_stations(con: duckdb.DuckDBPyConnection, crosswalk: pd.DataFrame) -> pd.DataFrame:
    """Departures + arrivals per canonical station inside the ranking window."""
    con.register("crosswalk", crosswalk)
    return con.sql(
        f"""
        WITH ends AS (
            SELECT start_station_id AS station_id FROM trips
            WHERE started_at >= DATE '{config.STATION_RANK_START}'
              AND started_at < DATE '{config.STATION_RANK_END}'
            UNION ALL
            SELECT end_station_id FROM trips
            WHERE started_at >= DATE '{config.STATION_RANK_START}'
              AND started_at < DATE '{config.STATION_RANK_END}'
        )
        SELECT c.canonical_id, count(*) AS rank_volume
        FROM ends e JOIN crosswalk c USING (station_id)
        GROUP BY 1 ORDER BY rank_volume DESC, canonical_id
        """
    ).df()


def neighbor_changes(stats: pd.DataFrame, top_ids: set[str]) -> pd.DataFrame:
    """Top stations with an unmerged neighbor raw ID (within MERGE_RADIUS_M) that opened or closed
    during the data range. A new dock next door can pull demand away: a structural break."""
    s = stats.reset_index(drop=True)
    start, end = s["first_seen"].min(), s["last_seen"].max()
    rows = []
    for t in s[s["station_id"].isin(top_ids)].itertuples():
        dist = haversine_m(t.lat, t.lng, s["lat"], s["lng"])
        near = s[(dist <= MERGE_RADIUS_M) & (s["station_id"] != t.station_id)]
        for n in near.itertuples():
            opened = n.first_seen > start + pd.Timedelta(days=30)
            closed = n.last_seen < end - pd.Timedelta(days=30)
            if opened or closed:
                rows.append(
                    {
                        "top_station": t.station_id,
                        "top_name": t.name,
                        "neighbor": n.station_id,
                        "neighbor_name": n.name,
                        "neighbor_first_seen": pd.Timestamp(n.first_seen).date(),
                        "neighbor_last_seen": pd.Timestamp(n.last_seen).date(),
                    }
                )
    return pd.DataFrame(rows)


def write_merges_report(
    merges: pd.DataFrame, crosswalk: pd.DataFrame, stats: pd.DataFrame, changes: pd.DataFrame
) -> None:
    multi = stats[stats["n_names"] > 1]
    canon = dict(zip(crosswalk["station_id"], crosswalk["canonical_id"], strict=True))
    lines = [
        "# Station merges (generated by stations.py)",
        "",
        f"Raw station IDs: {len(stats)}. "
        f"Canonical stations: {crosswalk['canonical_id'].nunique()}. "
        f"Merged pairs: {len(merges)}. IDs with more than one name (kept as one station, latest "
        f"name used): {len(multi)}.",
        "",
        "| ID A | ID B | Name A | Name B | Distance (m) | Overlap (days) | Reason | Canonical |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in merges.itertuples():
        lines.append(
            f"| {r.id_a} | {r.id_b} | {r.name_a} | {r.name_b} | {r.dist_m:.0f} | "
            f"{r.overlap_days} | {r.reason} | {canon[r.id_a]} |"
        )
    lines += [
        "",
        "## Top stations with a neighbor that opened or closed",
        "",
        "Possible structural breaks in demand. Not merged, because both ran at the same time.",
        "",
        "| Top station | Name | Neighbor | Neighbor name | Neighbor active |",
        "|---|---|---|---|---|",
    ]
    for r in changes.itertuples():
        lines.append(
            f"| {r.top_station} | {r.top_name} | {r.neighbor} | {r.neighbor_name} | "
            f"{r.neighbor_first_seen} to {r.neighbor_last_seen} |"
        )
    MERGES_REPORT.parent.mkdir(parents=True, exist_ok=True)
    MERGES_REPORT.write_text("\n".join(lines) + "\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    con = duckdb.connect()
    register_trips(con)
    stats = station_stats(con)
    merges = find_merges(stats)
    crosswalk = build_crosswalk(stats, merges)
    ranks = rank_stations(con, crosswalk)

    canonical = (
        stats.merge(crosswalk, on="station_id")
        .groupby("canonical_id")
        .agg(first_seen=("first_seen", "min"), last_seen=("last_seen", "max"))
        .reset_index()
        .merge(
            stats[["station_id", "name", "lat", "lng"]],
            left_on="canonical_id",
            right_on="station_id",
        )
        .drop(columns="station_id")
        .merge(ranks, on="canonical_id", how="left")
        .fillna({"rank_volume": 0})
        .astype({"rank_volume": "int64"})
        .sort_values(["rank_volume", "canonical_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    canonical["is_top"] = canonical.index < config.N_TOP_STATIONS

    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    crosswalk.to_parquet(config.PROCESSED / "crosswalk.parquet", index=False)
    canonical.to_parquet(config.PROCESSED / "stations.parquet", index=False)
    top = canonical[canonical["is_top"]]
    top.to_csv(config.PROCESSED / "top_stations.csv", index=False)
    write_merges_report(merges, crosswalk, stats, neighbor_changes(stats, set(top["canonical_id"])))

    log.info(
        "%d raw IDs -> %d canonical stations (%d merges). Top %d cover %.1f%% of ranking volume.",
        len(stats),
        crosswalk["canonical_id"].nunique(),
        len(merges),
        len(top),
        100 * top["rank_volume"].sum() / canonical["rank_volume"].sum(),
    )
    last_test_day = pd.Timestamp(max(config.TEST_WEEKS)) + pd.Timedelta(days=6)
    gone = top[pd.to_datetime(top["last_seen"]) < last_test_day]
    if len(gone):
        log.warning("Top stations not active through the last test week:\n%s", gone)


if __name__ == "__main__":
    main()
