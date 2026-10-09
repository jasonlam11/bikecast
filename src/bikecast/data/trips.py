"""Shared DuckDB view over the cleaned monthly trip files."""

import duckdb

from bikecast import config


def register_trips(con: duckdb.DuckDBPyConnection, glob: str | None = None) -> None:
    """Create a `trips` view over all cleaned months.

    Some trips appear in two monthly files (usually ones that cross a month boundary), so we keep
    one row per (ride_id, started_at). Ride IDs alone are not unique: a few are reused for
    different trips.
    """
    glob = glob or str(config.INTERIM / "trips_*.parquet")
    con.execute(
        f"""
        CREATE OR REPLACE VIEW trips AS
        SELECT * EXCLUDE (rn) FROM (
            SELECT *, row_number() OVER (PARTITION BY ride_id, started_at) AS rn
            FROM read_parquet('{glob}')
        ) WHERE rn = 1
        """
    )
