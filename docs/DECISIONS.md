# Decisions

One short paragraph per decision: what we chose, what we rejected, and why. Newest at the bottom.

## Data source: discover files from the S3 bucket listing
The Bluebikes System Data page links to the public bucket `s3.amazonaws.com/hubway-data`. File names are not
consistent (`.zip`, `.csv.zip`, and one bare `.csv`), so `download.py` reads the bucket listing and matches
`YYYYMM-bluebikes-tripdata.*` instead of building URLs by hand. Rejected: hardcoded URL templates, which
would silently miss the odd months.

## Process one month at a time
The laptop has little free disk and the raw CSVs total several GB. The pipeline downloads one month, cleans
it to Parquet, deletes the raw CSV, and moves on. Parquet is written to a temp name and renamed only when
complete, so a crash never leaves a half-written month that looks finished. Rejected: download everything
first, then clean.

## Support both CSV schemas, use only the new one
The format changed in April 2023 (legacy `tripduration, starttime, usertype` to new `ride_id, started_at,
member_casual`). Our range starts January 2024, so every file we use is the new schema. The cleaner still
detects and normalizes the legacy schema, tested on a fixture, because the plan requires it and it lets us
extend back later. Legacy files have no ride ID, so duplicates there are detected on the full row. Legacy
station IDs are numeric and do not match new IDs, so mixing eras would need a coordinate-based crosswalk.

## Cleaning rules
Applied in order, each logged per month in `data/interim/cleaning_log.csv`: duplicate ride ID, unparseable
timestamp, duration <= 0, duration under 60 seconds, duration over 24 hours, start or end at a test or
maintenance station, and no station on either end. Bluebikes says it removes trips under 60 seconds, but
about 1.3% of January 2024 rows were still under 60 seconds, so we enforce it ourselves. Trips over 24 hours
are lost or unreturned bikes, not demand. Rejected: a tighter cap such as 3 hours, which would drop real
day-pass rides.

## Keep a trip if one end has a station
A trip with a start station but no end station still counts as a departure at its start. We only drop a
trip when both ends are missing. Rejected: dropping any trip with a missing end, which would undercount
departures for no reason.

## Cleaning results and the June 2024 short-trip change
Across 33 months (Jan 2024 to Sep 2026), 12,893,499 trips came in and 12,863,734 were kept (0.23% removed):
517 non-positive duration, 17,033 under 60 seconds, 9,120 over 24 hours, 1,920 at a test or depot station,
1,175 with no station. All 17,033 short trips are from Jan to May 2024. From June 2024 the files contain
none, so Bluebikes started filtering them upstream then. Applying the same 60 second rule ourselves keeps
the series consistent across that change. The only test or depot station found is ID X32999 ("18 Dorrance
Warehouse", later "440 Rutherford Ave Depot"), so the rule matches on that ID as well as on name patterns.
"Temporary Winter Location" stations are real seasonal stations and are kept. Station IDs are case
sensitive: C32014 and c32014 are different stations.

## Duplicate trips across monthly files
75 trips appear in two monthly files with the same ride ID and start time (mostly trips that cross a month
boundary). Another 123 ride IDs are reused for different trips. Downstream steps dedupe on (ride ID, start
time), which removes true repeats without merging different trips that share an ID.
