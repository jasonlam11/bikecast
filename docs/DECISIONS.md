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

## Station crosswalk: merge only the same place, never neighbors
Bluebikes sometimes gives a station a new ID. We merge two IDs only when they are the same physical place
and did not run side by side. That means a replacement (within 50 m, or the same name within 200 m, and
active periods overlapping by at most 1 day) or a ghost (within 50 m, and the smaller ID was active for 31
days or less). IDs that are close but both active, such as the two Lafayette Square stations 16 m apart,
are separate dock groups with separate demand and stay separate. The canonical ID of a merged group is the
one seen most recently. This gave 8 merges, taking 640 raw IDs to 632 stations. Every merge is listed in
`reports/station_merges.md`, which `stations.py` generates. 39 IDs have more than one name (mostly
punctuation, like "Tremont St. at Court St." vs "Tremont St at Court St"). Each keeps one station with its
latest name. Location is the median of classic-bike trip coordinates, because e-bike GPS points drift.
Rejected: matching on name alone (names change, and different stations share names) and matching on
distance alone (it would merge real neighbors).

## Top 50 stations ranked on 2024 only
The top 50 are ranked by 2024 departures plus arrivals, which is entirely before the first test week
(2025-01-13). Ranking on the full range would use test-period volume to choose what we forecast, a small
form of leakage. All 50 stay active through September 2026. Together they cover 35.8% of 2024 trip ends.

## Known structural breaks at top stations
Two top stations got a new dock next door during the data range. These are flagged rather than merged, and
are expected to show up in error analysis. Nashua Street at Red Auerbach Way (A32025) gained an
"[Extension]" station on 2025-04-28, which replaced West End Park the same day. Copley Square (D32005)
gained Boylston St at Dartmouth St (D32055) on 2024-05-01.

## Hourly grid in naive Boston local time, with explicit zeros
Counts are hourly per canonical station on a complete grid (every station, every hour), so empty hours are
stored as 0 instead of going missing. A departure counts in the hour the trip started, an arrival in the
hour it ended. We use naive local wall-clock time because that is how Bluebikes publishes timestamps, and
every day then has exactly 24 hours, which suits a 24-hour day-ahead forecast. The cost is at daylight
saving changes, as checked in the data: the spring-forward 2am hour is always 0 (it does not exist), and
the fall-back 1am hour holds two real hours (378 system departures vs 293 the hour before on 2024-11-03).
That is 2 hours a year, small and documented. Rejected: UTC, which gives 23 and 25 hour local days and
complicates the daily forecast for no real gain. System-wide totals use every station, not only the top 50.

## Mask closed stations instead of scoring them as zero demand
Some top stations go dark for days or months: Newbury St at Hereford St (B32000) and Beacon St at
Massachusetts Ave (B32016) are removed in winter, and the Copley Square and Boylston St stations close for
the Boston Marathon each April. A run of 48 or more hours with no departures and no arrivals is flagged
`in_service = False`. That is 17,508 station-hours (1.5%) across 11 stations. Overnight lulls at busy
stations are much shorter than 48 hours. Closed hours stay in the table but are excluded from training
targets and from scoring, because predicting 0 for a closed station is trivial and would make winter and
April results look better than they are. This assumes operators know their own planned closures, which is
realistic. The flag is never a model input: a run's length is only known after it ends, so using it as a
feature would leak the future. In the backtest it is recomputed on each training slice so that slice uses
no later data. Rejected: swapping the winter-only stations for year-round ones (it does not handle the
Marathon closures) and scoring closed hours as zeros.

## Weather: archived forecasts for models, actuals for analysis
In real use, tomorrow's weather is a forecast. Models use Open-Meteo's Historical Forecast API (what
weather models predicted at the time) rather than observed weather. Limitation, to repeat in the README:
Open-Meteo builds this series from the first hours of each model run, so it is still somewhat more accurate
than a true 24-hour-ahead forecast, and results will be a little optimistic. Actuals (Open-Meteo archive,
ERA5 reanalysis) are also stored and used for EDA and for the rain vs dry breakdown, since that should
reflect what really happened. A model rerun on actuals gives a sensitivity check on how much weather
accuracy matters. Rejected: actuals as model inputs (they are more accurate than anything available the
night before, which leaks future information) and NOAA station data (needs a token and has gaps).

## How the two weather series compare
Temperature agrees closely (hourly correlation 0.991, mean absolute difference 1.1 C), and wind reasonably
(correlation 0.80). Precipitation differs much more: hourly precipitation in 5.4% of forecast hours vs
15.3% of actual hours, daily correlation 0.70, and 230 vs 344 days with at least 1 mm. Reanalysis tends to
report lots of light drizzle, and the forecast misses some rain and invents some. That gap is real and is
part of what makes day-ahead forecasting hard. The forecast series also has a few model artifacts, such as
119.5 mm in one hour on 2024-06-14, when the actual day total was 10.4 mm. The stored data stays raw; the
feature step will clip and log-transform precipitation so single spikes cannot dominate.

## Weather time zones
We request UTC from Open-Meteo and convert to naive Boston time to match the trip grid. At fall-back the
two UTC hours that map to local 1am are averaged. At spring-forward the missing local 2am is interpolated.
Tested on both 2024 DST dates.

## Calendar features
Hour, weekday, month, weekend, US federal plus Massachusetts holidays from the `holidays` package (this
includes Patriots' Day, Marathon Monday), and an approximate university semester flag (Jan 20 to May 15,
Sep 1 to Dec 20, every year). Many top stations are at MIT, Harvard, and BU, so terms likely matter. The
semester dates are rough by a few days at each edge. Exact academic calendars were rejected for now because
they differ by school and year, and a few days of edge error should not matter. All of these features are
known far in advance, so none can leak. Note: test week 2025-10-13 starts on Columbus Day, which is
realistic and is kept.

## EDA findings that shape the models
From `notebooks/01_eda.ipynb`: demand is flat year over year (Jan to Sep trips: 2025 -0.4%, 2026 -1.6% vs
2024), so models get no growth term. Season (3.7x between September and January), weekday vs weekend
shape, temperature, rain, holidays (-31%), and the academic calendar (MIT and Harvard share of top-50
weekday departures 25.0% in term vs 20.9% out) all matter, so each becomes a feature. The worst days are
storms, such as the 2026-02-23 snowstorm (1 trip system-wide), and these stay in the data as real
forecast failures rather than being masked, because a station-level 48-hour closure rule does not cover a
one-day system shutdown and an operator could not have known about it in advance with certainty. Ruff rule
PD010 (prefer pivot_table over unstack) is turned off, since groupby then unstack is clear, idiomatic pandas.
