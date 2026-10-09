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

## Baselines
Seasonal naive forecasts each station-hour as the value at the same station and hour one week earlier.
Historical average forecasts the mean of the same station, weekday, and hour over the previous 8 weeks,
skipping out-of-service hours. Eight weeks is long enough to average out noise in small counts and short
enough to follow the season. It was set before looking at any test results and is not tuned. When last
week's hour was out of service, seasonal naive falls back to the historical average, because copying a
closure forward would forecast 0 for a station that has reopened. Rejected: forecasting 0 in that case,
which punishes the baseline for a known closure rather than for forecasting skill.

## Two baselines, two skill columns
Per the plan, skill is reported against seasonal naive. On our data the historical average is clearly
stronger: station-level MAE 1.89 vs 2.32 (about 19% better), because a single week's count at one station
is very noisy and averaging 8 weeks smooths it out. So results.md also reports skill against the historical
average, and a Phase 2 model only counts as a real improvement if it beats both. Rejected: reporting only
the weaker baseline, which would overstate any model's gain.

## Backtest protocol
For each of the 10 test weeks, models are fit once on all data before the week starts. Each day D is then
forecast at midnight using only data before D. Fitting once per week rather than once per day keeps
Prophet and PyTorch affordable in Phase 2 (10 fits instead of 70), while predictions still use the most
recent data through their lag features. The harness hands models only the visible slice and checks that
predictions fall inside D and cover every station-hour. A test with a spy model confirms that no model
ever sees a timestamp at or after its cutoff. The out-of-service flag is recomputed on each visible slice.
Scoring leaves out hours that turned out to be closed (1,193 station-hours across the test weeks).

## First feature table
`features/build.py` builds lags at 24, 48, and 168 hours, the same-hour mean over the previous 7 days, the
previous day's mean, calendar features, and forecast weather (precipitation clipped at 30 mm and
log-transformed). Every demand feature for day D uses only data from before D. Row shifts are only valid on
a complete hourly grid, so the builder checks the grid first and fails if it is incomplete. Phase 2 models
will use this table.

## How we test for leakage
`tests/test_leakage.py` builds features and baseline forecasts for a day D, then sets every demand value
at or after D to 1,000,000 and builds them again. Nothing for D may change. It checks five cutoffs,
including two in the middle of a station closure, and runs the full backtest with the last test day
corrupted. Weather for D itself is not corrupted, because the archived forecast for D is a legitimate
input; weather after D is. A control test adds a deliberately leaky feature (one hour back) and confirms
the same check catches it, so the test is not passing trivially. Phase 2 models will be added to the same
test.

## Prophet: two daily curves instead of one
Prophet's default daily seasonality is one curve shared by every day. EDA shows weekdays have two commute
peaks (8am and 5pm) while weekends and holidays have one midday hump, so one shared curve would average
two different shapes into a shape that fits neither. We turn the default off and add two conditional
daily seasonalities (Fourier order 12, enough for sharp peaks): one for working days and one for weekends
plus holidays. Weekly seasonality, yearly seasonality, US and Massachusetts holidays, and extra regressors
for forecast temperature, precipitation (clipped, log1p), wind, and the semester flag complete the model.

## Prophet: raw counts, clipped at zero, one fit per week
One Prophet per station per target (100 per test week, plus 2 for system totals) is fit on all in-service
history before the test week. The target is the raw hourly count, and forecasts and the 80% interval are
clipped at 0. Rejected: a log target, because back-transforming biases the mean forecast low. Prophet uses
no recent lags, so a single fit per week forecasts all 7 days, with weather forecasts and calendar as the
only day-specific inputs. That is how Prophet is meant to be used, and it is one reason a lag-based model
could beat it on day-ahead forecasts. Fits run in 6 parallel processes.

## Known limitation: Prophet yearly seasonality with under 2 years of history
Prophet warns that yearly seasonality is under-identified with less than 730 days of history, which
applies to the first four test weeks (12 to 21 months of training data). We noticed the warning during the
first real run and kept the setting as planned. Changing the model after it has started scoring test weeks
would be tuning on the test set. The temperature regressor carries much of the seasonal signal anyway.
Expect Prophet to be weaker in 2025 than in 2026.

## Global MLP: one station-day per sample
The PyTorch model forecasts all 24 hours of day D for one station at once (48 outputs: departures and
arrivals). Inputs are the same day-ahead features as the feature table (lags 24, 48, and 168 hours,
same-hour 7-day mean, previous-day mean, all log1p), D's forecast weather by hour (standardized on
training days only), D's calendar (weekday one-hot, month as sin and cos, holiday, semester), the share of
hours the station was closed on D-1 and D-7, and an 8-dimensional learned station embedding. The network
is small on purpose: two hidden layers (128, 64), ReLU, dropout 0.1, about 50,000 parameters for roughly
18,000 to 49,000 training samples. One sample per station-day matches how the forecast is used (a full day
at midnight) and lets the model learn the shape of a day jointly.

## Global MLP: loss, validation, early stopping
The 14 days before each test week are the validation window, and the model trains only on days before
that. Training stops after 10 epochs without validation improvement (max 200) and keeps the best epoch.
Two losses are trained per week: Poisson negative log-likelihood (natural for counts, predicts the mean)
and L1 (predicts the median, which is what MAE rewards). The one with lower validation MAE is kept, and the
choice is logged. Closed hours get zero weight in the loss. Seeds are fixed and algorithms deterministic,
and a test confirms that two runs give identical forecasts. Hyperparameters were set once before any
results and are written with each week's best epoch to `reports/torch_mlp_training_log.json`. Rejected:
tuning hyperparameters on test weeks, and a larger network or LSTM before the small MLP is shown to work.

## Global MLP at station level only
The system-wide total is one series with about 1,000 days. A global model exists to share patterns across
many series, so it has nothing to share there. System-level results compare the baselines and Prophet.

## Phase 2 results (from reports/results.md)
The global MLP is the best station-level model: MAE 1.66 bikes per hour vs 1.89 for the historical
average (+12% skill) and 2.32 for seasonal naive (+28%). It also has the lowest RMSE, so its L1 training
(median forecasts) did not cost it on large misses. It beats the historical average in 9 of 10 weeks, with
a slight loss in the July 2025 week. Its gain is largest on rainy days (18% vs the historical average on
rainy days, 9% on dry days), so the weather inputs are doing real work. Prophet only matches the
historical average at station level (+1.5%) but is the best system-wide model (+4 to 5% over the
historical average). L1 beat Poisson on validation MAE in all 10 weeks. A full rerun produced identical
predictions.

## Why Prophet struggles at station level
Prophet's components add together, so the summer level shift and the weekday peak shape are added at
3am too. Night MAE is 0.81 for Prophet vs 0.49 for the historical average. Its noise model assumes the
same spread at every hour, while count data spreads more when demand is higher. Coverage of the 80%
interval is 84.9% overall, but 97.9% at night (too wide) and 66.7% in the 4 to 7pm peak (too narrow).
Multiplicative seasonality or a variance-stabilizing target transform would likely help. These were not
tried, because choosing them after seeing test results would be tuning on the test set; they are noted
as future work. At system level, counts are large and closer to normally distributed, which suits
Prophet's assumptions, and that is where it does best.

## Weather sensitivity
Rerunning the MLP with actual weather instead of archived forecasts improves station MAE by only 1.5%
(1.664 to 1.639). So the forecast-vs-actual weather gap is small for this model, and the headline numbers,
which use forecasts, are close to what perfect weather knowledge would give. The archived forecasts are
still slightly better than true next-day forecasts (see the weather entry above), so the real-world gap
could be a little larger.

## Rebalancing takeaway: method and thresholds
The takeaway uses the MLP's forecasts on the 48 non-holiday weekdays in the backtest. Morning net flow is
forecast arrivals minus departures over 7, 8, and 9am, skipping station-mornings with a closed hour. A
station "drains" if that flow is below -5 bikes on at least 80% of mornings and "fills" if above +5 on at
least 80%. These thresholds were set before looking at any output: 5 bikes over three hours is a clear
imbalance rather than noise for a station with a few trips per hour, and 80% means "almost every
weekday". Both are judgment calls, not operator figures. Bikes to move is the sum of the flagged stations' median
morning flow, reported next to the actual median for the same stations and days. The historical average
flags almost the same stations, and the report says so: the list of stations is a property of the city,
not of the model; the model's value is the day-specific amount. Rejected: estimating truck trips, which
would need dock capacity and starting inventory that trip data does not have.

## Committed results snapshot for the dashboard
The rule is that raw and processed data are never committed. The one exception is `reports/app_data/`
(about 4 MB): backtest predictions for all four models plus station names and coordinates, written by
`make snapshot`. It holds model outputs only, no trips and no hourly demand tables beyond the scored
backtest rows, and it lets anyone run the dashboard right after cloning, or host it, without a 40-minute
rebuild. Rejected: requiring the full pipeline before the app can open, which would mean almost nobody
reviewing the repo sees it.

## Dashboard framing: backtest days, not live forecasts
The app shows past days from the backtest, labeled "backtest day", with forecast and actual side by side.
We have no live data feed, and presenting backtest output as "tomorrow" would be misleading. The map colors
stations by forecast net flow over a time window (7 to 10am or 4 to 7pm), using the same red-drains,
blue-fills colors as the EDA. The app pins Streamlit's light theme, because the chart palette is designed
for a light background.

## README numbers are copied by code
The README's results tables, per-station skill, weather sensitivity, and rebalancing summary sit between
marker comments and are filled by `make readme` from reports/results.md and reports/rebalancing.md. A test
re-renders the README and fails if anything differs, so a stale or hand-edited number cannot reach the main
page. The planned `02_baselines.ipynb` was dropped: results.md already reports both baselines with every
breakdown, so a notebook would only duplicate it.

## No event feature, even though events cause the biggest misses
Error analysis shows the worst days for both learned models are crowd events near the Charles River (likely
the Head of the Charles Regatta on 2025-10-18, and Boston Marathon weekend on 2025-04-19). An event
calendar is the obvious fix, but adding it now would mean changing the model because of what the test weeks
showed, which is tuning on the test set. It is listed as the first item of future work instead. A fair test
of it would need new test weeks chosen before the feature is built.

## Seeded Prophet intervals
The fresh-clone reproducibility test matched every point forecast exactly, but Prophet's interval bounds
differed slightly, moving a few coverage figures by 0.1 point. Prophet simulates its interval with numpy's
global random state, which we had not seeded. Each station and target now seeds it from a hash of its
names right before forecasting, so intervals are identical across runs and independent of which worker
process handles which model.

## Reproducibility check result
A fresh clone of the repo, run with `uv sync` and `make all` on 2026-10-09, reproduced reports/results.md,
reports/rebalancing.md, README.md, the station merges, the MLP training log, and the dashboard snapshot
exactly (after the seeded-interval fix above). The Phase 3 done criterion, that a stranger can clone, run
the make targets, and get the same results table, is met.
