# BikeCast Project Plan

## 1. The question

**Can we forecast tomorrow's hourly bike pickups and dropoffs at Boston's busiest Bluebikes
stations well enough to plan rebalancing, and which kind of model does it best?**

Bike share operators move bikes by truck so stations don't run empty or full. Knowing tomorrow's
demand by hour lets them plan routes the night before. We compare three model families and end
with a concrete recommendation.

**Deliverables**
1. A reproducible pipeline from public data to forecasts.
2. An honest model comparison on a rolling backtest.
3. Error analysis: where and when each model fails.
4. One operational takeaway (for example: "these 10 stations drain every weekday morning;
   restock them before 7am").
5. A Streamlit dashboard and a strong README.

## 2. Scope

- **Target:** hourly departures (pickups) and arrivals (dropoffs) per station.
- **Stations:** the top 50 stations by total trip volume over the training period. Also forecast
  system-wide hourly totals. (Per-station data for small stations is mostly zeros and adds noise.)
- **Horizon:** day-ahead. At midnight, forecast all 24 hours of the next day.
- **Time range:** January 2024 through the most recent full month available. Check what exists.
- **Derived metric:** net flow = arrivals minus departures. Strongly negative net flow means the
  station is draining. This drives the rebalancing takeaway.

## 3. Data sources (all free)

- **Bluebikes trip history:** monthly CSVs linked from the Bluebikes "System Data" page.
  Find the current download location there rather than assuming a URL.
  - **Watch out:** the CSV format changed around 2023 (new column names like `started_at`,
    `start_station_id`, `rideable_type`). Write the cleaner to detect and handle each schema,
    and test it on a sample of each.
  - **Watch out:** station IDs and names can change over time. Build a station crosswalk
    (`stations.py`) using IDs, names, and coordinates, and log any merges in DECISIONS.md.
  - Drop obvious bad trips (negative or near-zero duration, extremely long trips, test or
    maintenance stations) and document the exact rules and how many rows each removes.
  - **Disk constraint:** process one month at a time (download, convert to Parquet, delete the
    raw CSV) so the full raw history never has to sit on disk at once.
- **Weather:** hourly temperature, precipitation, and wind for Boston from the Open-Meteo
  historical weather API (free, no key) or NOAA.
  - **Leakage note:** in real life, tomorrow's weather is a *forecast*, not the actual weather.
    Using actual weather makes results look slightly better than reality. Either use historical
    weather *forecasts* if a free source exists, or use actuals and state this limitation clearly
    in the README and DECISIONS.md.
- **Calendar:** US and Massachusetts holidays (the `holidays` package), plus day of week,
  hour, month. Optional: Boston-area university semester dates, since students drive demand.

## 4. Pipeline

1. `download.py`: fetch monthly trip files into `data/raw/` (skip months already converted).
2. `clean.py`: normalize both schemas into one, apply cleaning rules, write Parquet to
   `data/interim/`, then delete that month's raw file.
3. `stations.py`: build the station crosswalk, pick the top 50 stations.
4. `aggregate.py`: DuckDB query to hourly counts per station (departures, arrivals), with
   **explicit zero rows for hours with no trips** (missing hours must not silently disappear).
5. `weather.py` and `calendar.py`: hourly features joined on timestamp.
6. `features/build.py`: model-ready tables. Lag features only use data available at forecast
   time (for day-ahead, the most recent usable lag is 24 hours back).

Each step is a script with a Makefile target, is idempotent, and logs row counts in and out.

## 5. Models

**1. Seasonal naive baseline:** forecast = same station, same hour, one week earlier.
Also report a "historical average for this station, weekday, and hour" baseline. Every other model
must beat these to matter.

**2. Prophet:** one model per station per target, with daily and weekly seasonality, holidays,
and weather as extra regressors. Use Prophet's built-in prediction intervals.

**3. PyTorch global model:** **one** neural network trained across all 50 stations, with a learned
station embedding, so stations share patterns. Inputs: lags (24h, 48h, 168h back), rolling means,
calendar features, and weather. Output: the next 24 hours for one station (multi-output head).
Start with a small MLP; try a small LSTM or 1D-CNN only if the MLP is working and there's time.
Use early stopping on a validation window that comes before the test window in time. Fix random
seeds and log hyperparameters.

Optional, only if time allows: SARIMA on system-wide totals as a classic statistical reference.

## 6. Evaluation

**Rolling-origin backtest:** pick at least 8 test weeks spread across seasons (including summer
peak and a winter week). For each test week, train only on data before it, forecast each day
day-ahead, and score. Never tune on test weeks.

**Metrics**
- **MAE** (main metric, in bikes per hour)
- **RMSE** (punishes big misses)
- **WAPE** = total absolute error / total actual. Use this instead of MAPE, which breaks when the
  actual value is 0, which happens a lot.
- **Skill vs baseline** = 1 minus (model MAE / seasonal naive MAE). Positive means better.
- **Prediction interval coverage** for Prophet (and for PyTorch if quantile outputs are added):
  what share of actual values fall inside the 80% interval.

**Break results down by** station, hour of day, weekday vs weekend, season, and weather (rainy vs dry).

**Reporting:** `report.py` writes `reports/results.md` with a results table and saves charts to
`reports/figures/`. README numbers come only from this file.

## 7. Error analysis and takeaway

- Where does each model fail worst? (Likely: rainy days, holidays, special events, the first warm days of spring.)
- Do the neural network's gains come from busy stations or everywhere?
- Rebalancing takeaway: using forecast net flow, list the stations and hours that consistently
  drain or fill, and estimate how many bikes would need moving on a typical weekday morning.
  Keep it as a clear, defensible statement, not an overclaim.

## 8. Dashboard (Streamlit)

- Map of the 50 stations colored by tomorrow's forecast net flow.
- Pick a station: actual vs forecast for each model over a backtest week, with Prophet's interval.
- Model comparison table and skill-vs-baseline chart.
- "Rebalancing list" for the selected day.
The app reads precomputed results from disk; it never trains models on page load.

## 9. Testing and quality

- pytest: schema normalization on both formats, cleaning rules, zero-filling of empty hours,
  no-leakage check (features for day D use nothing from day D or later), metric functions
  (including WAPE with zeros), backtest window boundaries.
- ruff for lint and formatting. GitHub Actions runs lint and tests on a small sample dataset
  committed to `tests/fixtures/` (small, synthetic or heavily sampled, never the full data).

## 10. Phases

### Phase 1 (week 1): Data, EDA, baselines, backtest harness
Repo setup, download, clean, station crosswalk, hourly aggregation with zero-fill, weather and
calendar features, EDA notebook (demand by hour, weekday, season, weather; busiest stations;
net flow patterns), both baselines, backtest harness, metrics, first `results.md`.
**Done when:** `make data && make backtest` produces baseline results on all test weeks, and
leakage and zero-fill tests pass.

### Phase 2 (week 2): Prophet and PyTorch
Prophet per station with holidays and weather. PyTorch global model with station embeddings.
Both run through the same backtest. Results table compares all models.
**Done when:** results.md shows all models on identical windows, with skill vs baseline.

### Phase 3 (week 3): Error analysis, takeaway, dashboard, polish
Error analysis notebook, rebalancing takeaway, Streamlit app, README (question, data, method,
results table, key charts, takeaway, limitations, how to run), DECISIONS.md complete, CI green.
**Done when:** a stranger can clone the repo, run `make` targets, and reproduce the results table.

## 11. Resume bullets this should produce (fill in real numbers)

- Forecast day-ahead hourly demand for Boston's 50 busiest bike share stations using [N]M+ trips
  and weather data, comparing a seasonal baseline, Prophet, and a PyTorch model on a rolling backtest
- [Best model] cut forecast error by [X]% versus the seasonal baseline, with [Y]% of actuals
  inside Prophet's 80% prediction intervals
- Identified [N] stations that drain every weekday morning, estimating [Z] bikes per day to
  restock before peak hours
