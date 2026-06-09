# Capstone Dataset — Bike Sharing Demand (Hourly)

You will forecast **hourly bike-rental demand** from weather and calendar features.
This is a real, genuinely messy time-series regression problem — perfect for practising
cleaning, feature engineering, and (critically) **avoiding data leakage**.

## Get the full dataset (do this in Phase 1)

| Source | Link |
|--------|------|
| UCI Machine Learning Repository | https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset |
| Kaggle mirror | https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset |

The UCI download is a ZIP containing **`hour.csv`** (use this — ~17,379 hourly records,
2011–2012) and `day.csv` (daily aggregates — ignore). Download `hour.csv` and place it in
this `data/` folder.

```bash
# Example (UCI direct):
curl -L -o bike.zip https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip
unzip bike.zip hour.csv -d .
```

## Start coding immediately with the bundled sample

`bike_sharing_hourly_sample.csv` (505 rows) ships in this folder so your pipeline can run
**before** the full download finishes. It uses the **exact same schema** as the real
`hour.csv`, including the leakage columns and some injected messiness (missing `hum`/`windspeed`
values and one duplicate row), so cleaning code written against it works on the real file too.

> ⚠️ The sample is **synthetic, seeded data for smoke-testing only**. Train and evaluate your
> final model on the real `hour.csv` from UCI — never report metrics from the sample.

## Schema

| Column | Type | Description |
|--------|------|-------------|
| `instant` | int | Record index (drop — it's just a row id) |
| `dteday` | date | Date (`YYYY-MM-DD`) — parse this for datetime features |
| `season` | int | 1=winter, 2=spring, 3=summer, 4=fall |
| `yr` | int | Year (0=2011, 1=2012) |
| `mnth` | int | Month (1–12) |
| `hr` | int | Hour (0–23) |
| `holiday` | int | 1 if the day is a holiday |
| `weekday` | int | Day of week (0=Sunday … 6=Saturday) |
| `workingday` | int | 1 if a working day (not weekend/holiday) |
| `weathersit` | int | 1=clear, 2=mist, 3=light rain/snow, 4=heavy rain/snow |
| `temp` | float | Normalised temperature (0–1) |
| `atemp` | float | Normalised "feels-like" temperature (0–1) |
| `hum` | float | Normalised humidity (0–1) |
| `windspeed` | float | Normalised wind speed (0–1) |
| `casual` | int | Count of casual (non-registered) riders |
| `registered` | int | Count of registered riders |
| `cnt` | int | **TARGET** — total rentals this hour |

## 🚨 The leakage trap — read this before you train

By construction:

```
casual + registered = cnt
```

`casual` and `registered` are **components of the target**, recorded *after* the hour is over.
If you leave them in your feature set, your model will "predict" `cnt` with near-perfect R²
(~1.0) — because it is literally adding two numbers, not learning demand. That model is useless
in production, where you must forecast `cnt` for a *future* hour you have no rider counts for yet.

**Required:** drop `casual` and `registered` (and `instant`) before training. Your `data_prep`
step must remove them, and your evaluation checklist explicitly verifies this. A suspiciously
perfect R² is the tell-tale sign you left them in.

## Suggested engineered features

- Cyclical encodings of `hr` and `mnth` (`sin`/`cos`) so 23:00 is "next to" 00:00.
- `dayofweek`, `is_weekend` from `dteday`.
- Interaction or binned features for `temp` × `hr` (rush-hour demand depends on weather).
