# Bike Demand Forecaster — Model Card

## Purpose
Forecasts hourly total bike-rental demand (`cnt`) for a city bike-share system using weather and calendar features. The model is designed for operational rebalancing: predicting demand before each hour so bikes can be redistributed to stations ahead of the next rush.

## Champion Model
The champion is selected by the lowest **RMSE** on a held-out test set (the last 20% of the chronologically sorted data).

| Metric | Value |
|--------|-------|
| Model | *(filled by training)* |
| RMSE  | *(filled by training)* |
| MAE   | *(filled by training)* |
| R²    | *(filled by training)* |

*Run `python -m src.train` to populate the actual metrics.*

## Features
- **Calendar:** hour, month, weekday, working day indicator, holiday flag, season
- **Cyclical:** sin/cos encodings of hour and month to preserve circular proximity
- **Weather:** normalised temperature, feels-like temperature, humidity, wind speed, weather situation (clear/mist/rain/snow)
- **Derived:** day of week, weekend flag

Leakage columns (`casual`, `registered`, `instant`) are dropped before training.

## Training Data
UCI Bike Sharing Dataset (hourly) — ~17,400 records from 2011–2012. Time-ordered split: 80% train, 20% test.

## Limitations
- Only trained on 2 years of data from a single city; may not generalise to other cities or time periods.
- Weather features are normalised (0–1) and the model does not ingest raw weather forecasts.
- No external factors (events, holidays of specific cities, or real-time station-level data).

## Agent Reflection
I worked with an AI coding agent (opencode) to implement the full pipeline from scaffold stubs to deployed app. The agent handled the boilerplate — data cleaning, feature engineering, MLflow wiring, and the Streamlit UI — correctly on the first pass. The main corrections I needed were around the inference feature alignment: ensuring the `predict` function re-applied the exact same transformations as training, and making sure the Streamlit form collected enough inputs to match the training feature set. The agent's speed let me iterate fast, but reviewing the output for correctness was essential — especially around the leakage trap, which the agent handled correctly per the brief but needed manual confirmation.
