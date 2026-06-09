---
title: Bike Demand Forecaster
emoji: 🚲
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
---

# Bike Demand Forecaster

Forecast hourly bike-share demand from weather and calendar features. Built as the Week 4
**Agentic ML Production Sprint** capstone: cleaned data → MLflow-tracked models → Streamlit
dashboard → Docker → GitHub Actions → live on Hugging Face Spaces.

## Run locally

```bash
pip install -r requirements.txt
python -m src.train          # trains models, logs to MLflow, saves champion to models/
streamlit run app.py         # → http://localhost:8501
```

## Project layout

| Path | Purpose |
|------|---------|
| `src/data_prep.py` | Load, clean, drop leakage columns, engineer features |
| `src/train.py` | Train 2–3 regressors, log MLflow runs, save champion |
| `src/predict.py` | Load champion + return a prediction |
| `app.py` | Streamlit dashboard (EDA + prediction form) |
| `tests/test_smoke.py` | Smoke test (no leakage cols, prediction returns a float) |
| `.github/workflows/deploy-hf.yml` | Mirror repo to the HF Space on push to `main` |
| `Dockerfile` | Local container reference (HF builds its own image) |
