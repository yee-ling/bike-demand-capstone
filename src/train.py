"""
Model training with MLflow experiment tracking.
================================================

CONTRACT — implement ``train_and_log`` so it trains several models, records each as an
MLflow run, selects a champion, and serialises it. Keep the signature.

Run as a module:  ``python -m src.train``
"""
from __future__ import annotations

import os
from typing import Optional

import joblib
import mlflow
import numpy as np
import pandas as pd
from huggingface_hub import HfApi
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data_prep import TARGET, add_features, load_and_clean

#: Where the champion model is written.
CHAMPION_PATH: str = "models/champion.pkl"

#: Columns to exclude from features (target + non-predictive).
NON_FEATURE_COLUMNS: list[str] = [TARGET, "dteday"]


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


def train_and_log(df: pd.DataFrame, champion_path: str = CHAMPION_PATH) -> str:
    """Train 2-3 regressors, log them to MLflow, and save the champion.

    Must:
      - split features/target with NO leakage (``TARGET`` and leakage cols excluded from X),
      - train at least TWO models (e.g. LinearRegression, RandomForestRegressor,
        GradientBoostingRegressor),
      - for EACH model open an ``mlflow.start_run`` and log params + metrics
        (RMSE, MAE, R2) + the model artifact,
      - select the champion by the best held-out RMSE,
      - serialise the champion (e.g. ``joblib.dump``) to ``champion_path``.

    Args:
        df: Feature-engineered DataFrame (post :func:`add_features`).
        champion_path: Destination path for the serialised champion.

    Returns:
        The path the champion was written to.
    """
    os.makedirs(os.path.dirname(champion_path) or ".", exist_ok=True)

    df_sorted = df.sort_values(["dteday", "hr"]).reset_index(drop=True)
    feature_cols = _get_feature_cols(df_sorted)
    X = df_sorted[feature_cols]
    y = df_sorted[TARGET]

    split_idx = int(len(df_sorted) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    mlflow.set_experiment("bike_demand_forecast")
    best_rmse = float("inf")
    best_model = None
    best_name = ""

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            mae = float(mean_absolute_error(y_test, y_pred))
            r2 = float(r2_score(y_test, y_pred))

            mlflow.log_params(
                {"model_type": name, **_get_model_params(model)}
            )
            mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
            mlflow.sklearn.log_model(model, artifact_path="model")

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_name = name

    joblib.dump(best_model, champion_path, compress=3)
    print(f"Champion: {best_name} (RMSE={best_rmse:.2f}) -> {champion_path}")

    token = os.environ.get("HF_TOKEN")
    if token:
        repo_id = "YeeLing24/bike-demand-model"
        api = HfApi(token=token)
        api.create_repo(repo_id, repo_type="model", exist_ok=True)
        api.upload_file(
            path_or_fileobj=champion_path,
            path_in_repo="champion.pkl",
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"Uploaded champion to HF Hub: {repo_id}")
    else:
        print("HF_TOKEN not set — skipping HF Hub upload")

    return champion_path


def _get_model_params(model) -> dict:
    params = model.get_params()
    return {k: str(v) if isinstance(v, (list, dict)) else v for k, v in params.items()}


def main(data_path: Optional[str] = None) -> str:
    """Entry point: clean -> feature-engineer -> train_and_log."""
    path = data_path or "data/hour.csv"
    df = add_features(load_and_clean(path))
    return train_and_log(df)


if __name__ == "__main__":
    print(f"Champion saved to: {main()}")
