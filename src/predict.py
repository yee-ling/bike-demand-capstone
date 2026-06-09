"""
Inference: load the champion model and predict hourly demand.
=============================================================

CONTRACT — implement ``load_model`` and ``predict`` so the Streamlit app can serve
predictions. Keep the signatures.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from .data_prep import TARGET
from .train import CHAMPION_PATH

#: HF Hub model repo where the champion is uploaded.
HF_MODEL_REPO: str = "YeeLing24/bike-demand-model"


@lru_cache(maxsize=1)
def load_model(path: str = CHAMPION_PATH) -> Any:
    """Load and return the serialised champion model.

    If the file does not exist locally, downloads it from the Hugging Face
    Model Hub (``HF_MODEL_REPO``).

    Args:
        path: Path to the serialised champion (default :data:`CHAMPION_PATH`).

    Returns:
        The loaded model object.
    """
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename="champion.pkl",
            local_dir=os.path.dirname(path) or ".",
        )
    return joblib.load(path)


def predict(model: Any, inputs: Dict[str, Any]) -> float:
    """Return predicted demand for a single hour.

    ``inputs`` is a dict of raw user choices from the app (e.g. hour, weekday,
    weathersit, temp, hum, windspeed). This function must transform them into the SAME
    feature shape used at training time, then return a single non-negative number.

    Args:
        model: A loaded champion model.
        inputs: User-supplied feature values.

    Returns:
        Predicted total rentals (``cnt``) for the hour, as a non-negative float.
    """
    row = {
        "season": _season_from_month(inputs.get("mnth", 6)),
        "yr": inputs.get("yr", 1),
        "mnth": inputs.get("mnth", 6),
        "hr": inputs.get("hr", 12),
        "holiday": inputs.get("holiday", 0),
        "weekday": inputs.get("weekday", 0),
        "workingday": inputs.get("workingday", 0),
        "weathersit": inputs.get("weathersit", 1),
        "temp": inputs.get("temp", 0.5),
        "atemp": inputs.get("atemp", inputs.get("temp", 0.5)),
        "hum": inputs.get("hum", 0.5),
        "windspeed": inputs.get("windspeed", 0.2),
    }
    df = pd.DataFrame([row])

    df["dayofweek"] = df["weekday"]
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["hr_sin"] = np.sin(2 * np.pi * df["hr"] / 24)
    df["hr_cos"] = np.cos(2 * np.pi * df["hr"] / 24)
    df["mnth_sin"] = np.sin(2 * np.pi * (df["mnth"] - 1) / 12)
    df["mnth_cos"] = np.cos(2 * np.pi * (df["mnth"] - 1) / 12)

    feature_cols = [c for c in df.columns if c != TARGET]
    pred = model.predict(df[feature_cols])[0]
    return max(float(pred), 0.0)


def _season_from_month(mnth: int) -> int:
    if mnth in (12, 1, 2):
        return 1
    elif mnth in (3, 4, 5):
        return 2
    elif mnth in (6, 7, 8):
        return 3
    else:
        return 4
