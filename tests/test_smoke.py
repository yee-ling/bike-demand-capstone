"""
Smoke tests — keep the CI workflow honest.
===========================================

CONTRACT — make these real assertions pass once your pipeline works. They are intentionally
minimal: at least prove (1) cleaning removes leakage columns, and (2) prediction returns a
non-negative float. Add more as you go.
"""
import pytest

from src.data_prep import LEAKAGE_COLUMNS, load_and_clean

SAMPLE = "data/bike_sharing_hourly_sample.csv"


def test_no_leakage_columns_after_cleaning():
    """Cleaned data must not contain any leakage columns."""
    df = load_and_clean(SAMPLE)
    for col in LEAKAGE_COLUMNS:
        assert col not in df.columns, f"leakage column {col!r} survived cleaning"


def test_prediction_is_non_negative_float():
    """A prediction should be a sane, non-negative number."""
    from src.predict import load_model, predict

    model = load_model()
    value = predict(model, {"hr": 8, "weekday": 1, "weathersit": 1,
                             "temp": 0.5, "hum": 0.5, "windspeed": 0.2})
    assert isinstance(value, float)
    assert value >= 0

def test_data_prep_deterministic():
    """Cleaning is deterministic — same shape on re-run."""
    from src.data_prep import load_and_clean
    df1 = load_and_clean("data/bike_sharing_hourly_sample.csv")
    df2 = load_and_clean("data/bike_sharing_hourly_sample.csv")
    assert df1.shape == df2.shape
    assert list(df1.columns) == list(df2.columns)

def test_add_features_produces_expected_columns():
    """Feature engineering adds the expected cyclical columns."""
    from src.data_prep import add_features, load_and_clean
    df = add_features(load_and_clean("data/bike_sharing_hourly_sample.csv"))
    for col in ("dayofweek", "is_weekend", "hr_sin", "hr_cos", "mnth_sin", "mnth_cos"):
        assert col in df.columns
