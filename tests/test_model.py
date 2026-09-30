import sys
from pathlib import Path
import joblib
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pipeline import ALL_FEATURES, load_features
from feature_engineering import engineer_features
from retention_engine import risk_bucket, recommend_action

MODEL_PATH = ROOT / "models" / "churn_model.pkl"


@pytest.fixture(scope="module")
def model():
    if not MODEL_PATH.exists():
        pytest.skip("Model not trained yet — run src/train_models.py first")
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def sample_features():
    df = load_features()
    return df.iloc[:5]


def test_model_predicts_probability_in_range(model, sample_features):
    proba = model.predict_proba(sample_features[ALL_FEATURES])[:, 1]
    assert (proba >= 0).all() and (proba <= 1).all()


def test_model_output_shape_matches_input(model, sample_features):
    proba = model.predict_proba(sample_features[ALL_FEATURES])[:, 1]
    assert len(proba) == len(sample_features)


def test_risk_bucket_thresholds():
    assert risk_bucket(0.1) == "Low"
    assert risk_bucket(0.29) == "Low"
    assert risk_bucket(0.30) == "Medium"
    assert risk_bucket(0.59) == "Medium"
    assert risk_bucket(0.60) == "High"
    assert risk_bucket(0.79) == "High"
    assert risk_bucket(0.80) == "Critical"
    assert risk_bucket(0.99) == "Critical"


def test_recommend_action_returns_string_for_all_risk_levels():
    for p in [0.05, 0.35, 0.55, 0.65, 0.85]:
        row = {"churn_proba": p, "refund_rate": 0.03, "is_month_to_month": 1,
               "avg_monthly_revenue": 400, "services_subscribed": 1}
        action = recommend_action(row)
        assert isinstance(action, str) and len(action) > 0


def test_month_to_month_increases_predicted_risk(model):
    """Sanity check: switching a customer to month-to-month should not decrease
    predicted churn probability, given everything else held constant — this is
    a known strong driver validated in EDA (chi-square p<0.001)."""
    df = load_features().iloc[[0]].copy()
    df_mtm = df.copy()
    df_mtm["Contract"] = "Month-to-Month"
    df_mtm["is_month_to_month"] = 1
    df_two_year = df.copy()
    df_two_year["Contract"] = "Two Year"
    df_two_year["is_month_to_month"] = 0

    p_mtm = model.predict_proba(df_mtm[ALL_FEATURES])[0, 1]
    p_two_year = model.predict_proba(df_two_year[ALL_FEATURES])[0, 1]
    assert p_mtm >= p_two_year
