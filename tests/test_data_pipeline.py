"""
Run: pytest tests/ -v
"""
import sys
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_preprocessing import clean, load_raw
from feature_engineering import engineer_features, tenure_bucket


@pytest.fixture(scope="module")
def raw_df():
    return load_raw()


@pytest.fixture(scope="module")
def clean_df(raw_df):
    return clean(raw_df)


def test_clean_drops_joined_customers(clean_df):
    assert "Joined" not in clean_df["Customer_Status"].unique()


def test_clean_creates_binary_target(clean_df):
    assert set(clean_df["churn"].unique()) <= {0, 1}


def test_clean_no_missing_in_service_columns(clean_df):
    service_cols = ["Internet_Type", "Online_Security", "Multiple_Lines", "Value_Deal"]
    for col in service_cols:
        assert clean_df[col].isnull().sum() == 0, f"{col} still has nulls after cleaning"


def test_tenure_bucket_boundaries():
    assert tenure_bucket(3) == "0-6mo"
    assert tenure_bucket(6) == "0-6mo"
    assert tenure_bucket(7) == "6-12mo"
    assert tenure_bucket(24) == "1-2yr"
    assert tenure_bucket(25) == "2-4yr"
    assert tenure_bucket(60) == "4yr+"


def test_feature_engineering_no_nulls_in_new_features(clean_df):
    feat_df = engineer_features(clean_df)
    new_cols = ["avg_monthly_revenue", "refund_rate", "services_subscribed",
                "has_internet", "is_month_to_month", "revenue_per_referral"]
    for col in new_cols:
        assert feat_df[col].isnull().sum() == 0, f"{col} has unexpected nulls"


def test_avg_monthly_revenue_matches_manual_calc(clean_df):
    feat_df = engineer_features(clean_df)
    row = feat_df.iloc[0]
    expected = row["Total_Revenue"] / max(row["Tenure_in_Months"], 1)
    assert abs(row["avg_monthly_revenue"] - expected) < 1e-6


def test_services_subscribed_range(clean_df):
    feat_df = engineer_features(clean_df)
    assert feat_df["services_subscribed"].between(0, 8).all()
