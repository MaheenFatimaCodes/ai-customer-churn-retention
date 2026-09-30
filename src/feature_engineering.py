"""
Phase B5 — Feature engineering.
Turns raw billing/service columns into signals that actually explain churn behavior.

Run standalone: python src/feature_engineering.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

IN_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "clean_data.csv"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "features.csv"

SERVICE_FLAGS = [
    "Online_Security", "Online_Backup", "Device_Protection_Plan",
    "Premium_Support", "Streaming_TV", "Streaming_Movies",
    "Streaming_Music", "Unlimited_Data",
]


def tenure_bucket(months: int) -> str:
    if months <= 6:
        return "0-6mo"
    if months <= 12:
        return "6-12mo"
    if months <= 24:
        return "1-2yr"
    if months <= 48:
        return "2-4yr"
    return "4yr+"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    safe_tenure = df["Tenure_in_Months"].clip(lower=1)

    df["avg_monthly_revenue"] = df["Total_Revenue"] / safe_tenure
    df["refund_rate"] = df["Total_Refunds"] / (df["Total_Charges"] + 1)
    df["services_subscribed"] = (df[SERVICE_FLAGS] == "Yes").sum(axis=1)
    df["has_internet"] = (df["Internet_Service"] == "Yes").astype(int)
    df["is_month_to_month"] = (df["Contract"] == "Month-to-Month").astype(int)
    df["long_distance_share"] = df["Total_Long_Distance_Charges"] / (df["Total_Revenue"] + 1)
    df["extra_data_share"] = df["Total_Extra_Data_Charges"] / (df["Total_Revenue"] + 1)
    df["revenue_per_referral"] = df["Total_Revenue"] / (df["Number_of_Referrals"] + 1)
    df["tenure_bucket"] = df["Tenure_in_Months"].apply(tenure_bucket)
    df["has_value_deal"] = (df["Value_Deal"] != "No Deal").astype(int)
    df["is_paperless"] = (df["Paperless_Billing"] == "Yes").astype(int)

    return df


def main():
    df = pd.read_csv(IN_PATH)
    feat_df = engineer_features(df)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(OUT_PATH, index=False)
    new_cols = [
        "avg_monthly_revenue", "refund_rate", "services_subscribed", "has_internet",
        "is_month_to_month", "long_distance_share", "extra_data_share",
        "revenue_per_referral", "tenure_bucket", "has_value_deal", "is_paperless",
    ]
    print(f"Saved feature-engineered data -> {OUT_PATH}  shape={feat_df.shape}")
    print("\nNew feature summary:")
    print(feat_df[new_cols].describe(include="all").T)


if __name__ == "__main__":
    main()
