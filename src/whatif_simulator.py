"""
Phase — What-if simulator.
Given a customer's current feature values and a dict of overrides, recompute
churn probability and derived features consistently, so the dashboard slider
interaction and the API give identical answers.

Run standalone: python src/whatif_simulator.py
"""
import joblib
import pandas as pd
from pathlib import Path

from pipeline import ALL_FEATURES
from feature_engineering import engineer_features

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


def load_model():
    return joblib.load(MODELS_DIR / "churn_model.pkl")


def simulate(base_row: pd.Series, overrides: dict, model) -> dict:
    """
    base_row: a raw (pre-feature-engineering) customer row, e.g. from clean_data.csv
    overrides: dict of raw column -> new value, e.g. {"Total_Refunds": 0, "Contract": "Two Year"}
    Returns before/after churn probability plus the recomputed engineered features used.
    """
    original_df = pd.DataFrame([base_row])
    modified_df = original_df.copy()
    for col, val in overrides.items():
        modified_df.loc[modified_df.index[0], col] = val

    original_feat = engineer_features(original_df)
    modified_feat = engineer_features(modified_df)

    proba_before = float(model.predict_proba(original_feat[ALL_FEATURES])[0, 1])
    proba_after = float(model.predict_proba(modified_feat[ALL_FEATURES])[0, 1])

    return {
        "churn_probability_before": round(proba_before, 4),
        "churn_probability_after": round(proba_after, 4),
        "change": round(proba_after - proba_before, 4),
        "overrides_applied": overrides,
    }


def main():
    model = load_model()
    df = pd.read_csv(ROOT / "data" / "processed" / "clean_data.csv")
    sample = df.iloc[0]

    print("Base customer:", sample["Customer_ID"])
    print("Current: Complaints/refunds=%.2f, Tenure=%d, Contract=%s" % (
        sample["Total_Refunds"], sample["Tenure_in_Months"], sample["Contract"]))

    result = simulate(sample, {
        "Total_Refunds": 0,
        "Contract": "Two Year",
        "Online_Security": "Yes",
    }, model)

    print("\n=== What-if result ===")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
