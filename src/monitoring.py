"""
Phase — Model Monitoring (lightweight MLOps layer).

Compares a new batch of scored customers against the training-time reference
distribution to flag data drift, prediction drift, and missing-value spikes.
This is a foundation to build on (e.g. schedule it, wire alerts to Slack/email)
rather than a full production monitoring stack.

Run standalone: python src/monitoring.py
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parent.parent
REFERENCE_PATH = ROOT / "data" / "processed" / "features.csv"
REPORTS_DIR = ROOT / "reports"

NUMERIC_DRIFT_COLS = [
    "Tenure_in_Months", "Monthly_Charge", "Total_Revenue",
    "avg_monthly_revenue", "services_subscribed",
]


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index — standard drift metric.
    <0.1 = no significant shift, 0.1-0.25 = moderate, >0.25 = major shift."""
    breakpoints = np.linspace(0, 100, bins + 1)
    cut_points = np.percentile(reference.dropna(), breakpoints)
    cut_points[0], cut_points[-1] = -np.inf, np.inf

    ref_counts = np.histogram(reference, bins=cut_points)[0] / len(reference)
    cur_counts = np.histogram(current, bins=cut_points)[0] / len(current)

    ref_counts = np.clip(ref_counts, 1e-4, None)
    cur_counts = np.clip(cur_counts, 1e-4, None)

    return float(np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts)))


def check_data_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
    results = {}
    for col in NUMERIC_DRIFT_COLS:
        if col not in current_df.columns:
            continue
        psi_score = psi(reference_df[col], current_df[col])
        ks_stat, ks_p = ks_2samp(reference_df[col], current_df[col])
        flag = "MAJOR_SHIFT" if psi_score > 0.25 else ("MODERATE_SHIFT" if psi_score > 0.1 else "STABLE")
        results[col] = {"psi": round(psi_score, 4), "ks_p_value": round(float(ks_p), 4), "status": flag}
    return results


def check_prediction_drift(reference_proba: pd.Series, current_proba: pd.Series) -> dict:
    psi_score = psi(reference_proba, current_proba)
    flag = "MAJOR_SHIFT" if psi_score > 0.25 else ("MODERATE_SHIFT" if psi_score > 0.1 else "STABLE")
    return {
        "reference_mean_churn_proba": round(float(reference_proba.mean()), 4),
        "current_mean_churn_proba": round(float(current_proba.mean()), 4),
        "psi": round(psi_score, 4),
        "status": flag,
    }


def check_missing_values(current_df: pd.DataFrame) -> dict:
    missing = current_df.isnull().mean().round(4)
    return {col: pct for col, pct in missing.items() if pct > 0}


def main():
    reference_df = pd.read_csv(REFERENCE_PATH)
    # Demo: simulate a "new batch" as a random sample (in production this would
    # be this week's freshly scored customers)
    current_df = reference_df.sample(frac=0.3, random_state=7).reset_index(drop=True)

    drift_report = {
        "data_drift": check_data_drift(reference_df, current_df),
        "missing_value_spikes": check_missing_values(current_df),
    }

    test_preds_path = ROOT / "data" / "processed" / "test_predictions.csv"
    if test_preds_path.exists():
        test_preds = pd.read_csv(test_preds_path)
        drift_report["prediction_drift"] = check_prediction_drift(
            test_preds["churn_proba"], test_preds["churn_proba"].sample(frac=0.5, random_state=1))

    print(json.dumps(drift_report, indent=2))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "monitoring_report.json", "w") as f:
        json.dump(drift_report, f, indent=2)
    print(f"\nSaved monitoring report -> {REPORTS_DIR/'monitoring_report.json'}")


if __name__ == "__main__":
    main()
