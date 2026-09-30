"""
Phase — Retention Recommendation Engine + Revenue-at-Risk / Business Impact calculator.

Rule-based layer on top of the churn model: turns a probability into an action.
Thresholds are business-tunable, not hardcoded truths — documented as assumptions.

Run standalone: python src/retention_engine.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def risk_bucket(proba: float) -> str:
    if proba < 0.30:
        return "Low"
    elif proba < 0.60:
        return "Medium"
    elif proba < 0.80:
        return "High"
    return "Critical"


def recommend_action(row: pd.Series) -> str:
    """
    Rule-based retention recommendation. `row` must contain:
    churn_proba, Tenure_in_Months (or days-since-purchase proxy not available -> use tenure/refund),
    refund_rate, is_month_to_month, avg_monthly_revenue, services_subscribed
    """
    p = row["churn_proba"]
    refund_rate = row.get("refund_rate", 0)
    mtm = row.get("is_month_to_month", 0)
    revenue = row.get("avg_monthly_revenue", 0)
    services = row.get("services_subscribed", 0)

    if p > 0.75 and refund_rate > 0.02 and mtm == 1:
        return "Priority support escalation + personalized retention offer"
    if p > 0.70 and revenue > 300 and services <= 2:
        return "High-value save call: bundle discount to raise engagement"
    if p > 0.60 and mtm == 1:
        return "Contract upgrade incentive (move off month-to-month)"
    if p > 0.50 and services <= 1:
        return "Re-engagement campaign: highlight underused add-on services"
    if p > 0.30:
        return "Monitor + include in next loyalty email campaign"
    return "No action needed — low risk"


def apply_engine(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["risk_level"] = df["churn_proba"].apply(risk_bucket)
    df["recommended_action"] = df.apply(recommend_action, axis=1)
    return df


def revenue_at_risk(df: pd.DataFrame, revenue_col: str = "Total_Revenue",
                     expected_retention_rate: float = 0.15) -> dict:
    """
    Business-impact estimate. Assumptions (make these explicit to the user):
    - 'Revenue at risk' = sum of Total_Revenue for customers flagged High or Critical risk.
      This is an exposure estimate, NOT a forecast of guaranteed lost revenue.
    - 'Expected retention rate' is an assumption representing the fraction of at-risk
      revenue a retention campaign is expected to save; default 15%, override with your
      own campaign historical performance if available.
    """
    at_risk_mask = df["risk_level"].isin(["High", "Critical"])
    at_risk_customers = int(at_risk_mask.sum())
    revenue_at_risk_total = float(df.loc[at_risk_mask, revenue_col].sum())
    potential_retained = revenue_at_risk_total * expected_retention_rate

    return {
        "total_customers_scored": int(len(df)),
        "high_or_critical_risk_customers": at_risk_customers,
        "revenue_at_risk": round(revenue_at_risk_total, 2),
        "assumed_retention_rate": expected_retention_rate,
        "potential_revenue_retained_if_campaign_succeeds": round(potential_retained, 2),
    }


def main():
    df = pd.read_csv(ROOT / "data" / "processed" / "test_predictions.csv")
    # need refund_rate, is_month_to_month, avg_monthly_revenue, services_subscribed - already in features
    df = apply_engine(df)

    print("=== Risk level distribution ===")
    print(df["risk_level"].value_counts())

    print("\n=== Recommended action distribution ===")
    print(df["recommended_action"].value_counts())

    impact = revenue_at_risk(df)
    print("\n=== Revenue at risk / business impact ===")
    for k, v in impact.items():
        print(f"  {k}: {v}")

    out_path = ROOT / "data" / "processed" / "scored_customers.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved scored customers -> {out_path}")


if __name__ == "__main__":
    main()
