"""
Phase D — Exploratory Data Analysis.
Produces summary stats + saved figures in reports/figures/, and prints
statistical test results comparing churned vs stayed customers.

Run standalone: python src/eda.py
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

IN_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "features.csv"
FIG_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"

sns.set_style("whitegrid")


def univariate(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    df["churn"].value_counts().rename({0: "Stayed", 1: "Churned"}).plot(
        kind="bar", ax=axes[0], color=["#4C72B0", "#C44E52"])
    axes[0].set_title("Churn class balance")
    df["Tenure_in_Months"].hist(bins=30, ax=axes[1], color="#55A868")
    axes[1].set_title("Tenure distribution (months)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_univariate.png", dpi=120)
    plt.close()


def bivariate(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    sns.boxplot(data=df, x="churn", y="Tenure_in_Months", ax=axes[0])
    axes[0].set_xticklabels(["Stayed", "Churned"])
    axes[0].set_title("Tenure vs Churn")

    contract_churn = pd.crosstab(df["Contract"], df["churn"], normalize="index") * 100
    contract_churn[1].sort_values().plot(kind="barh", ax=axes[1], color="#C44E52")
    axes[1].set_title("Churn rate % by Contract type")
    axes[1].set_xlabel("Churn rate (%)")

    sns.boxplot(data=df, x="churn", y="Monthly_Charge", ax=axes[2])
    axes[2].set_xticklabels(["Stayed", "Churned"])
    axes[2].set_title("Monthly Charge vs Churn")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_bivariate.png", dpi=120)
    plt.close()

    # Does churn increase as days-since-purchase-proxy (tenure bucket, inverse) increases?
    bucket_order = ["0-6mo", "6-12mo", "1-2yr", "2-4yr", "4yr+"]
    rate_by_bucket = df.groupby("tenure_bucket")["churn"].mean().reindex(bucket_order)
    print("\nChurn rate by tenure bucket:")
    print((rate_by_bucket * 100).round(1))


def multivariate(df: pd.DataFrame):
    # high-value + low-engagement combination
    high_value = df["avg_monthly_revenue"] > df["avg_monthly_revenue"].median()
    low_engagement = df["services_subscribed"] <= df["services_subscribed"].median()
    seg = pd.Series("other", index=df.index)
    seg[high_value & low_engagement] = "high_value_low_engagement"
    seg[high_value & ~low_engagement] = "high_value_high_engagement"
    seg[~high_value & low_engagement] = "low_value_low_engagement"
    seg[~high_value & ~low_engagement] = "low_value_high_engagement"

    churn_by_seg = df.groupby(seg)["churn"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    print("\nChurn rate by value x engagement segment:")
    print(churn_by_seg)

    fig, ax = plt.subplots(figsize=(7, 4))
    (churn_by_seg["mean"] * 100).plot(kind="bar", ax=ax, color="#8172B2")
    ax.set_ylabel("Churn rate (%)")
    ax.set_title("Churn rate: value x engagement segments")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_multivariate.png", dpi=120)
    plt.close()


def statistical_tests(df: pd.DataFrame):
    print("\n=== Statistical significance: churned vs stayed ===")
    numeric_cols = ["Tenure_in_Months", "Monthly_Charge", "Total_Revenue",
                     "services_subscribed", "avg_monthly_revenue", "refund_rate"]
    for col in numeric_cols:
        churned = df.loc[df["churn"] == 1, col]
        stayed = df.loc[df["churn"] == 0, col]
        t_stat, p_val = stats.mannwhitneyu(churned, stayed, alternative="two-sided")
        sig = "significant (p<0.05)" if p_val < 0.05 else "not significant"
        print(f"{col:25s} Mann-Whitney U p={p_val:.2e}  -> {sig}  "
              f"(churned mean={churned.mean():.2f}, stayed mean={stayed.mean():.2f})")

    # Chi-square for Contract type
    contingency = pd.crosstab(df["Contract"], df["churn"])
    chi2, p, dof, _ = stats.chi2_contingency(contingency)
    print(f"\n{'Contract (chi-square)':25s} p={p:.2e} -> "
          f"{'significant' if p < 0.05 else 'not significant'}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(IN_PATH)
    univariate(df)
    bivariate(df)
    multivariate(df)
    statistical_tests(df)
    print(f"\nFigures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
