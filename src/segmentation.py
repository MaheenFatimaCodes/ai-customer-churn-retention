"""
Phase — Unsupervised customer segmentation (K-Means), combined with churn probability
to produce actionable segments like "High-value but at-risk".

Run standalone: python src/segmentation.py
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from pipeline import load_features, get_X_y

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

SEGMENT_FEATURES = [
    "Tenure_in_Months", "avg_monthly_revenue", "services_subscribed",
    "Number_of_Referrals", "Total_Revenue", "refund_rate",
]

SEGMENT_LABELS = {
    # filled in dynamically based on cluster centroid characteristics
}


def label_clusters(centroids_df: pd.DataFrame) -> dict:
    """Assign a human-readable label to each cluster based on its centroid profile."""
    value_median = centroids_df["avg_monthly_revenue"].median()
    engagement_median = centroids_df["services_subscribed"].median()
    tenure_median = centroids_df["Tenure_in_Months"].median()
    refund_median = centroids_df["refund_rate"].median()

    labels = {}
    for cluster_id, row in centroids_df.iterrows():
        high_value = row["avg_monthly_revenue"] >= value_median
        high_engagement = row["services_subscribed"] >= engagement_median
        new_customer = row["Tenure_in_Months"] < tenure_median * 0.6
        high_refunds = row["refund_rate"] > refund_median

        if new_customer:
            labels[cluster_id] = "New customers"
        elif high_value and high_engagement:
            labels[cluster_id] = "Loyal high-value"
        elif high_value and not high_engagement:
            labels[cluster_id] = "High-value but at-risk"
        elif not high_value and high_engagement:
            labels[cluster_id] = "Engaged low-spend"
        elif high_refunds:
            labels[cluster_id] = "Low-engagement, frustrated"
        else:
            labels[cluster_id] = "Low-engagement"

        # de-duplicate if two clusters still land on the same label
        base_label = labels[cluster_id]
        suffix = 2
        existing = set(list(labels.values())[:-1])
        while labels[cluster_id] in existing:
            labels[cluster_id] = f"{base_label} ({suffix})"
            suffix += 1
    return labels


def run_segmentation(df: pd.DataFrame, k: int = 4, random_state: int = 42):
    X_seg = df[SEGMENT_FEATURES].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_seg)

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, clusters)

    df = df.copy()
    df["cluster"] = clusters

    centroids = df.groupby("cluster")[SEGMENT_FEATURES].mean()
    labels = label_clusters(centroids)
    df["segment"] = df["cluster"].map(labels)

    return df, kmeans, scaler, centroids, labels, sil_score


def main():
    df = load_features()
    seg_df, kmeans, scaler, centroids, labels, sil_score = run_segmentation(df)

    print(f"Silhouette score (k=4): {sil_score:.3f}")
    print("\nCluster centroids:")
    print(centroids.round(2))
    print("\nCluster -> segment labels:")
    print(labels)

    print("\nSegment sizes and churn rate:")
    summary = seg_df.groupby("segment").agg(
        customers=("churn", "count"),
        churn_rate=("churn", "mean"),
        avg_revenue=("Total_Revenue", "mean"),
    ).sort_values("churn_rate", ascending=False)
    print(summary.round(3))

    joblib.dump({"kmeans": kmeans, "scaler": scaler, "labels": labels,
                 "features": SEGMENT_FEATURES}, MODELS_DIR / "segmentation_model.pkl")

    out_path = ROOT / "data" / "processed" / "segmented_customers.csv"
    seg_df.to_csv(out_path, index=False)
    print(f"\nSaved segmented data -> {out_path}")
    print(f"Saved segmentation model -> {MODELS_DIR/'segmentation_model.pkl'}")


if __name__ == "__main__":
    main()
