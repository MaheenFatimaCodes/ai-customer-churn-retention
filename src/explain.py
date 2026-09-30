"""
Phase — Explainable AI (SHAP).
Generates global feature importance + per-customer explanations.

Run standalone:
    python src/explain.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import shap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path

from pipeline import load_features, get_X_y


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"


# ============================================================
# FEATURE NAMES
# ============================================================

def get_feature_names(preprocessor):
    """Return names of all transformed features."""

    num_features = preprocessor.transformers_[0][2]

    cat_encoder = preprocessor.transformers_[1][1]

    cat_features = list(
        cat_encoder.get_feature_names_out(
            preprocessor.transformers_[1][2]
        )
    )

    return list(num_features) + cat_features


# ============================================================
# BUILD SHAP EXPLAINER
# ============================================================

def build_explainer():
    """Load trained model and create SHAP TreeExplainer."""

    model = joblib.load(
        MODELS_DIR / "churn_model.pkl"
    )

    preprocessor = model.named_steps["prep"]
    clf = model.named_steps["clf"]

    df = load_features()

    X, y = get_X_y(df)

    X_transformed = preprocessor.transform(X)

    feature_names = get_feature_names(
        preprocessor
    )

    explainer = shap.TreeExplainer(clf)

    return (
        explainer,
        preprocessor,
        feature_names,
        X,
        X_transformed
    )


# ============================================================
# HANDLE DIFFERENT SHAP OUTPUT FORMATS
# ============================================================

def get_churn_shap_values(shap_values):
    """
    Normalize SHAP output for binary classification.

    We use class 1 = churn.

    Depending on the SHAP version/model,
    shap_values may be:

    1. list:
       [class_0_values, class_1_values]

    2. 3D ndarray:
       (samples, features, classes)

    3. 2D ndarray:
       (samples, features)
    """

    # Older SHAP versions:
    # [class_0, class_1]
    if isinstance(shap_values, list):

        if len(shap_values) == 2:
            return np.asarray(shap_values[1])

        return np.asarray(shap_values[0])

    shap_values = np.asarray(shap_values)

    # Newer SHAP format:
    # (samples, features, classes)
    if shap_values.ndim == 3:

        if shap_values.shape[2] == 2:
            return shap_values[:, :, 1]

        raise ValueError(
            f"Unexpected number of SHAP classes: "
            f"{shap_values.shape[2]}"
        )

    # Standard format:
    # (samples, features)
    if shap_values.ndim == 2:
        return shap_values

    raise ValueError(
        f"Unexpected SHAP output shape: "
        f"{shap_values.shape}"
    )


# ============================================================
# GLOBAL SHAP IMPORTANCE
# ============================================================

def global_importance(
    explainer,
    X_transformed,
    feature_names,
    sample_size=1000
):
    """Generate global SHAP feature importance."""

    rng = np.random.default_rng(42)

    idx = rng.choice(
        X_transformed.shape[0],
        size=min(
            sample_size,
            X_transformed.shape[0]
        ),
        replace=False
    )

    sample = X_transformed[idx]

    if hasattr(sample, "toarray"):
        sample = sample.toarray()

    # Calculate SHAP values
    raw_shap_values = explainer.shap_values(
        sample
    )

    # Keep class 1 = churn
    shap_values = get_churn_shap_values(
        raw_shap_values
    )

    print(
        "SHAP values shape:",
        shap_values.shape
    )

    print(
        "Number of feature names:",
        len(feature_names)
    )

    # Mean absolute SHAP value
    mean_abs = np.abs(
        shap_values
    ).mean(axis=0)

    print(
        "Mean absolute SHAP shape:",
        mean_abs.shape
    )

    # Safety check
    if len(mean_abs) != len(feature_names):

        raise ValueError(
            f"Feature mismatch: SHAP produced "
            f"{len(mean_abs)} feature values, "
            f"but feature_names contains "
            f"{len(feature_names)} names."
        )

    # Create importance Series
    importance = pd.Series(
        mean_abs,
        index=feature_names
    ).sort_values(
        ascending=False
    )

    # ========================================================
    # PLOT
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    importance.head(15).sort_values().plot(
        kind="barh",
        ax=ax,
        color="#4C72B0"
    )

    ax.set_title(
        "Top 15 churn drivers "
        "(mean |SHAP value|)"
    )

    ax.set_xlabel(
        "Mean |SHAP value| "
        "(impact on churn probability)"
    )

    plt.tight_layout()

    plt.savefig(
        FIG_DIR / "04_shap_global_importance.png",
        dpi=120
    )

    plt.close()

    return importance


# ============================================================
# SINGLE CUSTOMER EXPLANATION
# ============================================================

def explain_customer(
    explainer,
    preprocessor,
    feature_names,
    customer_row: pd.DataFrame,
    top_n=5
):
    """
    Return the top risk-increasing and
    risk-decreasing factors for one customer.
    """

    X_t = preprocessor.transform(
        customer_row
    )

    if hasattr(X_t, "toarray"):
        X_t = X_t.toarray()

    raw_shap_values = explainer.shap_values(
        X_t
    )

    # Keep class 1 = churn
    shap_values = get_churn_shap_values(
        raw_shap_values
    )

    shap_vals = shap_values[0]

    # Safety check
    if len(shap_vals) != len(feature_names):

        raise ValueError(
            f"Customer SHAP mismatch: "
            f"{len(shap_vals)} SHAP values vs "
            f"{len(feature_names)} feature names."
        )

    contrib = pd.Series(
        shap_vals,
        index=feature_names
    ).sort_values(
        ascending=False
    )

    # Positive = increases churn risk
    increasing = contrib[
        contrib > 0
    ].head(top_n)

    # Negative = decreases churn risk
    decreasing = contrib[
        contrib < 0
    ].tail(top_n).sort_values()

    return {
        "increasing_risk": [
            {
                "feature": feature,
                "impact": round(
                    float(value),
                    4
                )
            }
            for feature, value
            in increasing.items()
        ],

        "decreasing_risk": [
            {
                "feature": feature,
                "impact": round(
                    float(value),
                    4
                )
            }
            for feature, value
            in decreasing.items()
        ]
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # Create figures directory
    FIG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Build SHAP explainer
    (
        explainer,
        preprocessor,
        feature_names,
        X,
        X_transformed
    ) = build_explainer()

    # ========================================================
    # GLOBAL IMPORTANCE
    # ========================================================

    importance = global_importance(
        explainer,
        X_transformed,
        feature_names
    )

    print(
        "\n=== Top 15 global churn drivers ==="
    )

    print(
        importance.head(15).round(4)
    )

    # Save CSV
    importance.to_csv(
        ROOT
        / "reports"
        / "shap_global_importance.csv"
    )

    # ========================================================
    # SINGLE CUSTOMER EXPLANATION
    # ========================================================

    sample_customer = X.iloc[[0]]

    explanation = explain_customer(
        explainer,
        preprocessor,
        feature_names,
        sample_customer
    )

    print(
        "\n=== Example single-customer "
        "explanation (row 0) ==="
    )

    print(
        json.dumps(
            explanation,
            indent=2
        )
    )

    # ========================================================
    # SAVE EXPLAINER
    # ========================================================

    joblib.dump(
        {
            "explainer": explainer,
            "feature_names": feature_names
        },
        MODELS_DIR / "shap_explainer.pkl"
    )

    print(
        f"\nSaved SHAP explainer -> "
        f"{MODELS_DIR / 'shap_explainer.pkl'}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()