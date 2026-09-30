"""
Phase — ML training, cross-validation, hyperparameter tuning, evaluation.

Trains Logistic Regression (baseline), Random Forest, and XGBoost, compares
them on held-out test data using metrics appropriate for an imbalanced
churn problem (recall matters — a missed churner is costly), then saves
the best model + preprocessing pipeline + metrics to models/.

Run standalone: python src/train_models.py
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, recall_score,
    precision_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from xgboost import XGBClassifier

from pipeline import load_features, build_preprocessor, get_X_y, ALL_FEATURES

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
RANDOM_STATE = 42


def make_splits(X, y):
    # 70 / 15 / 15 train / val / test, stratified on churn (imbalanced target)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE)
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate(name, model, X, y, threshold=0.5):
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)
    metrics = {
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "f1": f1_score(y, preds),
        "recall": recall_score(y, preds),
        "precision": precision_score(y, preds),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
    }
    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k:10s}: {v:.4f}")
    print(f"  confusion_matrix (rows=actual, cols=pred) [stayed, churned]:")
    for row in metrics["confusion_matrix"]:
        print(f"    {row}")
    return metrics


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_features()
    X, y = get_X_y(df)
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Churn rate — train: {y_train.mean():.3f}, val: {y_val.mean():.3f}, test: {y_test.mean():.3f}")

    preprocessor = build_preprocessor()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    fitted_models = {}

    # 1. Baseline: Logistic Regression -----------------------------------
    logreg = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    logreg.fit(X_train, y_train)
    cv_auc = cross_val_score(logreg, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"\nLogistic Regression 5-fold CV ROC-AUC: {cv_auc.mean():.4f} (+/- {cv_auc.std():.4f})")
    results["logistic_regression"] = evaluate("Logistic Regression (val)", logreg, X_val, y_val)
    fitted_models["logistic_regression"] = logreg

    # 2. Random Forest with light hyperparameter search -------------------
    rf_pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    rf_param_dist = {
        "clf__n_estimators": [200, 300, 400, 500],
        "clf__max_depth": [6, 10, 14, None],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
    }
    rf_search = RandomizedSearchCV(
        rf_pipe, rf_param_dist, n_iter=12, scoring="roc_auc", cv=cv,
        random_state=RANDOM_STATE, n_jobs=-1)
    rf_search.fit(X_train, y_train)
    print(f"\nRandom Forest best CV ROC-AUC: {rf_search.best_score_:.4f}")
    print(f"Random Forest best params: {rf_search.best_params_}")
    best_rf = rf_search.best_estimator_
    results["random_forest"] = evaluate("Random Forest (val)", best_rf, X_val, y_val)
    fitted_models["random_forest"] = best_rf

    # 3. XGBoost with light hyperparameter search --------------------------
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", XGBClassifier(
            eval_metric="logloss", scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    xgb_param_dist = {
        "clf__n_estimators": [150, 250, 350],
        "clf__max_depth": [3, 4, 5, 6],
        "clf__learning_rate": [0.03, 0.05, 0.1],
        "clf__subsample": [0.7, 0.85, 1.0],
        "clf__colsample_bytree": [0.7, 0.85, 1.0],
    }
    xgb_search = RandomizedSearchCV(
        xgb_pipe, xgb_param_dist, n_iter=15, scoring="roc_auc", cv=cv,
        random_state=RANDOM_STATE, n_jobs=-1)
    xgb_search.fit(X_train, y_train)
    print(f"\nXGBoost best CV ROC-AUC: {xgb_search.best_score_:.4f}")
    print(f"XGBoost best params: {xgb_search.best_params_}")
    best_xgb = xgb_search.best_estimator_
    results["xgboost"] = evaluate("XGBoost (val)", best_xgb, X_val, y_val)
    fitted_models["xgboost"] = best_xgb

    # --- Model comparison & selection (val ROC-AUC, tie-break recall) ---
    comparison = pd.DataFrame(results).T[["roc_auc", "pr_auc", "f1", "recall", "precision"]]
    comparison = comparison.sort_values("roc_auc", ascending=False)
    print("\n=== MODEL COMPARISON (validation set) ===")
    print(comparison.round(4))

    best_name = comparison.index[0]
    best_model = fitted_models[best_name]
    print(f"\nSelected model: {best_name}")

    # --- Final evaluation on untouched test set ---
    test_metrics = evaluate(f"{best_name} (FINAL - test set)", best_model, X_test, y_test)

    # Threshold sweep to document business-tunable risk cutoffs
    proba_test = best_model.predict_proba(X_test)[:, 1]
    print("\nRisk bucket sizes on test set (proba thresholds 0.3/0.6/0.8):")
    risk = pd.cut(proba_test, bins=[0, 0.3, 0.6, 0.8, 1.0],
                   labels=["Low", "Medium", "High", "Critical"], include_lowest=True)
    print(pd.Series(risk).value_counts().sort_index())

    # Save artifacts
    joblib.dump(best_model, MODELS_DIR / "churn_model.pkl")
    joblib.dump(fitted_models, MODELS_DIR / "all_models.pkl")
    with open(MODELS_DIR / "feature_list.json", "w") as f:
        json.dump(ALL_FEATURES, f, indent=2)

    metrics_out = {
        "model_comparison_val": comparison.round(4).to_dict(orient="index"),
        "best_model": best_name,
        "test_metrics": test_metrics,
        "logreg_cv_auc_mean": float(cv_auc.mean()),
        "rf_best_params": rf_search.best_params_,
        "rf_cv_auc": float(rf_search.best_score_),
        "xgb_best_params": xgb_search.best_params_,
        "xgb_cv_auc": float(xgb_search.best_score_),
    }
    with open(REPORTS_DIR / "model_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)

    # Save test split for downstream SHAP / segmentation / dashboard demo
    X_test.assign(churn=y_test.values, churn_proba=proba_test).to_csv(
        Path(__file__).resolve().parent.parent / "data" / "processed" / "test_predictions.csv", index=False)

    print(f"\nSaved best model -> {MODELS_DIR/'churn_model.pkl'}")
    print(f"Saved metrics -> {REPORTS_DIR/'model_metrics.json'}")


if __name__ == "__main__":
    main()
