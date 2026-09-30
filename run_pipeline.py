"""
Runs the entire pipeline end-to-end, in order.

Usage: python run_pipeline.py
"""
import subprocess
import sys
from pathlib import Path

STEPS = [
    ("Data cleaning", "src/data_preprocessing.py"),
    ("Feature engineering", "src/feature_engineering.py"),
    ("EDA", "src/eda.py"),
    ("Model training (LogReg, RF, XGBoost + tuning)", "src/train_models.py"),
    ("SHAP explainability", "src/explain.py"),
    ("Customer segmentation", "src/segmentation.py"),
    ("Retention engine + revenue at risk", "src/retention_engine.py"),
    ("Model monitoring (drift check)", "src/monitoring.py"),
]

ROOT = Path(__file__).resolve().parent


def main():
    for label, script in STEPS:
        print(f"\n{'='*70}\n▶ {label}\n{'='*70}")
        result = subprocess.run([sys.executable, script], cwd=ROOT)
        if result.returncode != 0:
            print(f"\n❌ Step failed: {label}. Stopping.")
            sys.exit(1)
    print("\n✅ Pipeline complete. Now run:")
    print("   streamlit run app/streamlit_app.py   (dashboard)")
    print("   uvicorn app.api:app --reload         (API)")


if __name__ == "__main__":
    main()
