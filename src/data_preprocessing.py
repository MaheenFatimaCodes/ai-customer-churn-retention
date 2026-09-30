"""
Phase B/C — Data collection, quality checks, and cleaning.

Run standalone:  python src/data_preprocessing.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "Customer_Data.csv"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "clean_data.csv"


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def quality_report(df: pd.DataFrame) -> dict:
    """Step 5 — data quality checks."""
    report = {
        "n_rows": len(df),
        "n_duplicates": int(df.duplicated().sum()),
        "duplicate_customer_ids": int(df["Customer_ID"].duplicated().sum()),
        "missing_by_col": df.isnull().sum().to_dict(),
        "negative_monthly_charge": int((df["Monthly_Charge"] < 0).sum()),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
    return report


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning decisions (documented, not silent):

    1. Drop exact duplicate rows.
    2. Drop customers with Customer_Status == 'Joined' — no churn outcome yet,
       including them would corrupt the base rate (see data_dictionary.md).
    3. Service add-on columns are blank ONLY when the customer has no internet
       service ("Internet_Type" also blank in that case) — this is NOT missing
       data, it is structurally "Not Applicable". Filled with 'No Internet'.
    4. Multiple_Lines is blank when Phone_Service == 'No' — same logic, filled
       with 'No Phone Service'.
    5. Value_Deal blank = customer is on no promotional deal -> filled 'No Deal'.
    6. Monthly_Charge has some negative values (source data quirk, likely
       promotional credits). These are legitimate billing values, not typos —
       kept as-is but flagged for awareness in EDA.
    7. Tenure_in_Months == 0 would break ratio features -> floored at 1 for
       any derived per-month ratios (raw tenure column itself is untouched).
    """
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates()

    df = df[df["Customer_Status"] != "Joined"].reset_index(drop=True)

    service_cols = [
        "Internet_Type", "Online_Security", "Online_Backup",
        "Device_Protection_Plan", "Premium_Support", "Streaming_TV",
        "Streaming_Movies", "Streaming_Music", "Unlimited_Data",
    ]
    for col in service_cols:
        df[col] = df[col].fillna("No Internet")

    df["Multiple_Lines"] = df["Multiple_Lines"].fillna("No Phone Service")
    df["Value_Deal"] = df["Value_Deal"].fillna("No Deal")

    # target
    df["churn"] = (df["Customer_Status"] == "Churned").astype(int)

    print(f"[clean] rows before: {before} -> after: {len(df)} "
          f"(dropped duplicates + 'Joined' customers)")
    return df


def main():
    df = load_raw()
    report = quality_report(df)
    print("=== QUALITY REPORT (raw) ===")
    for k, v in report.items():
        if k not in ("missing_by_col", "dtypes"):
            print(f"{k}: {v}")
    print("missing_by_col (non-zero only):")
    for k, v in report["missing_by_col"].items():
        if v:
            print(f"  {k}: {v}")

    clean_df = clean(df)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved cleaned data -> {OUT_PATH}  shape={clean_df.shape}")
    print("Churn rate:", clean_df["churn"].mean().round(4))


if __name__ == "__main__":
    main()
