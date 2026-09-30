"""
Shared preprocessing pipeline (used by training, API, and the what-if simulator)
so train-time and inference-time transformations can never drift apart.
"""
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "features.csv"

# Columns dropped before modeling: IDs and leakage fields (see data_dictionary.md)
DROP_COLS = ["Customer_ID", "Customer_Status", "Churn_Category", "Churn_Reason", "churn"]

NUMERIC_FEATURES = [
    "Age", "Number_of_Referrals", "Tenure_in_Months", "Monthly_Charge",
    "Total_Charges", "Total_Refunds", "Total_Extra_Data_Charges",
    "Total_Long_Distance_Charges", "Total_Revenue",
    "avg_monthly_revenue", "refund_rate", "services_subscribed",
    "long_distance_share", "extra_data_share", "revenue_per_referral",
]

CATEGORICAL_FEATURES = [
    "Gender", "Married", "State", "Value_Deal", "Phone_Service", "Multiple_Lines",
    "Internet_Service", "Internet_Type", "Online_Security", "Online_Backup",
    "Device_Protection_Plan", "Premium_Support", "Streaming_TV", "Streaming_Movies",
    "Streaming_Music", "Unlimited_Data", "Contract", "Paperless_Billing",
    "Payment_Method", "tenure_bucket",
]

BINARY_FEATURES = ["has_internet", "is_month_to_month", "has_value_deal", "is_paperless"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES


def load_features() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES + BINARY_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def get_X_y(df: pd.DataFrame):
    X = df[ALL_FEATURES].copy()
    y = df["churn"].copy()
    return X, y
