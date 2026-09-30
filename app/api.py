"""
REST API for the churn model.

Run:  uvicorn app.api:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from typing import Optional
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pipeline import ALL_FEATURES
from feature_engineering import engineer_features
from retention_engine import risk_bucket, recommend_action
from explain import explain_customer

MODELS_DIR = ROOT / "models"

app = FastAPI(
    title="Churn Prediction & Retention Intelligence API",
    description="Predicts customer churn probability, explains the drivers, "
                "and recommends a retention action.",
    version="1.0.0",
)

_model = None
_explainer = None
_feature_names = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODELS_DIR / "churn_model.pkl")
    return _model


def get_explainer():
    global _explainer, _feature_names
    if _explainer is None:
        bundle = joblib.load(MODELS_DIR / "shap_explainer.pkl")
        _explainer = bundle["explainer"]
        _feature_names = bundle["feature_names"]
    return _explainer, _feature_names


class CustomerInput(BaseModel):
    Age: int = Field(..., example=35)
    Gender: str = Field(..., example="Male")
    Married: str = Field(..., example="No")
    State: str = Field(..., example="Delhi")
    Number_of_Referrals: int = Field(..., example=2)
    Tenure_in_Months: int = Field(..., example=12)
    Value_Deal: str = Field("No Deal", example="No Deal")
    Phone_Service: str = Field(..., example="Yes")
    Multiple_Lines: str = Field("No Phone Service", example="No")
    Internet_Service: str = Field(..., example="Yes")
    Internet_Type: str = Field("No Internet", example="Fiber Optic")
    Online_Security: str = Field("No Internet", example="No")
    Online_Backup: str = Field("No Internet", example="No")
    Device_Protection_Plan: str = Field("No Internet", example="No")
    Premium_Support: str = Field("No Internet", example="No")
    Streaming_TV: str = Field("No Internet", example="No")
    Streaming_Movies: str = Field("No Internet", example="No")
    Streaming_Music: str = Field("No Internet", example="No")
    Unlimited_Data: str = Field("No Internet", example="Yes")
    Contract: str = Field(..., example="Month-to-Month")
    Paperless_Billing: str = Field(..., example="Yes")
    Payment_Method: str = Field(..., example="Credit Card")
    Monthly_Charge: float = Field(..., example=75.5)
    Total_Charges: float = Field(..., example=900.0)
    Total_Refunds: float = Field(0.0, example=0.0)
    Total_Extra_Data_Charges: float = Field(0.0, example=0.0)
    Total_Long_Distance_Charges: float = Field(0.0, example=120.0)
    Total_Revenue: float = Field(..., example=1020.0)


class PredictionOutput(BaseModel):
    churn_probability: float
    risk_level: str
    top_factors_increasing_risk: list
    top_factors_decreasing_risk: list
    recommended_action: str


class WhatIfInput(BaseModel):
    customer: CustomerInput
    overrides: dict = Field(default_factory=dict, example={"Contract": "Two Year", "Total_Refunds": 0})


@app.get("/")
def root():
    return {
        "service": "Churn Prediction & Retention Intelligence API",
        "endpoints": ["/predict", "/whatif", "/health"],
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


def _score(df_raw: pd.DataFrame) -> dict:
    model = get_model()
    feat = engineer_features(df_raw)
    proba = float(model.predict_proba(feat[ALL_FEATURES])[0, 1])
    risk = risk_bucket(proba)

    explainer, feature_names = get_explainer()
    preprocessor = model.named_steps["prep"]
    explanation = explain_customer(explainer, preprocessor, feature_names, feat[ALL_FEATURES])

    action = recommend_action({
        "churn_proba": proba,
        "refund_rate": feat["refund_rate"].iloc[0],
        "is_month_to_month": feat["is_month_to_month"].iloc[0],
        "avg_monthly_revenue": feat["avg_monthly_revenue"].iloc[0],
        "services_subscribed": feat["services_subscribed"].iloc[0],
    })

    return {
        "churn_probability": round(proba, 4),
        "risk_level": risk,
        "top_factors_increasing_risk": explanation["increasing_risk"],
        "top_factors_decreasing_risk": explanation["decreasing_risk"],
        "recommended_action": action,
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput):
    try:
        df_raw = pd.DataFrame([customer.model_dump()])
        result = _score(df_raw)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/whatif")
def whatif(payload: WhatIfInput):
    try:
        base_df = pd.DataFrame([payload.customer.model_dump()])
        before = _score(base_df)

        modified_df = base_df.copy()
        for col, val in payload.overrides.items():
            if col not in modified_df.columns:
                raise HTTPException(status_code=400, detail=f"Unknown column override: {col}")
            modified_df.loc[modified_df.index[0], col] = val
        after = _score(modified_df)

        return {
            "before": before,
            "after": after,
            "change_in_probability": round(after["churn_probability"] - before["churn_probability"], 4),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
