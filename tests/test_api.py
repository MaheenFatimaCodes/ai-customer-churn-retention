import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

MODEL_PATH = ROOT / "models" / "churn_model.pkl"

if not MODEL_PATH.exists():
    pytest.skip("Model not trained yet — run src/train_models.py first", allow_module_level=True)

from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)

SAMPLE_CUSTOMER = {
    "Age": 35, "Gender": "Male", "Married": "No", "State": "Delhi",
    "Number_of_Referrals": 0, "Tenure_in_Months": 3, "Value_Deal": "No Deal",
    "Phone_Service": "Yes", "Multiple_Lines": "No", "Internet_Service": "Yes",
    "Internet_Type": "Fiber Optic", "Online_Security": "No", "Online_Backup": "No",
    "Device_Protection_Plan": "No", "Premium_Support": "No", "Streaming_TV": "No",
    "Streaming_Movies": "No", "Streaming_Music": "No", "Unlimited_Data": "Yes",
    "Contract": "Month-to-Month", "Paperless_Billing": "Yes", "Payment_Method": "Bank Withdrawal",
    "Monthly_Charge": 95.0, "Total_Charges": 285.0, "Total_Refunds": 20.0,
    "Total_Extra_Data_Charges": 0, "Total_Long_Distance_Charges": 40.0, "Total_Revenue": 305.0,
}


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_returns_valid_schema():
    resp = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert resp.status_code == 200
    body = resp.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["risk_level"] in {"Low", "Medium", "High", "Critical"}
    assert isinstance(body["top_factors_increasing_risk"], list)
    assert isinstance(body["recommended_action"], str)


def test_whatif_improves_with_better_contract():
    payload = {
        "customer": SAMPLE_CUSTOMER,
        "overrides": {"Contract": "Two Year", "Total_Refunds": 0},
    }
    resp = client.post("/whatif", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["after"]["churn_probability"] <= body["before"]["churn_probability"]


def test_whatif_rejects_unknown_column():
    payload = {"customer": SAMPLE_CUSTOMER, "overrides": {"Not_A_Real_Column": 1}}
    resp = client.post("/whatif", json=payload)
    assert resp.status_code == 400
