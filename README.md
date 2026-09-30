# 📉 AI-Powered Customer Churn Prediction & Retention Intelligence System

An end-to-end machine learning system that scores every customer's churn risk, explains
*why* the model thinks so, groups customers into actionable segments, recommends a specific
retention action, estimates the revenue at stake, and exposes all of it through an
interactive dashboard and a REST API.

> Built progressively across ML foundations → advanced ML → product layer → production layer,
> on a real telecom customer dataset (6,418 customers, 32 raw fields).

---

## Problem Statement

Predict which customers are at risk of churning and give the business a reason and an
action for every prediction — not just a probability.

## Business Objective

Reduce revenue lost to churn by identifying at-risk customers early enough for a retention
team to intervene, prioritizing effort by expected revenue impact rather than treating every
customer the same.

## Dataset

- **Source:** `data/raw/Customer_Data.csv` — telecom customer data (India), 6,418 rows, 32 columns.
- **Target:** `Customer_Status` (`Stayed` / `Churned` / `Joined`) → binarized to `churn`.
  `Joined` customers (411 rows, too new to have an outcome) are excluded from modeling.
- Full column-by-column reference: **[`data_dictionary.md`](data_dictionary.md)**.
- ⚠️ **Honest scope note:** the source data is a status snapshot, not a purchase-event log —
  see the "Churn definition" section of the data dictionary for exactly what the model
  predicts and why a literal rolling 30-day window wasn't invented from data that doesn't
  support it.

## Architecture

```
CUSTOMER DATA → Validation → Cleaning → EDA → Feature Engineering
    → Train/Val/Test Split → Model Training (LogReg / RF / XGBoost)
    → Model Comparison → Hyperparameter Tuning → Final Model (Random Forest)
    → [ Churn Probability | SHAP Explainability | Customer Segments ]
    → Retention Recommendation Engine → Business Impact (Revenue at Risk)
    → [ Streamlit Dashboard | REST API ] → Model Monitoring
```

## Data Pipeline

| Stage | Script | What it does |
|---|---|---|
| Cleaning | `src/data_preprocessing.py` | Drops duplicates & `Joined` customers, fills structural (not random) missingness, documents every decision |
| Feature engineering | `src/feature_engineering.py` | Builds 11 engineered features (see data dictionary) |
| EDA | `src/eda.py` | Univariate/bivariate/multivariate analysis + Mann-Whitney U and chi-square significance tests |

Run individually or all at once:

```bash
pip install -r requirements.txt
python run_pipeline.py        # runs the full pipeline, in order, end to end
```

## Feature Engineering

11 engineered features turn raw billing/service columns into churn-relevant signals:
`avg_monthly_revenue`, `refund_rate` (complaint proxy), `services_subscribed`,
`is_month_to_month`, `long_distance_share`, `extra_data_share`, `revenue_per_referral`,
`tenure_bucket`, `has_value_deal`, `is_paperless`, `has_internet`. Full definitions in
[`data_dictionary.md`](data_dictionary.md).

## Models

Three models trained and compared with **5-fold stratified cross-validation** and
**RandomizedSearchCV** hyperparameter tuning, using `class_weight="balanced"` /
`scale_pos_weight` to handle the ~29% churn class imbalance:

- Logistic Regression (baseline)
- Random Forest (tuned: n_estimators, max_depth, min_samples_split/leaf)
- XGBoost (tuned: n_estimators, max_depth, learning_rate, subsample, colsample_bytree)

## Model Comparison

**Validation set performance**

| Model | ROC-AUC | PR-AUC | F1 | Recall | Precision |
|---|---:|---:|---:|---:|---:|
| **Random Forest** ⭐ | **0.888** | 0.803 | **0.704** | 0.792 | 0.634 |
| XGBoost | 0.886 | **0.807** | 0.689 | **0.800** | 0.605 |
| Logistic Regression | 0.864 | 0.744 | 0.662 | 0.812 | 0.560 |

**Random Forest is the current selected model** according to the stored model-selection results.

### Final held-out test set performance

| Metric | Score |
|---|---:|
| ROC-AUC | **0.898** |
| PR-AUC | **0.814** |
| F1 Score | **0.730** |
| Recall | **0.785** |
| Precision | **0.682** |

### Test Confusion Matrix

  text
                 Predicted
              No Churn  Churn
Actual No       547       95
Actual Churn     56      204


## Explainability

SHAP (`TreeExplainer`) provides both:
- **Global drivers** — top churn factors across the whole customer base (contract type,
  online security, tenure, payment method, monthly charge)
- **Per-customer explanations** — for any single customer, the specific factors pushing
  their risk up or down, surfaced in the dashboard and the `/predict` API response

## Customer Segmentation

K-Means clustering (k=4, features: tenure, avg monthly revenue, services subscribed,
referrals, total revenue, refund rate) combined with churn probability produces
labeled, actionable segments (e.g. *"High-value but at-risk"*, *"Loyal high-value"*,
*"New customers"*) — see `src/segmentation.py`.

## Retention Engine

A transparent, business-tunable rule layer sits on top of the model's probability output
(`src/retention_engine.py`) — e.g. `churn_probability > 0.75 AND refund_rate > 0.02 AND
month-to-month` → *"Priority support escalation + personalized retention offer"*. Thresholds
are documented assumptions, not hardcoded truths — tune them against your own campaign data.

## What-if Simulator

Change a customer's contract, tenure, refunds, or add-ons and watch churn probability
recompute live — in the dashboard's "🎛️ What-if Simulator" tab or via the `/whatif` API
endpoint. Example: a month-to-month customer with refunds at 89.6% churn risk drops to
15.1% after switching to a Two-Year contract and resolving refunds.

## Business Impact

Revenue-at-risk is computed as an **exposure estimate** (sum of `Total_Revenue` for
High/Critical-risk customers), paired with a documented, overridable assumption about
campaign retention rate (default 15%) — never presented as a guaranteed forecast.
See `src/retention_engine.py::revenue_at_risk()`.

## Deployment

### Dashboard
```bash
streamlit run app/streamlit_app.py
```
5 tabs: Executive Overview, Customer Risk Lookup, Segmentation, What-if Simulator, Model
Performance.

### API
```bash
uvicorn app.api:app --reload --port 8000
```
Interactive docs at `http://localhost:8000/docs`.

| Endpoint | Method | Purpose |
|---|---|---|
| `/predict` | POST | Churn probability + risk level + SHAP factors + recommended action for one customer |
| `/whatif` | POST | Before/after comparison after applying attribute overrides |
| `/health` | GET | Liveness check |

### Docker
```bash
python run_pipeline.py           # train models first — the image expects models/*.pkl to exist
docker compose up --build        # API on :8000, dashboard on :8501
```

## Monitoring

`src/monitoring.py` computes **PSI (Population Stability Index)** and **Kolmogorov-Smirnov**
tests to flag data drift, prediction drift, and missing-value spikes between a reference
distribution and a new batch — a foundation for scheduled monitoring, not a full production
observability stack.

## Testing

The project includes 16 automated tests covering:

- Data cleaning and preprocessing
- Feature engineering
- Model prediction and sanity checks
- Risk classification and recommendations
- FastAPI health and prediction endpoints
- What-if simulation and input validation

Run the test suite with:

```bash
pytest tests/ -v

Latest test result: **16 passed, 0 failed**
```

## Screenshots of EDA

Generated EDA and SHAP figures are in `reports/figures/`:
- `01_univariate.png` — churn class balance, tenure distribution
- `02_bivariate.png` — tenure/monthly charge vs churn, churn rate by contract
- `03_multivariate.png` — churn rate by value × engagement segment
- `04_shap_global_importance.png` — top 15 global churn drivers

## Screenshots of Dashboard

### Executive Overview

![Executive Overview](reports/screenshots/executive_overview.png)

The Executive Overview summarizes customer risk levels, revenue at risk, potential revenue retained, and recommended retention actions.

### Customer Risk & SHAP Explainability

![Customer Risk Lookup](reports/screenshots/customer_risk_lookup.png)

The Customer Risk Lookup provides customer-level churn probability, risk classification, recommended action, and SHAP-based factors influencing the prediction.

### Customer Segmentation

![Customer Segmentation](reports/screenshots/segmentation.png)

The segmentation view groups customers using K-Means clustering and presents segment-level churn rate, revenue, and tenure insights.

### What-if Retention Simulator

![What-if Simulator](reports/screenshots/what_if_simulator_1.png)

![What-if Simulator Results](reports/screenshots/what_if_simulator_2.png)

The What-if Simulator allows customer attributes to be adjusted and shows the resulting change in predicted churn probability and risk level.

### Model Performance

![Model Comparison](reports/screenshots/model_performance_1.png)

![Model Evaluation](reports/screenshots/model_performance_2.png)

The Model Performance view compares candidate models and presents final test-set metrics, confusion matrix, ROC curve, and Precision-Recall curve.

## Installation

```bash
git clone <this-repo>
cd churn_project
pip install -r requirements.txt
python run_pipeline.py                       # trains everything, ~2-3 min
streamlit run app/streamlit_app.py           # dashboard
# in a second terminal:
uvicorn app.api:app --reload --port 8000     # API
```

## Project Structure

```
churn_project/
├── data/
│   ├── raw/Customer_Data.csv
│   └── processed/               # generated: clean_data, features, scored_customers, ...
├── src/
│   ├── data_preprocessing.py    # cleaning + quality checks
│   ├── feature_engineering.py
│   ├── eda.py                   # univariate/bivariate/multivariate + stat tests
│   ├── pipeline.py              # shared sklearn preprocessing pipeline
│   ├── train_models.py          # LogReg/RF/XGBoost + CV + tuning + evaluation
│   ├── explain.py               # SHAP global + per-customer
│   ├── segmentation.py          # K-Means customer segments
│   ├── retention_engine.py      # rule-based recommendations + revenue at risk
│   ├── whatif_simulator.py      # what-if recomputation logic
│   └── monitoring.py            # drift detection (PSI, KS test)
├── app/
│   ├── streamlit_app.py         # 5-tab dashboard
│   └── api.py                   # FastAPI service
├── models/                      # generated: churn_model.pkl, shap_explainer.pkl, ...
├── reports/                     # generated: figures/, model_metrics.json, ...
├── tests/                       # pytest suite (16 tests)
├── data_dictionary.md
├── requirements.txt
├── Dockerfile / docker-compose.yml
└── run_pipeline.py              # one command, full pipeline
```

## Future Improvements

- Replace the billing-based proxies (`refund_rate`, `services_subscribed`) with a real
  complaints/support-ticket table if one becomes available, per the original brief.
- Add probability calibration (`CalibratedClassifierCV`) before using raw probabilities in
  the revenue-at-risk calculation.
- Wire `monitoring.py` into a scheduler (Airflow/cron) with alerting instead of on-demand runs.
- A/B test retention actions against a holdout group to replace the assumed 15% retention
  rate with a measured one.
- Add authentication and rate limiting to the API before any real deployment.

---

**Resume line:** *Built an end-to-end ML system to predict customer churn, incorporating
feature engineering, model comparison (Logistic Regression / Random Forest / XGBoost),
cross-validation and hyperparameter tuning, SHAP explainability, K-Means customer
segmentation, a rule-based retention recommendation engine, and an interactive what-if
risk simulator — deployed via a Streamlit dashboard and a FastAPI REST service with basic
drift monitoring.*
