"""
AI-Powered Customer Churn Prediction & Retention Intelligence System
Streamlit dashboard.

Run:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

from pipeline import ALL_FEATURES
from feature_engineering import engineer_features
from retention_engine import apply_engine, revenue_at_risk, risk_bucket, recommend_action
from explain import explain_customer

st.set_page_config(page_title="Churn Intelligence System", layout="wide", page_icon="📉")

MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "processed"


# ---------------------------------------------------------------- caching --
@st.cache_resource
def load_model():
    return joblib.load(MODELS_DIR / "churn_model.pkl")


@st.cache_resource
def load_explainer():
    bundle = joblib.load(MODELS_DIR / "shap_explainer.pkl")
    return bundle["explainer"], bundle["feature_names"]


@st.cache_data
def load_scored_data():
    df = pd.read_csv(DATA_DIR / "scored_customers.csv")
    return df


@st.cache_data
def load_segmented_data():
    df = pd.read_csv(DATA_DIR / "segmented_customers.csv")
    return df


@st.cache_data
def load_metrics():
    with open(ROOT / "reports" / "model_metrics.json") as f:
        return json.load(f)


@st.cache_data
def load_clean_data():
    return pd.read_csv(DATA_DIR / "clean_data.csv")


model = load_model()
explainer, feature_names = load_explainer()
scored_df = load_scored_data()
segmented_df = load_segmented_data()
metrics = load_metrics()
clean_df = load_clean_data()

preprocessor = model.named_steps["prep"]

st.title("📉 AI-Powered Customer Churn Prediction & Retention Intelligence System")
st.caption("End-to-end churn risk scoring, explainability, segmentation, retention actions, and business impact.")

tabs = st.tabs([
    "📊 Executive Overview", "🔍 Customer Risk Lookup", "🧩 Segmentation",
    "🎛️ What-if Simulator", "📈 Model Performance",
])

# =====================================================================
# TAB 1 — EXECUTIVE OVERVIEW
# =====================================================================
with tabs[0]:
    impact = revenue_at_risk(scored_df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total customers scored", f"{impact['total_customers_scored']:,}")
    c2.metric("High / Critical risk", f"{impact['high_or_critical_risk_customers']:,}")
    c3.metric("Revenue at risk", f"₹{impact['revenue_at_risk']:,.0f}")
    c4.metric("Potential revenue retained", f"₹{impact['potential_revenue_retained_if_campaign_succeeds']:,.0f}",
              help=f"Assumes a {impact['assumed_retention_rate']*100:.0f}% retention-campaign success rate — adjust in the code to match your own campaign data.")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        risk_counts = scored_df["risk_level"].value_counts().reindex(
            ["Low", "Medium", "High", "Critical"]).fillna(0)
        fig = px.bar(x=risk_counts.index, y=risk_counts.values,
                     color=risk_counts.index,
                     color_discrete_map={"Low": "#4C72B0", "Medium": "#DD8452",
                                          "High": "#C44E52", "Critical": "#8B0000"},
                     labels={"x": "Risk level", "y": "Customers"},
                     title="Customers by risk level (test set, n=902)")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        action_counts = scored_df["recommended_action"].value_counts()
        fig2 = px.pie(values=action_counts.values, names=action_counts.index,
                       title="Recommended actions distribution")
        st.plotly_chart(fig2, use_container_width=True)

    st.info(
        f"💡 **Assumption note:** revenue-at-risk sums `Total_Revenue` for every High/Critical-risk "
        f"customer — it is an *exposure* estimate, not a guaranteed loss. Potential revenue retained "
        f"assumes a **{impact['assumed_retention_rate']*100:.0f}%** campaign success rate; replace with "
        f"your own historical retention-campaign conversion once available."
    )

# =====================================================================
# TAB 2 — CUSTOMER RISK LOOKUP
# =====================================================================
with tabs[1]:
    st.subheader("Search for a customer")
    customer_ids = clean_df["Customer_ID"].tolist()
    selected_id = st.selectbox("Customer ID", customer_ids, index=0)

    row = clean_df[clean_df["Customer_ID"] == selected_id].iloc[0]
    feat_row = engineer_features(pd.DataFrame([row]))
    proba = float(model.predict_proba(feat_row[ALL_FEATURES])[0, 1])
    risk = risk_bucket(proba)
    action = recommend_action({
        "churn_proba": proba,
        "refund_rate": feat_row["refund_rate"].iloc[0],
        "is_month_to_month": feat_row["is_month_to_month"].iloc[0],
        "avg_monthly_revenue": feat_row["avg_monthly_revenue"].iloc[0],
        "services_subscribed": feat_row["services_subscribed"].iloc[0],
    })

    risk_colors = {"Low": "🟢", "Medium": "🟡", "High": "🟠", "Critical": "🔴"}
    c1, c2, c3 = st.columns(3)
    c1.metric("Churn probability", f"{proba*100:.1f}%")
    c2.metric("Risk level", f"{risk_colors[risk]} {risk}")
    c3.metric("Tenure", f"{int(row['Tenure_in_Months'])} months")

    st.success(f"**Recommended action:** {action}")

    st.divider()
    st.subheader("Why this prediction? (SHAP)")
    explanation = explain_customer(explainer, preprocessor, feature_names, feat_row[ALL_FEATURES])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🔺 Increasing churn risk**")
        for item in explanation["increasing_risk"]:
            st.write(f"+ {item['feature']}  (impact: {item['impact']:+.3f})")
    with col2:
        st.markdown("**🔻 Decreasing churn risk**")
        for item in explanation["decreasing_risk"]:
            st.write(f"- {item['feature']}  (impact: {item['impact']:+.3f})")

    with st.expander("Raw customer record"):
        st.dataframe(row.to_frame().T)

# =====================================================================
# TAB 3 — SEGMENTATION
# =====================================================================
with tabs[2]:
    st.subheader("Customer segments (K-Means, unsupervised)")
    seg_summary = segmented_df.groupby("segment").agg(
        customers=("churn", "count"),
        churn_rate=("churn", "mean"),
        avg_revenue=("Total_Revenue", "mean"),
        avg_tenure=("Tenure_in_Months", "mean"),
    ).sort_values("churn_rate", ascending=False).reset_index()
    seg_summary["churn_rate"] = (seg_summary["churn_rate"] * 100).round(1)
    st.dataframe(seg_summary.style.format({"avg_revenue": "₹{:.0f}", "avg_tenure": "{:.1f}"}),
                 use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(seg_summary, x="segment", y="churn_rate", color="segment",
                     title="Churn rate (%) by segment", labels={"churn_rate": "Churn rate (%)"})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.scatter(segmented_df.sample(min(1500, len(segmented_df)), random_state=1),
                          x="Tenure_in_Months", y="avg_monthly_revenue", color="segment",
                          opacity=0.6, title="Segments: Tenure vs Avg Monthly Revenue")
        st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# TAB 4 — WHAT-IF SIMULATOR
# =====================================================================
with tabs[3]:
    st.subheader("🎛️ Churn Risk Simulator")
    st.caption("Adjust a customer's attributes and see churn probability recalculate live.")

    sim_id = st.selectbox("Pick a customer to simulate", customer_ids, index=0, key="sim_id")
    base_row = clean_df[clean_df["Customer_ID"] == sim_id].iloc[0].copy()

    col1, col2, col3 = st.columns(3)
    with col1:
        new_contract = st.selectbox("Contract", ["Month-to-Month", "One Year", "Two Year"],
                                     index=["Month-to-Month", "One Year", "Two Year"].index(base_row["Contract"]))
        new_refunds = st.slider("Total Refunds (₹)", 0.0, 200.0, float(min(base_row["Total_Refunds"], 200)))
    with col2:
        new_security = st.selectbox("Online Security", ["Yes", "No"],
                                     index=0 if base_row.get("Online_Security") == "Yes" else 1)
        new_support = st.selectbox("Premium Support", ["Yes", "No"],
                                    index=0 if base_row.get("Premium_Support") == "Yes" else 1)
    with col3:
        new_tenure = st.slider("Tenure (months)", 1, 36, int(base_row["Tenure_in_Months"]))
        new_monthly = st.slider("Monthly Charge (₹)", 0.0, 150.0, float(max(base_row["Monthly_Charge"], 0)))

    overrides = {
        "Contract": new_contract,
        "Total_Refunds": new_refunds,
        "Online_Security": new_security,
        "Premium_Support": new_support,
        "Tenure_in_Months": new_tenure,
        "Monthly_Charge": new_monthly,
    }

    original_feat = engineer_features(pd.DataFrame([base_row]))
    modified_row = base_row.copy()
    for k, v in overrides.items():
        modified_row[k] = v
    modified_feat = engineer_features(pd.DataFrame([modified_row]))

    proba_before = float(model.predict_proba(original_feat[ALL_FEATURES])[0, 1])
    proba_after = float(model.predict_proba(modified_feat[ALL_FEATURES])[0, 1])

    c1, c2, c3 = st.columns(3)
    c1.metric("Before", f"{proba_before*100:.1f}%")
    c2.metric("After", f"{proba_after*100:.1f}%", delta=f"{(proba_after-proba_before)*100:+.1f} pts",
              delta_color="inverse")
    c3.metric("Risk level (after)", risk_bucket(proba_after))

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=proba_after * 100,
        delta={"reference": proba_before * 100},
        gauge={"axis": {"range": [0, 100]},
               "steps": [{"range": [0, 30], "color": "#c8e6c9"},
                         {"range": [30, 60], "color": "#fff9c4"},
                         {"range": [60, 80], "color": "#ffccbc"},
                         {"range": [80, 100], "color": "#ffcdd2"}],
               "bar": {"color": "#333"}},
        title={"text": "Churn probability (%)"},
    ))
    st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# TAB 5 — MODEL PERFORMANCE
# =====================================================================
with tabs[4]:
    st.subheader("Model comparison (validation set)")
    comp_df = pd.DataFrame(metrics["model_comparison_val"]).T
    st.dataframe(comp_df, use_container_width=True)
    st.caption(f"Selected model: **{metrics['best_model']}**")

    st.subheader("Final test-set performance")
    tm = metrics["test_metrics"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ROC-AUC", f"{tm['roc_auc']:.3f}")
    c2.metric("PR-AUC", f"{tm['pr_auc']:.3f}")
    c3.metric("F1", f"{tm['f1']:.3f}")
    c4.metric("Recall", f"{tm['recall']:.3f}")
    c5.metric("Precision", f"{tm['precision']:.3f}")

    cm = np.array(tm["confusion_matrix"])
    fig = px.imshow(cm, text_auto=True, x=["Pred: Stayed", "Pred: Churned"],
                     y=["Actual: Stayed", "Actual: Churned"], color_continuous_scale="Blues",
                     title="Confusion Matrix (test set)")
    st.plotly_chart(fig, use_container_width=True)

    fpr, tpr, _ = roc_curve(scored_df["churn"], scored_df["churn_proba"])
    prec, rec, _ = precision_recall_curve(scored_df["churn"], scored_df["churn_proba"])

    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(x=fpr, y=tpr, title=f"ROC Curve (AUC={tm['roc_auc']:.3f})",
                       labels={"x": "False Positive Rate", "y": "True Positive Rate"})
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray"))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.line(x=rec, y=prec, title=f"Precision-Recall Curve (AUC={tm['pr_auc']:.3f})",
                       labels={"x": "Recall", "y": "Precision"})
        st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("Built as an end-to-end ML system: data cleaning → feature engineering → model comparison "
           "→ SHAP explainability → segmentation → retention engine → simulator → business impact.")
