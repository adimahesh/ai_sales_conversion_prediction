"""Streamlit UI for scoring sales leads."""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path("models/best_sales_model.joblib")
METADATA_PATH = Path("models/model_metadata.json")
DATA_PATH = Path("data/sales_leads.csv")

st.set_page_config(page_title="Sales Conversion Predictor", page_icon="📈", layout="wide")
st.title("📈 AI-Based Sales Conversion Probability Prediction")
st.caption("Score leads, prioritize sales follow-up, and inspect the model's main drivers.")

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.warning("The trained model is missing. In the VS Code terminal, run:")
    st.code("python generate_data.py\npython train_model.py", language="powershell")
    st.stop()

model = joblib.load(MODEL_PATH)
metadata = json.loads(METADATA_PATH.read_text())
feature_columns = metadata["features"]
best_model = metadata["best_model"]

with st.sidebar:
    st.header("Model details")
    st.write(f"**Selected model:** {best_model}")
    st.write(f"**Test ROC-AUC:** {metadata['metrics'][best_model]['roc_auc']:.3f}")
    st.write(f"**Training rows:** {metadata['training_rows']:,}")
    st.info("Demo data is synthetic. Treat these scores as examples, not business guarantees.")

tab_score, tab_upload, tab_insights, tab_about = st.tabs([
    "Score one lead", "Upload leads CSV", "Model insights", "Project guide"
])

with tab_score:
    st.subheader("Enter lead details")
    col1, col2 = st.columns(2)
    with col1:
        lead_source = st.selectbox("Lead source", ["Website", "Referral", "LinkedIn", "Email Campaign", "Cold Call"])
        industry = st.selectbox("Industry", ["Technology", "Healthcare", "Finance", "Retail", "Manufacturing"])
        company_size = st.selectbox("Company size", ["Small", "Medium", "Large"])
        region = st.selectbox("Region", ["North", "South", "East", "West"])
        touchpoints = st.slider("Number of touchpoints", 1, 12, 5)
    with col2:
        days_to_contact = st.slider("Days until first contact", 0, 30, 3)
        website_visits = st.slider("Website visits", 0, 20, 4)
        email_engagement = st.slider("Email engagement (0–10)", 0, 10, 5)
        lead_score = st.slider("Lead score (10–100)", 10, 100, 60)
        deal_value = st.number_input("Estimated deal value", min_value=0.0, value=15000.0, step=500.0)

    if st.button("Predict conversion probability", type="primary"):
        row = pd.DataFrame([{
            "lead_source": lead_source,
            "industry": industry,
            "company_size": company_size,
            "region": region,
            "touchpoints": touchpoints,
            "days_to_contact": days_to_contact,
            "website_visits": website_visits,
            "email_engagement": email_engagement,
            "lead_score": lead_score,
            "deal_value": deal_value,
        }])[feature_columns]
        probability = float(model.predict_proba(row)[0, 1])
        st.metric("Estimated conversion probability", f"{probability:.1%}")
        if probability >= 0.70:
            st.success("High-priority lead: consider timely follow-up.")
        elif probability >= 0.40:
            st.warning("Medium-priority lead: review engagement and follow-up timing.")
        else:
            st.info("Lower predicted probability: consider nurturing or gathering more information.")
        st.progress(int(round(probability * 100)))
        st.caption("Priority bands are illustrative and should be calibrated using real sales outcomes.")

with tab_upload:
    st.subheader("Score multiple leads")
    st.write("Upload a CSV containing the feature columns below. The `converted` target column is not needed.")
    st.code(", ".join(feature_columns))
    uploaded = st.file_uploader("Choose a leads CSV", type=["csv"])
    if uploaded is not None:
        try:
            incoming = pd.read_csv(uploaded)
            missing = sorted(set(feature_columns) - set(incoming.columns))
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            else:
                scores = model.predict_proba(incoming[feature_columns])[:, 1]
                output = incoming.copy()
                output["conversion_probability"] = scores
                output["priority"] = pd.cut(
                    scores, bins=[-0.001, 0.40, 0.70, 1.001],
                    labels=["Low", "Medium", "High"]
                )
                output = output.sort_values("conversion_probability", ascending=False)
                st.dataframe(output, use_container_width=True)
                st.download_button(
                    "Download scored leads CSV",
                    data=output.to_csv(index=False).encode("utf-8"),
                    file_name="scored_sales_leads.csv",
                    mime="text/csv",
                )
        except Exception as exc:
            st.error(f"Could not process this CSV: {exc}")

with tab_insights:
    st.subheader("Evaluation summary")
    metrics = metadata["metrics"]
    metric_rows = []
    for name, values in metrics.items():
        metric_rows.append({
            "Model": name,
            "ROC-AUC": values["roc_auc"],
            "Accuracy (0.50)": values["accuracy_at_0_5"],
            "Precision (0.50)": values["precision_at_0_5"],
            "Recall (0.50)": values["recall_at_0_5"],
            "F1 (0.50)": values["f1_at_0_5"],
        })
    st.dataframe(pd.DataFrame(metric_rows).set_index("Model").style.format("{:.3f}"), use_container_width=True)
    st.caption("Metrics shown are from a held-out test split of synthetic demo data.")

    comparison_path = Path("reports/model_comparison.png")
    if comparison_path.exists():
        st.image(str(comparison_path), caption="ROC curves on the test set", use_container_width=True)

    importance_path = Path("reports/permutation_importance.csv")
    if importance_path.exists():
        importance_df = pd.read_csv(importance_path).sort_values("importance", ascending=True)
        st.subheader("Global feature importance")
        st.bar_chart(importance_df.set_index("feature")["importance"])
        st.caption("Permutation importance estimates how much ROC-AUC drops when a feature is shuffled.")

    st.subheader("Optional SHAP explanation")
    st.write("SHAP is listed in requirements. For a full per-lead SHAP explanation, use a model-appropriate SHAP explainer and validate it against the fitted preprocessing pipeline. This starter app uses permutation importance for a reliable global overview.")

with tab_about:
    st.subheader("What this project does")
    st.markdown("""
    1. Loads lead attributes and a binary conversion outcome.
    2. Cleans records and explores conversion patterns.
    3. Imputes missing values, scales numeric columns, and one-hot encodes categories.
    4. Trains Logistic Regression and Random Forest models.
    5. Compares ROC-AUC and saves the stronger model.
    6. Predicts conversion probabilities and ranks leads for follow-up.
    """)
    st.warning("Use only authorized data. Do not upload personal or confidential customer information to an unapproved environment.")
