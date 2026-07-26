"""
Streamlit inference dashboard for the Credit Card Default Predictor.

Run locally:   streamlit run app.py
Deploy free:   push this repo to GitHub, then deploy on
               https://share.streamlit.io (Streamlit Community Cloud) --
               point it at app.py, requirements.txt gets picked up automatically.
"""
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Default Risk", page_icon="💳", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    with open("metrics.json") as f:
        metrics = json.load(f)
    return model, metrics


model, metrics = load_artifacts()

st.title("💳 Credit Card Default Risk Predictor")
st.caption(
    "Logistic regression model estimating the probability a borrower defaults, "
    "with a business-tunable decision threshold."
)

tab_predict, tab_metrics, tab_about = st.tabs(["Predict", "Model Metrics", "About"])

# ---------------------------------------------------------------- Predict ---
with tab_predict:
    st.subheader("Enter borrower details")
    col1, col2 = st.columns(2)
    with col1:
        income = st.number_input("Annual Income ($)", min_value=0.0, value=45000.0, step=1000.0)
        age = st.number_input("Age", min_value=18.0, max_value=100.0, value=35.0, step=1.0)
    with col2:
        loan = st.number_input("Outstanding Loan ($)", min_value=0.0, value=4000.0, step=500.0)
        loan_to_income = loan / income if income > 0 else 0.0
        st.metric("Loan-to-Income ratio (auto-computed)", f"{loan_to_income:.3f}")

    st.divider()
    st.subheader("Decision threshold")
    threshold = st.slider(
        "Probability cutoff to flag as 'likely default'",
        min_value=0.01, max_value=0.99,
        value=float(metrics["threshold_tuned"]["threshold"]),
        step=0.01,
        help="Lower = catch more defaulters (higher recall, more false alarms). "
             "Higher = fewer false alarms, but more missed defaulters.",
    )

    X = pd.DataFrame(
        [[income, age, loan, loan_to_income]],
        columns=["Income", "Age", "Loan", "Loan to Income"],
    )
    proba = model.predict_proba(X)[0, 1]
    flagged = proba >= threshold

    st.divider()
    c1, c2 = st.columns(2)
    c1.metric("Predicted default probability", f"{proba:.1%}")
    c2.metric("Decision at this threshold", "⚠️ Likely default" if flagged else "✅ Likely repay")

# ---------------------------------------------------------------- Metrics ---
with tab_metrics:
    st.subheader("Held-out test set performance")
    st.metric("AUC-ROC", metrics["auc_roc"])

    default_report = metrics["threshold_default"]["report"]["1"]
    tuned_report = metrics["threshold_tuned"]["report"]["1"]

    st.markdown("**Class = Default (positive class)**")
    comp = pd.DataFrame(
        {
            "Threshold = 0.50 (default)": {
                "Precision": round(default_report["precision"], 3),
                "Recall": round(default_report["recall"], 3),
                "F1": round(default_report["f1-score"], 3),
            },
            f"Threshold = {metrics['threshold_tuned']['threshold']} (cost-tuned)": {
                "Precision": round(tuned_report["precision"], 3),
                "Recall": round(tuned_report["recall"], 3),
                "F1": round(tuned_report["f1-score"], 3),
            },
        }
    )
    st.dataframe(comp)

    st.markdown("**Confusion matrix at cost-tuned threshold**")
    cm = metrics["threshold_tuned"]["confusion_matrix"]
    st.dataframe(
        pd.DataFrame(
            cm,
            index=["Actual: No Default", "Actual: Default"],
            columns=["Pred: No Default", "Pred: Default"],
        )
    )

# ------------------------------------------------------------------ About ---
with tab_about:
    st.markdown(
        """
        **Model:** Logistic Regression on `Income`, `Age`, `Loan`, `Loan-to-Income`.

        **Why the threshold moved from 0.5 to {t}:** a missed defaulter (false
        negative) costs the bank far more than a wrongly-declined good
        customer (false positive) — a written-off loan versus a small amount
        of forgone interest margin. This dashboard uses a 5:1 FN:FP cost
        assumption to pick the operating threshold; adjust the slider on the
        Predict tab to see the recall/precision trade-off at other cutoffs.
        """.format(t=metrics["threshold_tuned"]["threshold"])
    )
