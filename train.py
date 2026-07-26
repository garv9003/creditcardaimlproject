"""
Trains the credit-card-default logistic regression model and saves the
artifacts (model.pkl, metrics.json) used by the Streamlit dashboard.

Run: python train.py
"""
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score
)

DATA_URL = "https://raw.githubusercontent.com/ybifoundation/Dataset/main/Credit%20Default.csv"
FEATURES = ["Income", "Age", "Loan", "Loan to Income"]
RANDOM_STATE = 2529


def main():
    df = pd.read_csv(DATA_URL)
    X = df[FEATURES]
    y = df["Default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=RANDOM_STATE
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)

    # Business-driven threshold: a missed defaulter (FN) costs the bank far
    # more than a wrongly-declined good customer (FP). We sweep thresholds
    # under a 5:1 FN:FP cost ratio and keep the cheapest one.
    fn_cost, fp_cost = 5, 1
    best_t, best_cost = 0.5, float("inf")
    for t in np.arange(0.05, 0.95, 0.01):
        preds = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        cost = fp * fp_cost + fn * fn_cost
        if cost < best_cost:
            best_cost, best_t = cost, t

    tuned_preds = (proba >= best_t).astype(int)
    default_preds = (proba >= 0.5).astype(int)

    metrics = {
        "features": FEATURES,
        "auc_roc": round(float(auc), 4),
        "threshold_default": {
            "threshold": 0.5,
            "report": classification_report(y_test, default_preds, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, default_preds).tolist(),
        },
        "threshold_tuned": {
            "threshold": round(float(best_t), 2),
            "cost_ratio_fn_to_fp": f"{fn_cost}:{fp_cost}",
            "report": classification_report(y_test, tuned_preds, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, tuned_preds).tolist(),
        },
    }

    joblib.dump(model, "model.pkl")
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print("\nSaved model.pkl and metrics.json")


if __name__ == "__main__":
    main()
