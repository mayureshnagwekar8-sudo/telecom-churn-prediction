"""Train the churn model chosen in the notebook and save it for the app.

Model: AdaBoost with sample weights for the minority (churn) class,
inside a pipeline with StandardScaler fitted on the training split only.

Usage: python train.py
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from churn import TARGET, clean, encode

DATA_PATH = "Customer-Churn.csv"
MODEL_PATH = "churn_model.joblib"
METRICS_PATH = "metrics.json"
SEED = 42


def main():
    df = clean(pd.read_csv(DATA_PATH))
    X = encode(df)
    y = df[TARGET].map({"No": 0, "Yes": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # Weight churners by (non-churners / churners) so the model doesn't ignore them
    weight_positive = (y_train == 0).sum() / (y_train == 1).sum()
    sample_weight = np.where(y_train == 1, weight_positive, 1.0)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", AdaBoostClassifier(n_estimators=50, random_state=SEED)),
    ])
    model.fit(X_train, y_train, model__sample_weight=sample_weight)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 3),
        "churn_recall": round(recall_score(y_test, y_pred), 3),
        "churn_f1": round(f1_score(y_test, y_pred), 3),
        "test_rows": int(len(y_test)),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save the pipeline plus what the app needs to build input forms
    raw = df.drop(columns=["customerID", TARGET])
    joblib.dump(
        {
            "pipeline": model,
            "columns": list(X.columns),
            "options": {
                c: sorted(raw[c].unique().tolist()) for c in raw.columns if raw[c].dtype == object
            },
            "numeric_ranges": {
                c: (float(raw[c].min()), float(raw[c].max()), float(raw[c].median()))
                for c in ["MonthlyCharges", "TotalCharges"]
            },
        },
        MODEL_PATH,
    )
    print(f"Saved {MODEL_PATH} and {METRICS_PATH}: {metrics}")


if __name__ == "__main__":
    main()
