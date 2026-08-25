"""
ml/evaluate_model.py

Loads the already-trained pipeline from models/predictive_maintenance_model.pkl
and re-evaluates it on a fresh stratified test split of data/ai4i2020.csv.
Useful for sanity-checking a saved model without re-training it.

Run:
    python ml/evaluate_model.py
"""
import os

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ai4i2020.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "predictive_maintenance_model.pkl")

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
CATEGORICAL_FEATURES = ["Type"]
TARGET = "Machine failure"


def main():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(
            f"No trained model found at {MODEL_PATH}.\nRun 'python ml/train_model.py' first."
        )

    df = pd.read_csv(DATA_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pipeline = joblib.load(MODEL_PATH)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["Healthy", "Failure"], zero_division=0))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")


if __name__ == "__main__":
    main()
