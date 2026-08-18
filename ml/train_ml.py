"""
ml/train_ml.py

Trains and compares candidate Machine Learning models on the AI4I 2020 dataset.
Prevents target leakage by dropping indicator columns (TWF, HDF, PWF, OSF, RNF, UDI, Product ID).
Applies class weighting and saves the best performing ML model pipeline.
"""
import json
import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "ai4i2020.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
CATEGORICAL_FEATURES = ["Type"]
TARGET = "Machine failure"
LEAKAGE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF", "UDI", "Product ID"]


def load_and_clean_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"[ML] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"[ML] Target distribution: Healthy={sum(df[TARGET]==0)}, Failure={sum(df[TARGET]==1)}")
    
    # Drop data leakage columns
    drop_cols = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    df = df.drop(columns=drop_cols)
    return df


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )


def get_ml_models():
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=RANDOM_STATE, max_depth=8
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_depth=12,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, random_state=RANDOM_STATE, max_depth=4, learning_rate=0.1
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.08,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    return models


def train_ml():
    df = load_and_clean_data()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor()
    candidates = get_ml_models()

    results = []
    fitted_pipelines = {}

    for name, clf in candidates.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
        if name == "XGBoost":
            neg, pos = np.bincount(y_train)
            pipe.set_params(classifier__scale_pos_weight=neg / pos)

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_test, y_proba))

        metrics = {
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "ROC-AUC": round(auc, 4),
        }
        results.append(metrics)
        fitted_pipelines[name] = pipe
        print(f"[ML] {name:<20} -> Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

    results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "ml_model_comparison.csv"), index=False)

    # Select best ML model by F1 + ROC-AUC + Recall balance
    results_df["rank_score"] = results_df["F1"] * 0.5 + results_df["ROC-AUC"] * 0.3 + results_df["Recall"] * 0.2
    best_name = results_df.sort_values("rank_score", ascending=False).iloc[0]["Model"]
    best_pipeline = fitted_pipelines[best_name]

    print(f"\n[ML] Best ML Model Selected: {best_name}")

    # Save best ML pipeline
    ml_model_path = os.path.join(MODEL_DIR, "best_ml_pipeline.pkl")
    joblib.dump(best_pipeline, ml_model_path)
    joblib.dump(best_pipeline, os.path.join(MODEL_DIR, "predictive_maintenance_model.pkl"))

    return results, best_name, best_pipeline, (X_test, y_test)


if __name__ == "__main__":
    train_ml()
