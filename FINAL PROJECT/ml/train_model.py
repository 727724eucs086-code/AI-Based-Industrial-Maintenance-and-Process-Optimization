"""
ml/train_model.py

Trains and compares several classification models on the AI4I 2020
predictive-maintenance dataset, selects the best one (by failure-class
recall / F1 / ROC-AUC - NOT accuracy alone, since failures are rare),
and saves the full preprocessing + model pipeline to
models/predictive_maintenance_model.pkl.

Run:
    python ml/train_model.py

Outputs:
    models/predictive_maintenance_model.pkl   - trained sklearn Pipeline
    outputs/model_comparison.csv              - metrics for every model
    outputs/confusion_matrix.png              - confusion matrix (best model)
    outputs/feature_importance.png            - feature importance (best model)
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
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ai4i2020.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
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

# Columns that would leak the target (they directly encode *why* the
# machine failed, i.e. they are only known once failure has already
# occurred) - these are dropped before modeling, matching the AI4I2020
# leakage-avoidance guidance.
LEAKAGE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF", "UDI", "Product ID"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Missing values:\n{df.isnull().sum().sum()} total")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"Failure rate: {df[TARGET].mean() * 100:.2f}%")
    drop_cols = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    df = df.drop(columns=drop_cols)
    return df


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def get_candidate_models():
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
            n_estimators=200, random_state=RANDOM_STATE, max_depth=3
        ),
    }
    if HAS_XGB:
        # scale_pos_weight approximates class_weight="balanced" for XGBoost
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    return models


def main():
    df = load_data()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nTrain size: {len(X_train)}  Test size: {len(X_test)}")
    print(f"Train failure rate: {y_train.mean()*100:.2f}%  Test failure rate: {y_test.mean()*100:.2f}%")

    preprocessor = build_preprocessor()
    candidates = get_candidate_models()

    results = []
    fitted_pipelines = {}

    for name, clf in candidates.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])

        # XGBoost has no native class_weight="balanced"; approximate it
        # with scale_pos_weight computed from the training split.
        if name == "XGBoost":
            neg, pos = np.bincount(y_train)
            pipe.set_params(classifier__scale_pos_weight=neg / pos)

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
        }
        results.append(metrics)
        fitted_pipelines[name] = pipe

        print(f"\n--- {name} ---")
        print(classification_report(y_test, y_pred, target_names=["Healthy", "Failure"], zero_division=0))

    results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
    print("\n=== Model comparison (sorted by F1 on the failure class) ===")
    print(results_df.to_string(index=False))

    # Select best model: rank primarily by F1 (balances precision/recall on
    # the minority failure class), tie-break with ROC-AUC then Recall -
    # deliberately NOT plain accuracy, since a model that always predicts
    # "healthy" would score >95% accuracy on this imbalanced dataset while
    # being useless in practice.
    results_df["rank_score"] = results_df["F1"] * 0.5 + results_df["ROC-AUC"] * 0.3 + results_df["Recall"] * 0.2
    best_name = results_df.sort_values("rank_score", ascending=False).iloc[0]["Model"]
    best_pipeline = fitted_pipelines[best_name]
    print(f"\n>>> Selected best model: {best_name} <<<")

    # --- Confusion matrix -----------------------------------------------
    y_pred_best = best_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Healthy", "Failure"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    # --- Feature importance (permutation importance works for any model) -
    feature_names = NUMERIC_FEATURES + list(
        best_pipeline.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .get_feature_names_out(CATEGORICAL_FEATURES)
    )
    perm = permutation_importance(
        best_pipeline, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, scoring="f1"
    )
    # permutation_importance operates on raw X columns (5), not the
    # one-hot expanded feature set, since it permutes the *input* columns
    raw_feature_names = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    importance_df = pd.DataFrame({
        "feature": raw_feature_names,
        "importance": perm.importances_mean,
    }).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(importance_df["feature"], importance_df["importance"], color="#2563eb")
    ax.set_xlabel("Permutation importance (mean F1 decrease)")
    ax.set_title(f"Feature Importance - {best_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
    plt.close()

    # --- Save pipeline + metadata -----------------------------------------
    model_path = os.path.join(MODEL_DIR, "predictive_maintenance_model.pkl")
    joblib.dump(best_pipeline, model_path)
    print(f"\nSaved trained pipeline to {model_path}")

    metadata = {
        "model_name": best_name,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "metrics": results_df[["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]].to_dict(orient="records"),
    }
    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata to {os.path.join(MODEL_DIR, 'model_metadata.json')}")


if __name__ == "__main__":
    main()
