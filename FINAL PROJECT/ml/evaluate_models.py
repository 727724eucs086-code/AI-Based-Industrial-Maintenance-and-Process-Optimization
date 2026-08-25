"""
ml/evaluate_models.py

Evaluates and compares traditional ML models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost)
and Deep Learning Neural Network models.

Outputs:
  - outputs/model_comparison.csv
  - outputs/confusion_matrix.png
  - outputs/feature_importance.png
  - models/model_metadata.json
  - models/best_ml_pipeline.pkl
  - models/deep_learning_model.pkl
  - models/predictive_maintenance_model.pkl (production model link)
"""
import datetime
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

from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
)

from train_ml import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET, train_ml
from train_dl import train_dl

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("=" * 70)
    print("   AI BASED INDUSTRIAL MAINTENANCE & PROCESS OPTIMIZATION: EVALUATION   ")
    print("=" * 70)

    # 1. Train traditional ML models
    ml_metrics, best_ml_name, best_ml_pipe, (X_test, y_test) = train_ml()

    # 2. Train Deep Learning model
    dl_metrics, dl_wrapper = train_dl()

    # 3. Combine metrics
    all_metrics = ml_metrics + [dl_metrics]
    metrics_df = pd.DataFrame(all_metrics)

    # Calculate ranking score balancing Failure F1, ROC-AUC, and Recall
    metrics_df["rank_score"] = (
        metrics_df["F1"] * 0.5 + metrics_df["ROC-AUC"] * 0.3 + metrics_df["Recall"] * 0.2
    )
    sorted_df = metrics_df.sort_values("rank_score", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 70)
    print("                    MODEL PERFORMANCE COMPARISON                   ")
    print("=" * 70)
    print(sorted_df[["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]].to_string(index=False))

    best_overall_row = sorted_df.iloc[0]
    selected_model_name = best_overall_row["Model"]

    print("\n" + "=" * 70)
    print(f"  >>> SELECTED PRODUCTION MODEL: {selected_model_name} <<<")
    print(f"  F1-Score: {best_overall_row['F1']:.4f} | ROC-AUC: {best_overall_row['ROC-AUC']:.4f} | Recall: {best_overall_row['Recall']:.4f}")
    print("=" * 70)

    # Save metrics table
    sorted_df[["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]].to_csv(
        os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False
    )

    # 4. Generate Confusion Matrix for selected model
    if "Neural Network" in selected_model_name:
        prod_model = dl_wrapper
    else:
        prod_model = best_ml_pipe

    # Save selected model to main production model path
    joblib.dump(prod_model, os.path.join(MODEL_DIR, "predictive_maintenance_model.pkl"))

    y_pred = prod_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Healthy", "Failure"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix - {selected_model_name}", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=200)
    plt.close()

    # 5. Feature Importance Analysis
    raw_feature_names = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    perm = permutation_importance(
        prod_model, X_test, y_test, n_repeats=10, random_state=42, scoring="f1"
    )

    importance_df = pd.DataFrame({
        "feature": raw_feature_names,
        "importance": np.maximum(0, perm.importances_mean),
    }).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.barh(importance_df["feature"], importance_df["importance"], color="#3b82f6", edgecolor="#1d4ed8")
    ax.set_xlabel("Permutation Importance (Mean F1-Score Decrease)", fontsize=11, fontweight="bold")
    ax.set_title(f"Top Risk Factors - {selected_model_name}", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=200)
    plt.close()

    feature_importances_dict = dict(zip(importance_df["feature"], [round(float(v), 4) for v in importance_df["importance"]]))

    # 6. Save model metadata
    metadata = {
        "model_name": selected_model_name,
        "selected_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "selection_rationale": (
            f"{selected_model_name} was automatically selected as the production model "
            f"because it achieved the highest balanced score across F1-score ({best_overall_row['F1']:.4f}), "
            f"ROC-AUC ({best_overall_row['ROC-AUC']:.4f}), and failure recall ({best_overall_row['Recall']:.4f})."
        ),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_importances": feature_importances_dict,
        "confusion_matrix": cm.tolist(),
        "metrics": sorted_df[["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]].to_dict(orient="records"),
    }

    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[SUCCESS] Pipeline training & evaluation complete!")
    print(f"[SUCCESS] Saved model metadata to {os.path.join(MODEL_DIR, 'model_metadata.json')}")


if __name__ == "__main__":
    main()
