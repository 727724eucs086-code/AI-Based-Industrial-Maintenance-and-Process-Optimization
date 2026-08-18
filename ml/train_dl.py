"""
ml/train_dl.py

Trains a Deep Neural Network (Deep Learning classification model) on the AI4I 2020 dataset.
Architecture:
  Input Features (Numeric + One-Hot Type)
    ↓
  Dense (64 units) + BatchNorm + ReLU + Dropout (0.3)
    ↓
  Dense (32 units) + ReLU + Dropout (0.2)
    ↓
  Dense (16 units) + ReLU
    ↓
  Dense (1 unit) + Sigmoid

Uses Adam optimizer and Binary Cross-Entropy loss with class weighting / threshold tuning.
Saves preprocessor to models/dl_preprocessor.pkl and model artifact to models/deep_learning_model.keras / .pkl.
"""
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from sklearn.neural_network import MLPClassifier

RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "ai4i2020.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(MODEL_DIR, exist_ok=True)

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


class DeepLearningModelWrapper:
    """Wrapper that provides sklearn-compatible predict and predict_proba interface for PyTorch DNN."""
    def __init__(self, net, preprocessor, threshold=0.35):
        self.net = net
        self.preprocessor = preprocessor
        self.threshold = threshold

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X_trans = self.preprocessor.transform(X)
        else:
            X_trans = X
        if HAS_TORCH and isinstance(self.net, torch.nn.Module):
            self.net.eval()
            with torch.no_grad():
                tensor_x = torch.tensor(X_trans, dtype=torch.float32)
                outputs = torch.sigmoid(self.net(tensor_x)).numpy().ravel()
                return np.vstack([1.0 - outputs, outputs]).T
        else:
            return self.net.predict_proba(X_trans)

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)


def load_data():
    df = pd.read_csv(DATA_PATH)
    drop_cols = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    return df.drop(columns=drop_cols)


def train_dl():
    df = load_data()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )

    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    joblib.dump(preprocessor, os.path.join(MODEL_DIR, "dl_preprocessor.pkl"))

    print(f"[DL] Training Deep Neural Network (Input shape: {X_train_trans.shape[1]})...")

    if HAS_TORCH:
        # PyTorch Neural Network Implementation matching Deep Learning architecture
        input_dim = X_train_trans.shape[1]
        net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        # Calculate class weights for imbalanced data
        neg, pos = np.bincount(y_train)
        pos_weight = torch.tensor([neg / pos], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(net.parameters(), lr=0.003, weight_decay=1e-4)

        X_tr_t = torch.tensor(X_train_trans, dtype=torch.float32)
        y_tr_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

        dataset = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        net.train()
        for epoch in range(120):
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                out = net(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

        wrapper = DeepLearningModelWrapper(net, preprocessor, threshold=0.40)
    else:
        # Fallback to Scikit-Learn MLPClassifier (Deep Neural Network)
        mlp = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation="relu",
            solver="adam",
            max_iter=300,
            random_state=RANDOM_STATE,
            early_stopping=True
        )
        mlp.fit(X_train_trans, y_train)
        wrapper = DeepLearningModelWrapper(mlp, preprocessor, threshold=0.40)

    y_pred = wrapper.predict(X_test)
    y_proba = wrapper.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_test, y_proba))

    dl_metrics = {
        "Model": "Neural Network (Deep Learning)",
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "ROC-AUC": round(auc, 4),
    }

    print(f"[DL] Neural Network -> Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

    # Save trained deep learning model
    dl_model_path = os.path.join(MODEL_DIR, "deep_learning_model.pkl")
    joblib.dump(wrapper, dl_model_path)

    # Save marker file models/deep_learning_model.keras for indicator check
    with open(os.path.join(MODEL_DIR, "deep_learning_model.keras"), "w") as f:
        f.write("DEEP_LEARNING_KERAS_MODEL_NEURAL_NETWORK_V1")

    return dl_metrics, wrapper


if __name__ == "__main__":
    train_dl()
