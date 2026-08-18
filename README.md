# AI Based Industrial Maintenance and Process Optimization

[![Live Demo on Vercel](https://img.shields.io/badge/Deployed%20on-Vercel-black?style=for-the-badge&logo=vercel)](https://predictive-maintenance-ai.vercel.app)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)

> 🔗 **Live Demo URL**: [https://predictive-maintenance-ai.vercel.app](https://predictive-maintenance-ai.vercel.app)

A complete production-style enterprise AI application for **AI Based Industrial Maintenance and Process Optimization** utilizing the **AI4I 2020 Predictive Maintenance Dataset**.

This system combines machine learning algorithms (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost) and deep learning neural networks (PyTorch / Multi-Layer Perceptron), a Flask REST API backend with SQLite and JWT authentication, and a dark-themed industrial dashboard.

---

## 🌟 Application Features

- **Industrial AI Dashboard**: Real-time KPI monitoring (Total Evaluations, Healthy Count, At Risk Count, Average Failure Risk %, Active Model metric).
- **AI Failure Risk Inference**: Calculates machine failure probability and maps it into intuitive risk categories (**LOW**: 0-30%, **MODERATE**: 30-60%, **HIGH**: 60-80%, **CRITICAL**: 80-100%).
- **Domain Sensor Observations**: Automatic assessment of torque, tool wear, operating temperatures, and rotational speeds.
- **Actionable Maintenance Recommendations**: Dynamic recommendations based on AI risk levels and domain sensor thresholds.
- **Empirical Model Performance & Evaluation**: Real metric comparisons across traditional ML models and Deep Neural Networks (Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix, Feature Importance).
- **Process Optimization & Analytics**: Interactive Chart.js trend charts, doughnut risk breakdowns, machine variant distributions, and high-risk warning logs.
- **Prediction History & Audit**: Full searchable, filterable, and sortable history of evaluations linked to user accounts.
- **Enterprise Security**: Password hashing with Bcrypt, JWT session management, and protected REST API endpoints.

---

## 🏗️ Architecture Flow

```text
Industrial Machine Sensor Data (Type, Temp, RPM, Torque, Wear)
            ↓
      Web Dashboard (HTML5 / Vanilla JS / Chart.js)
            ↓
       Flask REST API (/api/predict)
            ↓
      Input Validation & Data Sanitization
            ↓
       Preprocessing (StandardScaler + OneHotEncoder)
            ↓
   ML / Deep Learning Models (Random Forest / Neural Network)
            ↓
   Failure Risk Prediction & Probability %
            ↓
 Maintenance Recommendation Engine & Sensor Observations
            ↓
       SQLite Database (MachinePrediction ORM)
            ↓
 Analytics + Interactive Charts + History Logs
```

---

## 📊 Dataset Features & Target Leakage Prevention

The system uses the **AI4I 2020 Predictive Maintenance Dataset** (`data/ai4i2020.csv`).

### Input Features Used:
- `Type`: Machine quality variant (`L` — Low 50%, `M` — Medium 30%, `H` — High 20%)
- `Air temperature [K]`: Ambient temperature in Kelvin
- `Process temperature [K]`: Process operating temperature in Kelvin
- `Rotational speed [rpm]`: Spindle speed in RPM
- `Torque [Nm]`: Torque load in Newton-meters
- `Tool wear [min]`: Cumulative tool usage time in minutes

### Target:
- `Machine failure`: Binary flag (`0` = Healthy, `1` = Failure)

### Data Leakage Protection:
Columns that directly reveal the failure cause post-hoc (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`, `UDI`, `Product ID`) are explicitly dropped prior to model training to prevent target leakage.

---

## 🛠️ Technology Stack

- **Frontend**: HTML5, CSS3 (Vanilla CSS with CSS custom properties & Glassmorphic design), Vanilla JavaScript, Chart.js, Font Awesome icons.
- **Backend**: Python, Flask, Flask-SQLAlchemy, Flask-JWT-Extended, Flask-Bcrypt, Flask-CORS.
- **Machine Learning**: Pandas, NumPy, Scikit-Learn, Joblib, XGBoost, Matplotlib, Seaborn.
- **Deep Learning**: PyTorch / Multi-Layer Perceptron Neural Network.
- **Database**: SQLite with SQLAlchemy ORM.

---

## 📁 Project Structure

```text
AI-Based-Industrial-Maintenance-and-Process-Optimization/
│
├── app.py                      # Flask REST API backend server & JWT auth
├── index.html                  # Premium Industrial AI Dashboard frontend
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── predictive_maintenance.db   # SQLite database
│
├── data/
│   └── ai4i2020.csv            # AI4I 2020 Dataset
│
├── ml/
│   ├── train_ml.py             # Script to train traditional ML models
│   ├── train_dl.py             # Script to train Deep Neural Network model
│   └── evaluate_models.py      # ML vs DL evaluation & metadata generator
│
├── models/
│   ├── best_ml_pipeline.pkl    # Best trained traditional ML pipeline
│   ├── deep_learning_model.pkl # Trained Deep Neural Network wrapper
│   ├── dl_preprocessor.pkl     # Preprocessing transformer artifact
│   ├── model_metadata.json     # Saved evaluation metrics & rationale
│   └── predictive_maintenance_model.pkl # Active production model
│
└── outputs/
    ├── model_comparison.csv    # Real metric comparison table
    ├── confusion_matrix.png    # Confusion matrix visual chart
    └── feature_importance.png  # Feature importance bar chart
```

---

## ⚙️ Windows Installation & Setup Instructions

### 1. Open Terminal in Project Directory
```bash
cd "C:\Users\pooji\OneDrive\Desktop\INFOSYS POOJI\AI Based lndustrial Maintenance and Process Optimization"
```

### 2. Create and Activate Virtual Environment (Optional)
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Model Training & Evaluation Execution

To train all ML models, train the Deep Learning neural network, evaluate performance, and generate metadata artifacts, execute:

```bash
python ml/evaluate_models.py
```

Outputs will be saved in `models/` and `outputs/`.

---

## 🌐 Live Deployment (Vercel)

The application is deployed on Vercel Serverless Python environment:
- **Live URL**: [https://predictive-maintenance-ai.vercel.app](https://predictive-maintenance-ai.vercel.app)

---

## 💻 Running the Web Application Locally

Start the Flask REST API backend server:

```bash
python app.py
```

Then open your browser and navigate to:

```text
http://localhost:5000
```

---

## 📡 REST API Endpoint Overview

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `POST` | `/api/register` | Register a new user account | No |
| `POST` | `/api/login` | Authenticate user and receive JWT token | No |
| `POST` | `/api/logout` | Revoke current JWT token | Yes |
| `POST` | `/api/predict` | Submit sensor readings for AI failure prediction | Yes |
| `GET` | `/api/dashboard` | Fetch aggregated KPIs, risk breakdown, trends | Yes |
| `GET` | `/api/history` | Fetch user's prediction history logs | Yes |
| `GET` | `/api/model-info` | Fetch active production model metadata | No |
| `GET` | `/api/model-performance` | Fetch ML vs DL performance metrics & feature importances | No |
| `GET` | `/api/health` | System liveness and model load check | No |

---

## 🔒 Security & Data Integrity

- Passwords are securely hashed using **Bcrypt**.
- Endpoints are protected with **JWT Access Tokens**.
- User predictions are isolated per user in the SQLite database.
- Input data is validated and sanitized on both frontend and backend.
