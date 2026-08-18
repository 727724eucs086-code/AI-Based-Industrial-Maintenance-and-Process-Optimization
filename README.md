# AI Based Industrial Maintenance and Process Optimization

[![Live Demo on Vercel](https://img.shields.io/badge/Deployed%20on-Vercel-black?style=for-the-badge&logo=vercel)](https://ai-industrial-maintenance-optimizat.vercel.app)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

> 🔗 **Live Production URL**: [https://ai-industrial-maintenance-optimizat.vercel.app](https://ai-industrial-maintenance-optimizat.vercel.app)  

A production-grade, enterprise AI application for **AI Based Industrial Maintenance and Process Optimization** utilizing the **AI4I 2020 Predictive Maintenance Dataset**.

This platform combines Machine Learning algorithms (*Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost*) with Deep Neural Networks (*PyTorch Multi-Layer Perceptron*), a Flask REST API backend with SQLite and JWT authentication, and a dark-themed glassmorphic industrial operations dashboard.

---

## 🌟 Key Application Features

- 🖥️ **Industrial AI Dashboard**: Real-time KPI monitoring (Total Evaluations, Healthy Machines, At Risk Count, Average Failure Probability %, and Active Production Model status).
- 🧠 **Multi-Model AI Failure Inference**: Real-time machine failure risk scoring mapped into actionable categories:
  - 🟢 **LOW RISK** (0% – 30%): Normal operating parameters.
  - 🟡 **MODERATE RISK** (30% – 60%): Inspection recommended within next scheduled cycle.
  - 🟠 **HIGH RISK** (60% – 80%): Immediate component service and load reduction advised.
  - 🔴 **CRITICAL RISK** (80% – 100%): Urgent safety shutdown and tool replacement required.
- ⚙️ **Domain Sensor Observations**: Automatic anomaly detection across Spindle Speed (RPM), Torque (Nm), Cumulative Tool Wear (min), and Process vs. Air Operating Temperatures (K).
- 🛠️ **Dynamic Maintenance Recommendations**: Context-aware rule-and-AI maintenance guidance generated per prediction.
- 📊 **Process Optimization & Analytics**: Chart.js probability trend curves, doughnut risk distribution, variant breakdown, and recent alert logs.
- 📈 **Empirical Model Benchmarking**: Comparative analysis of traditional ML vs. Deep Learning with Confusion Matrix and Permutation Feature Importance visualizations.
- 📜 **Audit History & Filtering**: Searchable, filterable, and sortable evaluation records per authenticated operator.
- 🔒 **Enterprise Security**: Bcrypt-hashed credentials, JWT Bearer tokens with expiration, and strict API sanitization.

---

## 🏗️ Architecture Flow

```text
Industrial Sensor Readings (Type, Temp, RPM, Torque, Wear)
                          ↓
  Interactive Dashboard UI (HTML5 / Vanilla CSS / Chart.js)
                          ↓
          Flask REST API (/api/predict)
                          ↓
      Input Validation & Data Sanitization
                          ↓
     Preprocessing Pipeline (StandardScaler + OneHotEncoder)
                          ↓
  Best Model Ensemble (Gradient Boosting / Deep Neural Net)
                          ↓
   Failure Probability % + Severity Risk Level Mapping
                          ↓
 Recommendation Engine + Real-time Sensor Observations
                          ↓
      SQLite Database (MachinePrediction ORM)
                          ↓
   Analytics Visuals, Trend Logs & Audit History
```

---

## 📊 Dataset & Target Leakage Prevention

The system is trained and benchmarked on the **AI4I 2020 Predictive Maintenance Dataset** (`data/ai4i2020.csv`).

### Input Features:
| Feature Name | Type | Description |
|---|---|---|
| `Type` | Categorical | Product quality variant (`L` = 50%, `M` = 30%, `H` = 20%) |
| `Air temperature [K]` | Numerical | Ambient air temperature (Kelvin) |
| `Process temperature [K]` | Numerical | Process operating temperature (Kelvin) |
| `Rotational speed [rpm]` | Numerical | Machine spindle rotational velocity |
| `Torque [Nm]` | Numerical | Machine torque load |
| `Tool wear [min]` | Numerical | Cumulative cutter usage duration |

### Target:
- `Machine failure`: Binary classification (`0` = Normal / Healthy, `1` = Machine Failure)

### 🛡️ Target Leakage Protection:
Columns that explicitly reveal post-failure causes (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`, `UDI`, `Product ID`) are dropped during feature preparation to guarantee genuine prospective predictive power.

---

## 🏆 Model Performance & Comparison

The automated model selection pipeline evaluates 6 model architectures on stratified test splits:

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Production Status |
|---|---|---|---|---|---|---|
| **Gradient Boosting** | **98.95%** | **83.87%** | **61.90%** | **0.7123** | **0.9234** | 🥇 **Active Production Model** |
| **XGBoost** | 98.65% | 68.29% | 66.67% | 0.6747 | 0.9220 | Candidate |
| **Random Forest** | 98.65% | 71.43% | 59.52% | 0.6494 | 0.9114 | Candidate |
| **Neural Network (PyTorch MLP)** | 91.35% | 16.75% | 78.57% | 0.2762 | 0.9191 | Candidate |
| **Decision Tree** | 95.15% | 24.30% | 61.90% | 0.3490 | 0.6702 | Baseline |
| **Logistic Regression** | 72.45% | 05.27% | 71.43% | 0.0982 | 0.7224 | Baseline |

---

## 📁 Project Structure

```text
AI-Based-Industrial-Maintenance-and-Process-Optimization/
│
├── app.py                      # Flask REST API backend & JWT auth handlers
├── index.html                  # Responsive Industrial AI Dashboard frontend
├── requirements.txt            # Python package dependencies
├── vercel.json                 # Vercel Serverless deployment configuration
├── render.yaml                 # Render cloud deployment specification
├── Procfile                    # Gunicorn production process definition
├── README.md                   # Comprehensive project documentation
├── predictive_maintenance.db   # SQLite database & prediction audit store
│
├── data/
│   ├── ai4i2020.csv            # AI4I 2020 Industrial Dataset
│   └── generate_dataset.py     # Dataset generator script
│
├── ml/
│   ├── train_ml.py             # Traditional ML models training & tuning
│   ├── train_dl.py             # Deep Neural Network (PyTorch) training
│   ├── evaluate_models.py      # Automated benchmarking & model selection
│   └── evaluate_model.py       # Single-model validation script
│
├── models/
│   ├── best_ml_pipeline.pkl    # Serialized Gradient Boosting pipeline
│   ├── deep_learning_model.pkl # Serialized PyTorch neural network model
│   ├── dl_preprocessor.pkl     # Deep learning preprocessing pipeline
│   ├── model_metadata.json     # Benchmarking results & feature importances
│   └── predictive_maintenance_model.pkl # Active production model
│
└── outputs/
    ├── model_comparison.csv    # Real metric comparison table
    ├── confusion_matrix.png    # Production model confusion matrix
    └── feature_importance.png  # Permutation feature importance chart
```

---

## 🌐 Live Cloud Deployment

The application is deployed on **Vercel**:

- **Production URL**: [https://ai-industrial-maintenance-optimizat.vercel.app](https://ai-industrial-maintenance-optimizat.vercel.app)
- **Deployment URL**: [https://ai-industrial-maintenance-optimization-dm42a6wub.vercel.app](https://ai-industrial-maintenance-optimization-dm42a6wub.vercel.app)
- **Vercel Dashboard**: [https://vercel.com/poojitha6002-3558s-projects/ai-industrial-maintenance-optimization](https://vercel.com/poojitha6002-3558s-projects/ai-industrial-maintenance-optimization)

To deploy updates using your Vercel Token:
```bash
npx -y vercel --token <YOUR_VERCEL_TOKEN> --prod --yes
```

---

## 💻 Local Installation & Setup

### 1. Clone or Open Project Directory
```bash
cd "C:\Users\pooji\OneDrive\Desktop\INFOSYS POOJI\AI Based lndustrial Maintenance and Process Optimization"
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Model Training & Evaluation (Optional)
```bash
python ml/evaluate_models.py
```

### 5. Launch the Web Application
```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser to access the dashboard.

---

## 📡 REST API Reference

| Method | Endpoint | Description | Auth |
|---|---|---|---|
| `POST` | `/api/register` | Create a new operator account (`username`, `email`, `password`) | No |
| `POST` | `/api/login` | Authenticate user and receive JWT access token | No |
| `POST` | `/api/logout` | Revoke current JWT session | Yes |
| `POST` | `/api/predict` | Run AI failure inference on 5 sensor metrics + machine type | Yes |
| `GET` | `/api/dashboard` | Fetch aggregated KPIs, risk breakdown, trends, and recent records | Yes |
| `GET` | `/api/history` | Query user's historical evaluations | Yes |
| `GET` | `/api/model-info` | Retrieve active production model metadata & importances | No |
| `GET` | `/api/model-performance`| Fetch benchmark metrics for ML vs Deep Learning models | No |
| `GET` | `/api/health` | Service liveness and model load status | No |

---

## 🔒 Security & Data Governance

- **Password Hashing**: Secure salted hashes using `Flask-Bcrypt`.
- **Stateless Authentication**: JWT tokens with 12-hour expiration windows.
- **Tenant Isolation**: Evaluation records are isolated per authenticated user ID.
- **Input Sanitization**: Numerical range verification and categorical validation across both frontend and backend.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

