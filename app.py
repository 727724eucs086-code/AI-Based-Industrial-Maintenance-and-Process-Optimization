"""
app.py

Flask REST API backend for AI Based Industrial Maintenance & Process Optimization.

Endpoints:
    POST /api/register          - Create a new user (bcrypt-hashed password)
    POST /api/login             - Authenticate user, return JWT access token
    POST /api/logout            - Revoke current JWT access token
    POST /api/predict           - Perform AI failure prediction on machine sensor data
    GET  /api/dashboard         - Aggregated KPIs, risk distribution & trends
    GET  /api/history           - History of predictions for logged-in user
    GET  /api/model-info        - Selected production model metadata & feature importances
    GET  /api/model-performance - ML vs Deep Learning metrics, confusion matrix & feature rank
    GET  /api/health            - Liveness & model status check
"""
import json
import os
import re
from datetime import datetime, timedelta, timezone

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    get_jwt,
    get_jwt_identity,
    jwt_required,
)
from flask_sqlalchemy import SQLAlchemy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=None)

if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
    db_file = os.path.join("/tmp", "predictive_maintenance.db")
    src_db = os.path.join(BASE_DIR, "predictive_maintenance.db")
    if not os.path.exists(db_file) and os.path.exists(src_db):
        import shutil
        shutil.copy2(src_db, db_file)
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + db_file
else:
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(
        BASE_DIR, "predictive_maintenance.db"
    )

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["JWT_SECRET_KEY"] = os.environ.get(
    "JWT_SECRET_KEY", "predict-ai-industrial-saas-secret-key-2026"
)
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=12)

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)
CORS(app)

# Load production model & metadata
MODEL_PATH = os.path.join(BASE_DIR, "models", "predictive_maintenance_model.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "models", "model_metadata.json")

ml_pipeline = None
model_metadata = {}
model_load_error = None

def load_system_model():
    global ml_pipeline, model_metadata, model_load_error
    try:
        if os.path.exists(MODEL_PATH):
            ml_pipeline = joblib.load(MODEL_PATH)
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "r") as f:
                model_metadata = json.load(f)
    except Exception as exc:
        model_load_error = str(exc)
        print(f"[WARNING] Could not load model: {exc}")

load_system_model()

VALID_TYPES = {"L", "M", "H"}
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
JWT_BLOCKLIST = set()


@jwt.token_in_blocklist_loader
def check_if_token_revoked(jwt_header, jwt_payload):
    return jwt_payload["jti"] in JWT_BLOCKLIST


# ----------------------------------------------------------------------------
# Database models
# ----------------------------------------------------------------------------
class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    predictions = db.relationship(
        "MachinePrediction", backref="user", lazy=True, cascade="all, delete-orphan"
    )

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "created_at": self.created_at.isoformat(),
        }


class MachinePrediction(db.Model):
    __tablename__ = "machine_predictions"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    machine_type = db.Column(db.String(1), nullable=False)
    air_temperature = db.Column(db.Float, nullable=False)
    process_temperature = db.Column(db.Float, nullable=False)
    rotational_speed = db.Column(db.Float, nullable=False)
    torque = db.Column(db.Float, nullable=False)
    tool_wear = db.Column(db.Float, nullable=False)
    prediction_result = db.Column(db.String(40), nullable=False)
    failure_probability = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(20), nullable=False, default="LOW")
    model_used = db.Column(db.String(80), nullable=False, default="Machine Learning Model")
    recommendation = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            "id": self.id,
            "machine_type": self.machine_type,
            "air_temperature": self.air_temperature,
            "process_temperature": self.process_temperature,
            "rotational_speed": self.rotational_speed,
            "torque": self.torque,
            "tool_wear": self.tool_wear,
            "prediction_result": self.prediction_result,
            "failure_probability": self.failure_probability,
            "risk_level": self.risk_level,
            "model_used": self.model_used,
            "recommendation": self.recommendation,
            "created_at": self.created_at.isoformat(),
        }


with app.app_context():
    db.create_all()
    try:
        db.session.execute(db.text("ALTER TABLE machine_predictions ADD COLUMN risk_level VARCHAR(20) DEFAULT 'LOW'"))
        db.session.commit()
    except Exception:
        db.session.rollback()
    try:
        db.session.execute(db.text("ALTER TABLE machine_predictions ADD COLUMN model_used VARCHAR(80) DEFAULT 'Machine Learning Model'"))
        db.session.commit()
    except Exception:
        db.session.rollback()


# ----------------------------------------------------------------------------
# Helper logic
# ----------------------------------------------------------------------------
def error_response(message, status=400):
    return jsonify({"error": message}), status


def calculate_risk_level(prob):
    if prob >= 80.0:
        return "CRITICAL"
    elif prob >= 60.0:
        return "HIGH"
    elif prob >= 30.0:
        return "MODERATE"
    else:
        return "LOW"


def evaluate_sensor_observations(air_temp, process_temp, rot_speed, torque, tool_wear):
    """Domain threshold observations separate from model predictions."""
    obs = {}

    # Torque observation
    if torque >= 60.0:
        obs["torque"] = {"status": "Critical", "note": "Extremely high torque load."}
    elif torque >= 50.0:
        obs["torque"] = {"status": "Warning", "note": "Elevated torque."}
    else:
        obs["torque"] = {"status": "Normal", "note": "Within standard operating range."}

    # Tool Wear observation
    if tool_wear >= 200.0:
        obs["tool_wear"] = {"status": "Critical", "note": "Severe tool wear (>200 min). Replacement required."}
    elif tool_wear >= 150.0:
        obs["tool_wear"] = {"status": "Warning", "note": "High tool wear (>150 min). Inspect tooling."}
    else:
        obs["tool_wear"] = {"status": "Normal", "note": "Acceptable wear level."}

    # Temperature observation
    temp_diff = process_temp - air_temp
    if temp_diff < 8.6 and rot_speed < 1380:
        obs["temperature"] = {"status": "Warning", "note": "Low heat dissipation margin detected."}
    elif air_temp >= 303.0 or process_temp >= 313.0:
        obs["temperature"] = {"status": "Elevated", "note": "High operating temperature."}
    else:
        obs["temperature"] = {"status": "Normal", "note": "Thermal margin optimal."}

    # Rotational Speed observation
    if rot_speed < 1200 or rot_speed > 2800:
        obs["rotational_speed"] = {"status": "Warning", "note": "Abnormal rotational speed."}
    else:
        obs["rotational_speed"] = {"status": "Normal", "note": "Speed within standard bounds."}

    return obs


def build_recommendation(machine_type, air_temp, process_temp, rot_speed, torque, tool_wear, failure_prob):
    recs = []

    if failure_prob >= 80.0:
        recs.append("CRITICAL RISK: Stop or isolate machine immediately according to site safety procedures and perform comprehensive physical inspection.")
    elif failure_prob >= 60.0:
        recs.append("HIGH RISK: Schedule targeted maintenance before the next production cycle.")
    elif failure_prob >= 30.0:
        recs.append("MODERATE RISK: Plan closer inspection during upcoming scheduled maintenance window.")

    if tool_wear >= 180.0:
        recs.append("Inspect and replace worn cutting tool bit to prevent tool wear failure.")
    if torque >= 55.0:
        recs.append("Inspect drive motor bearings, gearboxes, and mechanical resistance.")
    if process_temp - air_temp < 8.6 and rot_speed < 1380:
        recs.append("Inspect cooling fluid levels and thermal management systems.")
    if rot_speed < 1200 or rot_speed > 2800:
        recs.append("Inspect speed control drive, inverter calibration, and mechanical coupling.")

    if not recs:
        return "Machine parameters healthy. Continue normal operation and routine monitoring."
    return " ".join(recs)


# ----------------------------------------------------------------------------
# Authentication API
# ----------------------------------------------------------------------------
@app.route("/api/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not username or not email or not password:
        return error_response("Username, email, and password are required.")
    if len(username) < 3:
        return error_response("Username must be at least 3 characters long.")
    if not EMAIL_RE.match(email):
        return error_response("Please enter a valid email address.")
    if len(password) < 6:
        return error_response("Password must be at least 6 characters long.")

    if User.query.filter_by(username=username).first():
        return error_response("Username is already taken.", 409)
    if User.query.filter_by(email=email).first():
        return error_response("An account with that email already exists.", 409)

    password_hash = bcrypt.generate_password_hash(password).decode("utf-8")
    user = User(username=username, email=email, password_hash=password_hash)

    try:
        db.session.add(user)
        db.session.commit()
    except Exception:
        db.session.rollback()
        return error_response("Failed to create account. Please try again.", 500)

    access_token = create_access_token(identity=str(user.id))
    return jsonify({
        "message": "Registration successful.",
        "access_token": access_token,
        "user": user.to_dict(),
    }), 201


@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    identifier = (data.get("username") or data.get("email") or "").strip()
    password = data.get("password") or ""

    if not identifier or not password:
        return error_response("Username/email and password are required.")

    user = User.query.filter(
        (User.username == identifier) | (User.email == identifier.lower())
    ).first()

    if not user or not bcrypt.check_password_hash(user.password_hash, password):
        return error_response("Invalid credentials.", 401)

    access_token = create_access_token(identity=str(user.id))
    return jsonify({"access_token": access_token, "user": user.to_dict()})


@app.route("/api/logout", methods=["POST"])
@jwt_required()
def logout():
    jti = get_jwt()["jti"]
    JWT_BLOCKLIST.add(jti)
    return jsonify({"message": "Logged out successfully."})


# ----------------------------------------------------------------------------
# AI Inference & Prediction Route
# ----------------------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
@jwt_required()
def predict():
    if ml_pipeline is None:
        load_system_model()
    if ml_pipeline is None:
        return error_response(
            "Prediction model not loaded on server. Run 'python ml/evaluate_models.py' first.", 503
        )

    data = request.get_json(silent=True) or {}
    required_fields = [
        "machine_type",
        "air_temperature",
        "process_temperature",
        "rotational_speed",
        "torque",
        "tool_wear",
    ]
    missing = [f for f in required_fields if data.get(f) in (None, "")]
    if missing:
        return error_response(f"Missing required field(s): {', '.join(missing)}")

    machine_type = str(data["machine_type"]).strip().upper()
    if machine_type not in VALID_TYPES:
        return error_response("machine_type must be one of: L, M, H.")

    try:
        air_temperature = float(data["air_temperature"])
        process_temperature = float(data["process_temperature"])
        rotational_speed = float(data["rotational_speed"])
        torque = float(data["torque"])
        tool_wear = float(data["tool_wear"])
    except (TypeError, ValueError):
        return error_response("Sensor readings must be numeric values.")

    input_df = pd.DataFrame([{
        "Air temperature [K]": air_temperature,
        "Process temperature [K]": process_temperature,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "Type": machine_type,
    }])

    try:
        if hasattr(ml_pipeline, "predict_proba"):
            proba_arr = ml_pipeline.predict_proba(input_df)
            failure_probability = float(proba_arr[0][1]) * 100.0
        else:
            failure_probability = 50.0

        if hasattr(ml_pipeline, "predict"):
            pred_class = int(ml_pipeline.predict(input_df)[0])
        else:
            pred_class = 1 if failure_probability >= 50.0 else 0

    except Exception as exc:
        return error_response(f"AI Prediction inference failed: {exc}", 500)

    prediction_result = "Machine Failure Likely" if pred_class == 1 or failure_probability >= 50.0 else "Machine Healthy"
    risk_level = calculate_risk_level(failure_probability)
    prod_model_name = model_metadata.get("model_name", "AI Prediction Model")

    recommendation = build_recommendation(
        machine_type, air_temperature, process_temperature, rotational_speed,
        torque, tool_wear, failure_probability,
    )

    sensor_obs = evaluate_sensor_observations(
        air_temperature, process_temperature, rotational_speed, torque, tool_wear
    )

    user_id = int(get_jwt_identity())
    record = MachinePrediction(
        user_id=user_id,
        machine_type=machine_type,
        air_temperature=air_temperature,
        process_temperature=process_temperature,
        rotational_speed=rotational_speed,
        torque=torque,
        tool_wear=tool_wear,
        prediction_result=prediction_result,
        failure_probability=round(failure_probability, 2),
        risk_level=risk_level,
        model_used=prod_model_name,
        recommendation=recommendation,
    )

    try:
        db.session.add(record)
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        return error_response(f"Prediction completed but failed to persist record: {exc}", 500)

    return jsonify({
        "prediction": {
            "prediction_result": prediction_result,
            "failure_probability": round(failure_probability, 2),
            "risk_level": risk_level,
            "model_used": prod_model_name,
            "recommendation": recommendation,
            "sensor_observations": sensor_obs,
        },
        "record": record.to_dict(),
    })


# ----------------------------------------------------------------------------
# Analytics & Dashboard Routes
# ----------------------------------------------------------------------------
@app.route("/api/dashboard", methods=["GET"])
@jwt_required()
def dashboard():
    user_id = int(get_jwt_identity())
    records = (
        MachinePrediction.query.filter_by(user_id=user_id)
        .order_by(MachinePrediction.created_at.asc())
        .all()
    )

    total = len(records)
    healthy = sum(1 for r in records if r.prediction_result == "Machine Healthy")
    at_risk = total - healthy
    avg_probability = round(sum(r.failure_probability for r in records) / total, 2) if total else 0.0

    # Risk level distribution
    risk_distribution = {"LOW": 0, "MODERATE": 0, "HIGH": 0, "CRITICAL": 0}
    for r in records:
        risk_distribution[r.risk_level] = risk_distribution.get(r.risk_level, 0) + 1

    # Machine Type distribution
    type_distribution = {"L": 0, "M": 0, "H": 0}
    for r in records:
        type_distribution[r.machine_type] = type_distribution.get(r.machine_type, 0) + 1

    # Risk by type
    risk_by_type = {}
    for r in records:
        risk_by_type.setdefault(r.machine_type, []).append(r.failure_probability)
    risk_by_type = {t: round(sum(v)/len(v), 2) for t, v in risk_by_type.items()}

    # Trend line data (last 30 predictions)
    trend = [
        {"created_at": r.created_at.isoformat(), "failure_probability": r.failure_probability, "risk_level": r.risk_level}
        for r in records[-30:]
    ]

    recent = [r.to_dict() for r in records[::-1][:5]]
    high_risk_recent = [r.to_dict() for r in records[::-1] if r.failure_probability >= 50.0][:5]

    production_model = model_metadata.get("model_name", "AI Prediction Pipeline")
    prod_metrics = model_metadata.get("metrics", [{}])[0]

    return jsonify({
        "total_predictions": total,
        "healthy_count": healthy,
        "at_risk_count": at_risk,
        "average_failure_probability": avg_probability,
        "risk_distribution": risk_distribution,
        "machine_type_distribution": type_distribution,
        "average_risk_by_type": risk_by_type,
        "failure_probability_trend": trend,
        "recent_predictions": recent,
        "high_risk_predictions": high_risk_recent,
        "production_model": production_model,
        "model_performance_summary": prod_metrics,
    })


@app.route("/api/history", methods=["GET"])
@jwt_required()
def history():
    user_id = int(get_jwt_identity())
    records = (
        MachinePrediction.query.filter_by(user_id=user_id)
        .order_by(MachinePrediction.created_at.desc())
        .all()
    )
    return jsonify({"history": [r.to_dict() for r in records]})


@app.route("/api/model-info", methods=["GET"])
def model_info():
    if not model_metadata:
        load_system_model()
    return jsonify(model_metadata)


@app.route("/api/model-performance", methods=["GET"])
def model_performance():
    if not model_metadata:
        load_system_model()
    return jsonify({
        "production_model": model_metadata.get("model_name", "Selected Model"),
        "selection_rationale": model_metadata.get("selection_rationale", ""),
        "selected_at": model_metadata.get("selected_at", ""),
        "metrics": model_metadata.get("metrics", []),
        "confusion_matrix": model_metadata.get("confusion_matrix", []),
        "feature_importances": model_metadata.get("feature_importances", {}),
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": ml_pipeline is not None,
        "production_model": model_metadata.get("model_name", "None"),
        "model_load_error": model_load_error,
    })


@app.route("/")
def serve_index():
    return send_from_directory(BASE_DIR, "index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
