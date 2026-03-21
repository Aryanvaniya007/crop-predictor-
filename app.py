"""
CropSense — Flask API
======================
Backend for crop_prediction.html frontend.
Loads svm_crop_model.pkl and exposes:

  POST /predict   → crop recommendation + probabilities
  GET  /health    → API status check
  GET  /crops     → list of all 22 supported crops
  GET  /           → serves crop_prediction.html directly (optional)

Run locally:
  pip install flask flask-cors scikit-learn numpy
  python app.py

Deploy (Render / Railway):
  gunicorn app:app
"""

import os
import pickle
import logging
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────
app = Flask(__name__, static_folder=".")
CORS(app)  # allow requests from any origin (frontend on different port/domain)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Load model bundle
# ─────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "svm_crop_model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        bundle  = pickle.load(f)
    model   = bundle["model"]
    scaler  = bundle["scaler"]
    encoder = bundle["encoder"]
    log.info(f"Model loaded from '{MODEL_PATH}'  |  Classes: {list(encoder.classes_)}")
except FileNotFoundError:
    log.error(f"Model file '{MODEL_PATH}' not found. Run crop_prediction_svm.py first.")
    model = scaler = encoder = None

# ─────────────────────────────────────────────
# Feature config
# ─────────────────────────────────────────────
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
RANGES   = {
    "N":           (0,   140),
    "P":           (5,   145),
    "K":           (5,   205),
    "temperature": (8,   44),
    "humidity":    (14,  100),
    "ph":          (3.5, 10),
    "rainfall":    (20,  300),
}

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def validate_inputs(data: dict):
    """
    Validate all 7 required fields.
    Returns (values_dict, error_message).
    error_message is None if valid.
    """
    values = {}
    for field, (lo, hi) in RANGES.items():
        if field not in data:
            return None, f"Missing field: '{field}'"
        try:
            v = float(data[field])
        except (ValueError, TypeError):
            return None, f"Field '{field}' must be a number."
        if not (lo <= v <= hi):
            return None, f"Field '{field}' = {v} is out of range [{lo}, {hi}]."
        values[field] = v
    return values, None


def run_prediction(values: dict):
    """
    Scale inputs and run SVM prediction.
    Returns dict with crop name and sorted probability dict.
    """
    arr    = np.array([[values[f] for f in FEATURES]])
    scaled = scaler.transform(arr)

    # Predicted class
    idx  = model.predict(scaled)[0]
    crop = encoder.inverse_transform([idx])[0]

    # Probabilities for all classes
    proba = model.predict_proba(scaled)[0]
    prob_dict = {
        cls: round(float(p) * 100, 2)
        for cls, p in zip(encoder.classes_, proba)
    }
    # Sort descending
    prob_sorted = dict(sorted(prob_dict.items(), key=lambda x: -x[1]))

    return {
        "crop":          crop,
        "confidence":    round(prob_sorted[crop], 2),
        "probabilities": prob_sorted,
    }


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Health check — used by the frontend Test button."""
    if model is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 503
    return jsonify({
        "status":  "ok",
        "model":   "SVM (RBF kernel)",
        "classes": len(encoder.classes_),
    }), 200


@app.route("/crops", methods=["GET"])
def crops():
    """Return list of all supported crop classes."""
    if encoder is None:
        return jsonify({"error": "Model not loaded"}), 503
    return jsonify({"crops": sorted(encoder.classes_.tolist())}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint.

    Accepts JSON body:
    {
        "N": 90, "P": 42, "K": 43,
        "temperature": 20.9,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.9
    }

    Returns:
    {
        "crop": "rice",
        "confidence": 86.48,
        "probabilities": { "rice": 86.48, "jute": 9.53, ... }
    }
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Run crop_prediction_svm.py first."}), 503

    # Parse request body
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    # Validate
    values, err = validate_inputs(data)
    if err:
        return jsonify({"error": err}), 422

    # Predict
    try:
        result = run_prediction(values)
        log.info(f"Prediction → {result['crop']}  ({result['confidence']}%)")
        return jsonify(result), 200
    except Exception as e:
        log.exception("Prediction failed")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/", methods=["GET"])
def serve_frontend():
    """
    Optionally serve crop_prediction.html directly from this folder.
    Useful when you want a single-command deploy with no separate static host.
    """
    html_file = "crop_prediction.html"
    if os.path.exists(html_file):
        return send_from_directory(".", html_file)
    return jsonify({"message": "CropSense API is running. POST to /predict"}), 200


# ─────────────────────────────────────────────
# Error handlers
# ─────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    log.info(f"Starting CropSense API on port {port}  |  debug={debug}")
    app.run(host="0.0.0.0", port=port, debug=debug)