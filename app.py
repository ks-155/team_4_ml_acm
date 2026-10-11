"""
app.py — Multi-page Flask Application for Food Delivery ETA.
Pages:
  /          -> Estimator
  /map       -> Interactive Route Map
  /analytics -> Model Analytics & Diagnostics
"""
import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder="templates")

# Load pickled model bundle
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
with open(MODEL_PATH, "rb") as f:
    _bundle = pickle.load(f)

MODEL    = _bundle["model"]
FEATURES = _bundle["features"]
RMSE     = float(_bundle.get("rmse", 3.86))
MAE      = float(_bundle.get("mae", 2.63))
R2       = float(_bundle.get("r2", 0.879))

# Model coefficients mapping
COEF_DICT = dict(zip(FEATURES, [round(float(c), 3) for c in MODEL.coef_]))
INTERCEPT = round(float(MODEL.intercept_), 3)

@app.route("/")
@app.route("/estimator")
def estimator_page():
    return render_template("index.html", active_page="estimator")

@app.route("/map")
def map_page():
    return render_template("map.html", active_page="map")

@app.route("/analytics")
def analytics_page():
    metrics = {
        "mae": MAE,
        "rmse": RMSE,
        "r2": R2,
        "intercept": INTERCEPT,
        "coefficients": COEF_DICT
    }
    return render_template("analytics.html", active_page="analytics", metrics=metrics)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        row  = [float(data[f]) for f in FEATURES]
        eta  = float(MODEL.predict([row])[0])
        eta  = max(1.0, eta)
        low  = max(1.0, eta - 1.5 * RMSE)
        high = eta + 1.5 * RMSE
        return jsonify({
            "eta_min"  : round(eta, 1),
            "low_min"  : round(low, 1),
            "high_min" : round(high, 1),
            "ok"       : True
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
