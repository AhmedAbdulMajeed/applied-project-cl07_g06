"""
evaluate.py — evaluation stage of the DVC pipeline.

Loads the trained model and the held-out test set, computes regression
metrics, and writes metrics.json (tracked by DVC) plus a metrics.txt summary
and a predictions-vs-actual plot.
"""

import os
import sys
import json
from datetime import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Load hyperparameters ───────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

METRICS_PATH = params["evaluate"]["metrics_path"]
EPOCHS = params["model"]["epochs"]

# ── Check prerequisites ────────────────────────────────────
for path in ["test/test.csv", "models/model.keras",
             "artifacts/feature_columns.json", "artifacts/training_history.json"]:
    if not os.path.exists(path):
        print(f"ERROR: {path} not found — run model.py first")
        sys.exit(1)

# ── Load test data, aligned to training feature columns ────
with open("artifacts/feature_columns.json", encoding="utf-8") as f:
    feature_columns = json.load(f)

dtest = pd.read_csv("test/test.csv")
y_test = dtest["y"].values
X_test = dtest[feature_columns].values
print(f"X_test: {X_test.shape}  y_test: {y_test.shape}")

with open("artifacts/training_history.json") as f:
    history_dict = json.load(f)

# ── Load model and predict ─────────────────────────────────
model = tf.keras.models.load_model("models/model.keras")
print("Model loaded successfully")
preds = model.predict(X_test, verbose=0).flatten()

# ── Metrics ────────────────────────────────────────────────
mse_val = mean_squared_error(y_test, preds)
mae_val = mean_absolute_error(y_test, preds)
r2_val = r2_score(y_test, preds)
rmse_val = float(np.sqrt(mse_val))
print(f"MSE: {mse_val:.4f}  RMSE: {rmse_val:.4f}  MAE: {mae_val:.4f}  R2: {r2_val:.4f}")

# ── Predictions vs actual plot ─────────────────────────────
plt.figure(figsize=(8, 6))
plt.scatter(y_test, preds, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("Predictions vs True Values")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("predictions_vs_actual.png", dpi=120, bbox_inches="tight")
plt.savefig("artifacts/predictions_vs_actual.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: predictions_vs_actual.png")

# ── metrics.txt (human-readable) ───────────────────────────
with open("metrics.txt", "w") as f:
    f.write("=" * 50 + "\n")
    f.write("MODEL PERFORMANCE METRICS\n")
    f.write("=" * 50 + "\n")
    f.write(f"MSE:  {mse_val:.4f}\n")
    f.write(f"RMSE: {rmse_val:.4f}\n")
    f.write(f"MAE:  {mae_val:.4f}\n")
    f.write(f"R2:   {r2_val:.4f}\n")
    f.write("=" * 50 + "\n")
print("Saved: metrics.txt")

# ── metrics.json (read by DVC) ─────────────────────────────
metrics_out = {
    "timestamp": datetime.now().isoformat(),
    "model_type": "Linear Regression",
    "epochs": EPOCHS,
    "metrics": {
        "mean_squared_error": float(mse_val),
        "root_mean_squared_error": rmse_val,
        "mean_absolute_error": float(mae_val),
        "r2_score": float(r2_val),
    },
    "training_history": {
        "final_train_loss": float(history_dict["loss"][-1]),
        "final_val_loss": float(history_dict["val_loss"][-1]),
    },
}
with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics_out, f, indent=4)
print(f"Saved: {METRICS_PATH}")
