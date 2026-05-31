"""
model.py — training stage of the DVC pipeline.

Reads hyperparameters from params.yaml and training data from train/train.csv,
trains a simple linear-regression network, and saves the model plus artifacts
(feature columns, training history) for the evaluate stage.
"""

import os
import sys
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import tensorflow as tf

# ── Load hyperparameters ───────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

EPOCHS = params["model"]["epochs"]
LEARNING_RATE = params["model"]["learning_rate"]
LOSS = params["model"]["loss"]
VAL_SPLIT = params["model"]["validation_split"]
SEED = params["data"]["random_seed"]

# ── Directories ────────────────────────────────────────────
os.makedirs("artifacts", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ── Load data ──────────────────────────────────────────────
if not os.path.exists("train/train.csv"):
    print("ERROR: train/train.csv not found — run src/prepare_data.py first")
    sys.exit(1)

data = pd.read_csv("train/train.csv")
print(f"Train shape: {data.shape}")

if "y" not in data.columns:
    print("ERROR: 'y' target column not found! Columns:", data.columns.tolist())
    sys.exit(1)

X = data.drop("y", axis=1).values
y = data["y"].values

# Save feature columns so evaluate.py aligns the test set identically
feature_columns = list(data.drop("y", axis=1).columns)
with open("artifacts/feature_columns.json", "w", encoding="utf-8") as f:
    json.dump(feature_columns, f, indent=4)
print(f"Saved: artifacts/feature_columns.json ({feature_columns})")

# ── Build model ────────────────────────────────────────────
tf.random.set_seed(SEED)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(X.shape[1],), name="linear_layer")
])
model.compile(
    loss=LOSS,
    optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE),
    metrics=["mae"],
)
model.summary()

# ── Train ──────────────────────────────────────────────────
print("\nTraining model...")
history = model.fit(
    X, y,
    epochs=EPOCHS,
    validation_split=VAL_SPLIT,
    verbose=1,
)
print("Training completed!")

# ── Save model ─────────────────────────────────────────────
model.save("models/model.keras")
print("Saved: models/model.keras")

# ── Save training history for evaluate.py ──────────────────
history_dict = {
    "loss": [float(v) for v in history.history["loss"]],
    "val_loss": [float(v) for v in history.history["val_loss"]],
}
with open("artifacts/training_history.json", "w", encoding="utf-8") as f:
    json.dump(history_dict, f, indent=4)
print("Saved: artifacts/training_history.json")

# ── Training history plot ──────────────────────────────────
plt.figure(figsize=(8, 5))
plt.plot(history_dict["loss"], label="Training Loss", linewidth=2)
plt.plot(history_dict["val_loss"], label="Validation Loss", linewidth=2)
plt.title("Model Loss (MAE)", fontweight="bold")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("model_results.png", dpi=120, bbox_inches="tight")
plt.savefig("artifacts/model_results.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: model_results.png")
print("\nmodel.py complete — run evaluate.py next")
