"""
prepare_data.py — bootstrap the raw dataset.

Converts the synthetic linear data (y = x + 10) into train/train.csv and
test/test.csv so DVC has real data files to version. Run this ONCE to seed
the project; afterwards the CSVs are tracked by DVC and new data is simply
appended to them to trigger retraining.
"""

import os
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# ── Load params ────────────────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

TEST_SIZE = params["data"]["test_size"]
SEED = params["data"]["random_seed"]

# ── Generate synthetic data (same relationship as the original model) ──
# X from -100..96 step 4 -> 50 samples; y = x + 10
X = np.arange(-100, 100, 4)
y = np.arange(-90, 110, 4)
df = pd.DataFrame({"x": X, "y": y})
print(f"Generated {len(df)} samples (columns: {list(df.columns)})")

# ── Split into train / test ────────────────────────────────
train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=SEED)
print(f"Train: {len(train_df)} rows  Test: {len(test_df)} rows")

# ── Write CSVs ─────────────────────────────────────────────
os.makedirs("train", exist_ok=True)
os.makedirs("test", exist_ok=True)
train_df.to_csv("train/train.csv", index=False)
test_df.to_csv("test/test.csv", index=False)
print("Saved: train/train.csv  test/test.csv")
