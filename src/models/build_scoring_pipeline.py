"""
build_scoring_pipeline.py — Serialize the best available model as scoring_pipeline.pkl.

Priority: lightgbm_tuned.pkl > lightgbm_model.pkl

Output: models/scoring_pipeline.pkl
    {
        'model'          : trained LGBMClassifier,
        'feature_columns': list[str],   # all features expected at inference
        'threshold'      : float,       # decision threshold (0.48)
        'version'        : str,
    }
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd
import joblib

PROJECT_ROOT = Path("C:/Users/magno/OneDrive/Desktop/pod-bank-credit-score")
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_DIR   = PROJECT_ROOT / "models"
DATA_PATH   = PROJECT_ROOT / "data/processed/train_final.parquet"
OUTPUT_PATH = MODEL_DIR / "scoring_pipeline.pkl"

# ── Load best model ────────────────────────────────────────────────────────────
tuned_path = MODEL_DIR / "lightgbm_tuned.pkl"
base_path  = MODEL_DIR / "lightgbm_model.pkl"

if tuned_path.exists():
    model_path = tuned_path
    model_name = "lightgbm_tuned"
    version    = "1.0-tuned"
else:
    model_path = base_path
    model_name = "lightgbm_model"
    version    = "1.0"

print(f"Loading model: {model_path}")
with open(model_path, "rb") as f:
    model = pickle.load(f)
print(f"  Model type: {type(model).__name__}")

# ── Extract feature columns from train_final.parquet ──────────────────────────
print(f"\nLoading feature schema from: {DATA_PATH}")
df = pd.read_parquet(DATA_PATH)
feature_columns = [c for c in df.columns if c not in ("TARGET", "SK_ID_CURR")]
print(f"  Feature columns: {len(feature_columns)}")
print(f"  Sample: {feature_columns[:5]} ... {feature_columns[-5:]}")

# ── Build pipeline dict ────────────────────────────────────────────────────────
THRESHOLD = 0.48

pipeline = {
    "model":           model,
    "feature_columns": feature_columns,
    "threshold":       THRESHOLD,
    "version":         version,
}

# ── Save ───────────────────────────────────────────────────────────────────────
joblib.dump(pipeline, OUTPUT_PATH)
print(f"\nScoring pipeline saved: {OUTPUT_PATH}")
print(f"  Model source : {model_name}")
print(f"  Features     : {len(feature_columns)}")
print(f"  Threshold    : {THRESHOLD}")
print(f"  Version      : {version}")
print("\nDone.")
