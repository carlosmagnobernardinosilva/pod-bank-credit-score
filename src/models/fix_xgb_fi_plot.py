"""Fix XGBoost feature importance plot with proper feature names."""
from __future__ import annotations
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path("C:/Users/magno/OneDrive/Desktop/pod-bank-credit-score")
sys.path.insert(0, str(PROJECT_ROOT))

df = pd.read_parquet(PROJECT_ROOT / "data/processed/train_final.parquet")
X = df.drop(columns=["SK_ID_CURR", "TARGET"])
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()
feature_names = cat_cols + num_cols

with open(PROJECT_ROOT / "models/xgboost_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

# Use feature_importances_ attribute which aligns with input features
fi_scores = xgb_model.feature_importances_
fi_df = (
    pd.DataFrame({"feature": feature_names, "importance": fi_scores})
    .sort_values("importance", ascending=False)
    .head(30)
    .reset_index(drop=True)
)

print("Top 30 XGBoost features by importance:")
print(fi_df.head(15).to_string(index=False))

# Plot
fig, ax = plt.subplots(figsize=(10, 12))
fi_sorted = fi_df.sort_values("importance")
ax.barh(fi_sorted["feature"], fi_sorted["importance"], color="darkorange")
ax.set_xlabel("Feature Importance (weight)")
ax.set_title("XGBoost — Top 30 Feature Importance")
plt.tight_layout()
fig_path = PROJECT_ROOT / "reports/figures/xgboost_feature_importance.png"
fig.savefig(fig_path, dpi=150)
plt.close()
print(f"\nUpdated plot saved: {fig_path}")

# Update MLflow artifact
from src.models.mlflow_setup import setup_mlflow
import mlflow

setup_mlflow("pod-bank-credit-score")

# Log updated feature importance to existing run
from datetime import datetime
with mlflow.start_run(run_name=f"xgboost_fi_fix_{datetime.now().strftime('%Y%m%d_%H%M')}") as run:
    tmp = PROJECT_ROOT / "tmp_fi.csv"
    fi_df.to_csv(tmp, index=False)
    mlflow.log_artifact(str(tmp), artifact_path="feature_importance")
    mlflow.log_artifact(str(fig_path), artifact_path="figures")
    tmp.unlink(missing_ok=True)
    print(f"Logged to MLflow run: {run.info.run_id}")

print("Done.")
