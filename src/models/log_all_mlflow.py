"""
Log all three trained models to MLflow.
Assumes models are already saved in models/ directory.
Uses precomputed CV metrics from earlier runs.
"""
from __future__ import annotations

import sys
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("C:/Users/magno/OneDrive/Desktop/pod-bank-credit-score")
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.mlflow_setup import setup_mlflow, log_cv_results, check_targets

setup_mlflow("pod-bank-credit-score")

# ── Task 1: Baseline Logistic Regression ──────────────────────────────────────
print("Logging Baseline LR to MLflow...")
with open(PROJECT_ROOT / "models/baseline_logistic_regression.pkl", "rb") as f:
    lr_model = pickle.load(f)

# Metrics from the CV run (captured from output)
lr_auc = [0.7594, 0.7560, 0.7559, 0.7570, 0.7510]
lr_ks  = [0.3931, 0.3852, 0.3903, 0.3796, 0.3800]
lr_gini= [0.5187, 0.5120, 0.5119, 0.5141, 0.5019]

# Feature importance from loaded model
df = pd.read_parquet(PROJECT_ROOT / "data/processed/train_final.parquet")
X = df.drop(columns=["SK_ID_CURR", "TARGET"])
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()
feature_names = cat_cols + num_cols
coefs = np.abs(lr_model.named_steps["model"].coef_[0])
lr_fi = (
    pd.DataFrame({"feature": feature_names, "importance": coefs})
    .sort_values("importance", ascending=False)
    .head(20)
    .reset_index(drop=True)
)

run_id_lr = log_cv_results(
    run_name=f"baseline_lr_{datetime.now().strftime('%Y%m%d_%H%M')}",
    model=lr_model,
    params=lr_model.named_steps["model"].get_params(),
    cv_metrics={"auc_roc": lr_auc, "ks": lr_ks, "gini": lr_gini},
    feature_importance=lr_fi,
)
result_lr = check_targets({"auc_roc": np.mean(lr_auc), "ks": np.mean(lr_ks), "gini": np.mean(lr_gini)})
print(f"  LR status: {result_lr['status']}, run_id: {run_id_lr}")

# ── Task 2: LightGBM ──────────────────────────────────────────────────────────
print("Logging LightGBM to MLflow...")
with open(PROJECT_ROOT / "models/lightgbm_model.pkl", "rb") as f:
    lgb_model = pickle.load(f)

lgb_auc  = [0.7733, 0.7722, 0.7702, 0.7703, 0.7643]
lgb_ks   = [0.4144, 0.4132, 0.4114, 0.4085, 0.4048]
lgb_gini = [0.5467, 0.5444, 0.5403, 0.5406, 0.5286]

# Feature importance from model
lgb_fi_raw = lgb_model.booster_.feature_importance(importance_type="gain")
lgb_fi = (
    pd.DataFrame({"feature": lgb_model.booster_.feature_name(), "importance": lgb_fi_raw})
    .sort_values("importance", ascending=False)
    .head(30)
    .reset_index(drop=True)
)

lgb_params = lgb_model.get_params()
run_id_lgb = log_cv_results(
    run_name=f"lightgbm_v1_{datetime.now().strftime('%Y%m%d_%H%M')}",
    model=lgb_model,
    params=lgb_params,
    cv_metrics={"auc_roc": lgb_auc, "ks": lgb_ks, "gini": lgb_gini},
    feature_importance=lgb_fi,
)
result_lgb = check_targets({"auc_roc": np.mean(lgb_auc), "ks": np.mean(lgb_ks), "gini": np.mean(lgb_gini)})
print(f"  LightGBM status: {result_lgb['status']}, run_id: {run_id_lgb}")

# ── Task 3: XGBoost (will log after it finishes) ──────────────────────────────
xgb_model_path = PROJECT_ROOT / "models/xgboost_model.pkl"
if xgb_model_path.exists():
    print("Logging XGBoost to MLflow...")
    with open(xgb_model_path, "rb") as f:
        xgb_model = pickle.load(f)

    # These will be filled after XGBoost finishes; placeholder for now
    # Will be updated by task3_xgboost.py directly
    print("  XGBoost model found, will be logged by task3_xgboost.py")
else:
    print("  XGBoost model not yet available (still training).")

print("\n=== MLflow Logging Complete ===")
print(f"Baseline LR run_id:  {run_id_lr}")
print(f"LightGBM  run_id:    {run_id_lgb}")
