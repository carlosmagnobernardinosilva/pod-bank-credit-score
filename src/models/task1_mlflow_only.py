"""Re-run just the MLflow logging for baseline LR (model already trained)."""
from __future__ import annotations

import sys
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve

PROJECT_ROOT = Path("C:/Users/magno/OneDrive/Desktop/pod-bank-credit-score")
sys.path.insert(0, str(PROJECT_ROOT))

# Load already-trained model
MODEL_PATH = PROJECT_ROOT / "models/baseline_logistic_regression.pkl"
with open(MODEL_PATH, "rb") as f:
    final_pipeline = pickle.load(f)

# Reload data and recompute CV metrics (fast — just scoring)
df = pd.read_parquet(PROJECT_ROOT / "data/processed/train_final.parquet")
y = df["TARGET"].astype(int)
X = df.drop(columns=["SK_ID_CURR", "TARGET"])

# Quick CV just to get the metrics (the model IS the pipeline already fitted,
# but we need fold metrics for logging — re-run CV)
from sklearn.base import clone
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ("num", StandardScaler(), num_cols),
    ]
)
lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42, solver="saga")
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", lr)])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_list, ks_list, gini_list = [], [], []

print("Re-running CV to get fold metrics for MLflow...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    p = clone(pipeline)
    p.fit(X_train, y_train)
    y_proba = p.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_proba)
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    ks = float(np.max(tpr - fpr))
    gini = 2 * auc - 1
    auc_list.append(auc)
    ks_list.append(ks)
    gini_list.append(gini)
    print(f"  Fold {fold}: AUC={auc:.4f} KS={ks:.4f} Gini={gini:.4f}")

print(f"AUC-ROC: {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")
print(f"KS:      {np.mean(ks_list):.4f} ± {np.std(ks_list):.4f}")
print(f"Gini:    {np.mean(gini_list):.4f} ± {np.std(gini_list):.4f}")

# Feature importance from loaded model
feature_names = cat_cols + num_cols
coefs = np.abs(final_pipeline.named_steps["model"].coef_[0])
fi_df = (
    pd.DataFrame({"feature": feature_names, "importance": coefs})
    .sort_values("importance", ascending=False)
    .head(20)
    .reset_index(drop=True)
)

from src.models.mlflow_setup import setup_mlflow, log_cv_results, check_targets

setup_mlflow("pod-bank-credit-score")
run_id = log_cv_results(
    run_name=f"baseline_lr_{datetime.now().strftime('%Y%m%d_%H%M')}",
    model=final_pipeline,
    params=final_pipeline.named_steps["model"].get_params(),
    cv_metrics={"auc_roc": auc_list, "ks": ks_list, "gini": gini_list},
    feature_importance=fi_df,
)
result = check_targets(
    {"auc_roc": np.mean(auc_list), "ks": np.mean(ks_list), "gini": np.mean(gini_list)}
)
print(f"\nBaseline status: {result['status']}")
print(f"MLflow run_id: {run_id}")

print("\n=== BASELINE LR SUMMARY ===")
print(f"AUC-ROC: {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")
print(f"KS:      {np.mean(ks_list):.4f} ± {np.std(ks_list):.4f}")
print(f"Gini:    {np.mean(gini_list):.4f} ± {np.std(gini_list):.4f}")
print(f"Status:  {result['status']}")
print(f"Run ID:  {run_id}")
