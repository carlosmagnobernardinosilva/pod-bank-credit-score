"""
TAREFA 1 — Baseline Logistic Regression
CRISP-DM Phase 4 — Modeling
"""
from __future__ import annotations

import sys
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("C:/Users/magno/OneDrive/Desktop/pod-bank-credit-score")
sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data/processed/train_final.parquet"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_parquet(DATA_PATH)
print(f"  Shape: {df.shape}")

y = df["TARGET"].astype(int)
X = df.drop(columns=["SK_ID_CURR", "TARGET"])

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()
print(f"  Categorical: {len(cat_cols)}, Numeric: {len(num_cols)}")

# ── Preprocessor ──────────────────────────────────────────────────────────────
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ("num", StandardScaler(), num_cols),
    ]
)

lr = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42,
    solver="saga",
    n_jobs=-1,
)

pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", lr)])

# ── Cross-validation ───────────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

auc_list, ks_list, gini_list = [], [], []
ks_train_list = []

print("\nRunning 5-fold cross-validation...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    pipeline.fit(X_train, y_train)

    # Validation metrics
    y_proba = pipeline.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_proba)
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    ks = float(np.max(tpr - fpr))
    gini = 2 * auc - 1

    # Train KS (overfitting diagnostic)
    y_proba_train = pipeline.predict_proba(X_train)[:, 1]
    fpr_tr, tpr_tr, _ = roc_curve(y_train, y_proba_train)
    ks_train = float(np.max(tpr_tr - fpr_tr))

    auc_list.append(auc)
    ks_list.append(ks)
    gini_list.append(gini)
    ks_train_list.append(ks_train)

    print(f"  Fold {fold}: KS_train={ks_train:.4f}, KS_val={ks:.4f}, gap={ks_train - ks:.4f} | AUC={auc:.4f}, Gini={gini:.4f}")

print(f"\nCV Results:")
print(f"  AUC-ROC:  {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")
print(f"  KS val:   {np.mean(ks_list):.4f} ± {np.std(ks_list):.4f}")
print(f"  KS train: {np.mean(ks_train_list):.4f} ± {np.std(ks_train_list):.4f}")
print(f"  KS gap:   {np.mean(ks_train_list) - np.mean(ks_list):.4f} (train - val)")
print(f"  Gini:     {np.mean(gini_list):.4f} ± {np.std(gini_list):.4f}")

# ── Final model trained on all data ───────────────────────────────────────────
print("\nTraining final model on all data...")
final_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", lr)])

# Re-clone components to avoid state issues
from sklearn.base import clone
preprocessor_final = clone(preprocessor)
lr_final = clone(lr)
final_pipeline = Pipeline(steps=[("preprocessor", preprocessor_final), ("model", lr_final)])
final_pipeline.fit(X, y)

# ── Feature importance (absolute coefficients) ────────────────────────────────
feature_names = cat_cols + num_cols
coefs = np.abs(final_pipeline.named_steps["model"].coef_[0])
fi_df = (
    pd.DataFrame({"feature": feature_names, "importance": coefs})
    .sort_values("importance", ascending=False)
    .head(20)
    .reset_index(drop=True)
)
print("\nTop 20 features by |coefficient|:")
print(fi_df.to_string(index=False))

# ── Save model ────────────────────────────────────────────────────────────────
model_path = MODEL_DIR / "baseline_logistic_regression.pkl"
with open(model_path, "wb") as f:
    pickle.dump(final_pipeline, f)
print(f"\nModel saved to: {model_path}")

# ── MLflow logging ─────────────────────────────────────────────────────────────
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
print(f"Details: {result['details']}")

# ── Summary for orchestrator ───────────────────────────────────────────────────
print("\n=== BASELINE LR SUMMARY ===")
print(f"AUC-ROC: {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")
print(f"KS:      {np.mean(ks_list):.4f} ± {np.std(ks_list):.4f}")
print(f"Gini:    {np.mean(gini_list):.4f} ± {np.std(gini_list):.4f}")
print(f"Status:  {result['status']}")
print(f"Run ID:  {run_id}")
