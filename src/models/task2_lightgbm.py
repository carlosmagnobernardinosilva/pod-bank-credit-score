"""
TAREFA 2 — LightGBM Model
CRISP-DM Phase 4 — Modeling
"""
from __future__ import annotations

import sys
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
import lightgbm as lgb

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("C:/Users/magno/OneDrive/Desktop/pod-bank-credit-score")
sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data/processed/train_final.parquet"
MODEL_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports/figures"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_parquet(DATA_PATH)
print(f"  Shape: {df.shape}")

y = df["TARGET"].astype(int)
X = df.drop(columns=["SK_ID_CURR", "TARGET"])

# Convert categoricals to dtype "category" for LightGBM native handling
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
for col in cat_cols:
    X[col] = X[col].astype("category")

print(f"  Categorical cols: {cat_cols}")

# ── LightGBM params ────────────────────────────────────────────────────────────
params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "scale_pos_weight": 11.4,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

# ── Cross-validation ───────────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

auc_list, ks_list, gini_list = [], [], []
ks_train_list = []
best_iters = []
oof_preds = np.zeros(len(y))
fold_importances = []

callbacks = [
    lgb.early_stopping(stopping_rounds=50, verbose=False),
    lgb.log_evaluation(period=0),
]

print("\nRunning 5-fold cross-validation...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks,
    )

    # Validation metrics
    y_proba = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = y_proba
    auc = roc_auc_score(y_val, y_proba)
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    ks = float(np.max(tpr - fpr))
    gini = 2 * auc - 1

    # Train KS (overfitting diagnostic)
    y_proba_train = model.predict_proba(X_train)[:, 1]
    fpr_tr, tpr_tr, _ = roc_curve(y_train, y_proba_train)
    ks_train = float(np.max(tpr_tr - fpr_tr))

    auc_list.append(auc)
    ks_list.append(ks)
    gini_list.append(gini)
    ks_train_list.append(ks_train)
    best_iters.append(model.best_iteration_)

    # Feature importances (gain)
    fi = pd.DataFrame({
        "feature": X.columns.tolist(),
        "importance": model.booster_.feature_importance(importance_type="gain"),
    })
    fold_importances.append(fi)

    print(f"  Fold {fold}: KS_train={ks_train:.4f}, KS_val={ks:.4f}, gap={ks_train - ks:.4f} | AUC={auc:.4f}, Gini={gini:.4f}, best_iter={model.best_iteration_}")

print(f"\nCV Results:")
print(f"  AUC-ROC:  {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")
print(f"  KS val:   {np.mean(ks_list):.4f} ± {np.std(ks_list):.4f}")
print(f"  KS train: {np.mean(ks_train_list):.4f} ± {np.std(ks_train_list):.4f}")
print(f"  KS gap:   {np.mean(ks_train_list) - np.mean(ks_list):.4f} (train - val)")
print(f"  Gini:     {np.mean(gini_list):.4f} ± {np.std(gini_list):.4f}")

best_n = int(np.mean(best_iters))
print(f"  Best n_estimators (mean of folds): {best_n}")

# ── Final model ────────────────────────────────────────────────────────────────
print(f"\nTraining final model with n_estimators={best_n}...")
final_params = {k: v for k, v in params.items()}
final_params["n_estimators"] = best_n

final_model = lgb.LGBMClassifier(**final_params)
final_model.fit(X, y)

# ── Feature importance ─────────────────────────────────────────────────────────
# Average across folds
mean_fi = (
    pd.concat(fold_importances)
    .groupby("feature")["importance"]
    .mean()
    .reset_index()
    .sort_values("importance", ascending=False)
    .head(30)
    .reset_index(drop=True)
)
print("\nTop 30 features by gain (mean across folds):")
print(mean_fi.head(10).to_string(index=False))

# ── Plot feature importance ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 12))
mean_fi_sorted = mean_fi.sort_values("importance")
ax.barh(mean_fi_sorted["feature"], mean_fi_sorted["importance"], color="steelblue")
ax.set_xlabel("Gain (mean across folds)")
ax.set_title("LightGBM — Top 30 Feature Importance (Gain)")
plt.tight_layout()
fig_path = FIGURES_DIR / "lightgbm_feature_importance.png"
fig.savefig(fig_path, dpi=150)
plt.close()
print(f"\nFeature importance plot saved to: {fig_path}")

# ── Save model ─────────────────────────────────────────────────────────────────
model_path = MODEL_DIR / "lightgbm_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(final_model, f)
print(f"Model saved to: {model_path}")

# ── MLflow logging ─────────────────────────────────────────────────────────────
from src.models.mlflow_setup import setup_mlflow, log_cv_results, check_targets

setup_mlflow("pod-bank-credit-score")
run_id = log_cv_results(
    run_name=f"lightgbm_v1_{datetime.now().strftime('%Y%m%d_%H%M')}",
    model=final_model,
    params={**final_params, "best_n_estimators": best_n},
    cv_metrics={"auc_roc": auc_list, "ks": ks_list, "gini": gini_list},
    feature_importance=mean_fi,
)
result = check_targets(
    {"auc_roc": np.mean(auc_list), "ks": np.mean(ks_list), "gini": np.mean(gini_list)}
)
print(f"\nLightGBM status: {result['status']}")
print(f"MLflow run_id: {run_id}")
print(f"Details: {result['details']}")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n=== LIGHTGBM SUMMARY ===")
print(f"AUC-ROC: {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")
print(f"KS:      {np.mean(ks_list):.4f} ± {np.std(ks_list):.4f}")
print(f"Gini:    {np.mean(gini_list):.4f} ± {np.std(gini_list):.4f}")
print(f"Status:  {result['status']}")
print(f"Run ID:  {run_id}")
