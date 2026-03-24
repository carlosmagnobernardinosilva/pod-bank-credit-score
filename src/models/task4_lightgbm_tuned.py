"""
TAREFA 4 — LightGBM Tuned (v2)
CRISP-DM Phase 5 — Evaluation / Regularization

Objetivo: reduzir overfitting do v1 via maior regularização.
Comparação direta com lightgbm_v1 pelo gap KS_train - KS_val.

Mudanças em relação ao v1:
  - num_leaves:          31  → 20   (árvores menos complexas)
  - min_child_samples:   20  → 80   (mais amostras por folha)
  - reg_alpha:           0.1 → 1.0  (L1 mais forte)
  - reg_lambda:          0.1 → 1.0  (L2 mais forte)
  - feature_fraction:    0.8 → 0.65
  - bagging_fraction:    0.8 → 0.65
  - min_split_gain:      0   → 0.01 (novo)
  - learning_rate:       0.05 → 0.03 (mais conservador)
  - early_stopping:      50  → 80   (mais paciência)
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

DATA_PATH   = PROJECT_ROOT / "data/processed/train_final.parquet"
MODEL_DIR   = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports/figures"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_parquet(DATA_PATH)
print(f"  Shape: {df.shape}")

y = df["TARGET"].astype(int)
X = df.drop(columns=["SK_ID_CURR", "TARGET"])

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
for col in cat_cols:
    X[col] = X[col].astype("category")

print(f"  Categorical cols: {len(cat_cols)}")

# ── v1 reference params (for comparison printout) ──────────────────────────────
V1_PARAMS = {
    "num_leaves": 31, "min_child_samples": 20,
    "reg_alpha": 0.1, "reg_lambda": 0.1,
    "feature_fraction": 0.8, "bagging_fraction": 0.8,
    "min_split_gain": 0.0, "learning_rate": 0.05,
}

# ── v2 tuned params ─────────────────────────────────────────────────────────────
params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "n_estimators": 1500,
    "learning_rate": 0.03,
    "num_leaves": 20,
    "max_depth": -1,
    "min_child_samples": 80,
    "min_split_gain": 0.01,
    "feature_fraction": 0.65,
    "bagging_fraction": 0.65,
    "bagging_freq": 5,
    "reg_alpha": 1.0,
    "reg_lambda": 1.0,
    "scale_pos_weight": 11.4,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

print("\nRegularization changes v1 -> v2:")
for k in V1_PARAMS:
    old = V1_PARAMS[k]
    new = params.get(k, "N/A")
    marker = " *" if old != new else ""
    print(f"  {k:<22}: {str(old):<8} -> {new}{marker}")

# ── Cross-validation ────────────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

auc_list, ks_list, gini_list = [], [], []
ks_train_list = []
best_iters = []
oof_preds = np.zeros(len(y))
fold_importances = []

callbacks = [
    lgb.early_stopping(stopping_rounds=80, verbose=False),
    lgb.log_evaluation(period=0),
]

print("\nRunning 5-fold cross-validation (v2 tuned)...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks,
    )

    y_proba = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = y_proba
    auc = roc_auc_score(y_val, y_proba)
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    ks = float(np.max(tpr - fpr))
    gini = 2 * auc - 1

    y_proba_train = model.predict_proba(X_train)[:, 1]
    fpr_tr, tpr_tr, _ = roc_curve(y_train, y_proba_train)
    ks_train = float(np.max(tpr_tr - fpr_tr))

    auc_list.append(auc)
    ks_list.append(ks)
    gini_list.append(gini)
    ks_train_list.append(ks_train)
    best_iters.append(model.best_iteration_)

    fi = pd.DataFrame({
        "feature": X.columns.tolist(),
        "importance": model.booster_.feature_importance(importance_type="gain"),
    })
    fold_importances.append(fi)

    gap = ks_train - ks
    print(f"  Fold {fold}: KS_train={ks_train:.4f}, KS_val={ks:.4f}, gap={gap:.4f} | AUC={auc:.4f}, Gini={gini:.4f}, best_iter={model.best_iteration_}")

# ── Results ─────────────────────────────────────────────────────────────────────
mean_auc   = np.mean(auc_list)
mean_ks    = np.mean(ks_list)
mean_gini  = np.mean(gini_list)
mean_gap   = np.mean(ks_train_list) - mean_ks

print(f"\nCV Results (v2 tuned):")
print(f"  AUC-ROC:  {mean_auc:.4f} ± {np.std(auc_list):.4f}")
print(f"  KS val:   {mean_ks:.4f} ± {np.std(ks_list):.4f}")
print(f"  KS train: {np.mean(ks_train_list):.4f} ± {np.std(ks_train_list):.4f}")
print(f"  KS gap:   {mean_gap:.4f} (train - val)  [v1 reference: ~0.10+]")
print(f"  Gini:     {mean_gini:.4f} ± {np.std(gini_list):.4f}")

best_n = int(np.mean(best_iters))
print(f"  Best n_estimators (mean): {best_n}")

# ── Final model (full data) ─────────────────────────────────────────────────────
print(f"\nTraining final model on full data, n_estimators={best_n}...")
final_params = {**params, "n_estimators": best_n}
final_model = lgb.LGBMClassifier(**final_params)
final_model.fit(X, y)

# ── Feature importance ──────────────────────────────────────────────────────────
mean_fi = (
    pd.concat(fold_importances)
    .groupby("feature")["importance"]
    .mean()
    .reset_index()
    .sort_values("importance", ascending=False)
    .head(30)
    .reset_index(drop=True)
)
print("\nTop 10 features (v2 tuned):")
print(mean_fi.head(10).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 12))
sorted_fi = mean_fi.sort_values("importance")
ax.barh(sorted_fi["feature"], sorted_fi["importance"], color="darkorange")
ax.set_xlabel("Gain (mean across folds)")
ax.set_title("LightGBM v2 Tuned — Top 30 Feature Importance (Gain)")
plt.tight_layout()
fig_path = FIGURES_DIR / "lightgbm_tuned_feature_importance.png"
fig.savefig(fig_path, dpi=150)
plt.close()
print(f"Feature importance saved: {fig_path}")

# ── OOF score distribution ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(oof_preds[y == 0], bins=60, alpha=0.6, color="steelblue", label="Adimplente (0)", density=True)
ax.hist(oof_preds[y == 1], bins=60, alpha=0.6, color="tomato",    label="Inadimplente (1)", density=True)
ax.set_xlabel("Score (probabilidade de inadimplência)")
ax.set_ylabel("Densidade")
ax.set_title("LightGBM v2 — Distribuição OOF por Classe")
ax.legend()
plt.tight_layout()
oof_fig_path = FIGURES_DIR / "lightgbm_tuned_oof_distribution.png"
fig.savefig(oof_fig_path, dpi=150)
plt.close()
print(f"OOF distribution saved: {oof_fig_path}")

# ── Save model ──────────────────────────────────────────────────────────────────
model_path = MODEL_DIR / "lightgbm_tuned.pkl"
with open(model_path, "wb") as f:
    pickle.dump(final_model, f)
print(f"Model saved: {model_path}")

# ── MLflow logging ───────────────────────────────────────────────────────────────
from src.models.mlflow_setup import setup_mlflow, log_cv_results, check_targets

setup_mlflow("pod-bank-credit-score")
run_id = log_cv_results(
    run_name=f"lightgbm_v2_tuned_{datetime.now().strftime('%Y%m%d_%H%M')}",
    model=final_model,
    params={**final_params, "best_n_estimators": best_n, "version": "v2_tuned"},
    cv_metrics={"auc_roc": auc_list, "ks": ks_list, "gini": gini_list},
    feature_importance=mean_fi,
)
result = check_targets({"auc_roc": mean_auc, "ks": mean_ks, "gini": mean_gini})

print(f"\nLightGBM v2 status: {result['status']}")
print(f"MLflow run_id: {run_id}")

# ── Summary ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("LIGHTGBM v2 TUNED — SUMMARY")
print("=" * 50)
print(f"AUC-ROC : {mean_auc:.4f} ± {np.std(auc_list):.4f}")
print(f"KS      : {mean_ks:.4f} ± {np.std(ks_list):.4f}")
print(f"Gini    : {mean_gini:.4f} ± {np.std(gini_list):.4f}")
print(f"KS gap  : {mean_gap:.4f}")
print(f"Status  : {result['status']}")
print(f"Run ID  : {run_id}")
print(f"Model   : {model_path}")
