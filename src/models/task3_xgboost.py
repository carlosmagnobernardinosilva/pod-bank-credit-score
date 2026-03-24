"""
TAREFA 3 — XGBoost Model
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
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb

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

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()
print(f"  Categorical: {len(cat_cols)}, Numeric: {len(num_cols)}")

# ── Encode categoricals ────────────────────────────────────────────────────────
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_cat_enc = enc.fit_transform(X[cat_cols])
X_enc = np.hstack([X_cat_enc, X[num_cols].values])
feature_names = cat_cols + num_cols

print(f"  Encoded shape: {X_enc.shape}")

# ── XGBoost params (XGBoost >= 2.0 compatible) ────────────────────────────────
# early_stopping_rounds goes in the constructor, not in params dict for XGB >= 2.0
base_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "scale_pos_weight": 11.4,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
    "early_stopping_rounds": 50,
}

# ── Cross-validation ───────────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

auc_list, ks_list, gini_list = [], [], []
ks_train_list = []
best_iters = []
oof_preds = np.zeros(len(y))
fold_importances = []

print("\nRunning 5-fold cross-validation...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_enc, y), start=1):
    X_train, X_val = X_enc[train_idx], X_enc[val_idx]
    y_train, y_val = y.iloc[train_idx].values, y.iloc[val_idx].values

    model = xgb.XGBClassifier(**base_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
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

    best_iter = model.best_iteration
    best_iters.append(best_iter)

    # Feature importance by gain
    fi_scores = model.get_booster().get_score(importance_type="gain")
    # Map f0, f1, ... back to feature names
    fi_named = {}
    for k, v in fi_scores.items():
        # XGBoost uses feature names if provided, else f0, f1, ...
        try:
            idx = int(k[1:])  # f0 -> 0
            fi_named[feature_names[idx]] = v
        except (ValueError, IndexError):
            fi_named[k] = v

    fi = pd.DataFrame({
        "feature": list(fi_named.keys()),
        "importance": list(fi_named.values()),
    })
    fold_importances.append(fi)

    print(f"  Fold {fold}: KS_train={ks_train:.4f}, KS_val={ks:.4f}, gap={ks_train - ks:.4f} | AUC={auc:.4f}, Gini={gini:.4f}, best_iter={best_iter}")

print(f"\nCV Results:")
print(f"  AUC-ROC:  {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")
print(f"  KS val:   {np.mean(ks_list):.4f} ± {np.std(ks_list):.4f}")
print(f"  KS train: {np.mean(ks_train_list):.4f} ± {np.std(ks_train_list):.4f}")
print(f"  KS gap:   {np.mean(ks_train_list) - np.mean(ks_list):.4f} (train - val)")
print(f"  Gini:     {np.mean(gini_list):.4f} ± {np.std(gini_list):.4f}")

best_n = int(np.mean(best_iters))
print(f"  Best n_estimators (mean of folds): {best_n}")

# ── Final model (no early stopping) ───────────────────────────────────────────
print(f"\nTraining final model with n_estimators={best_n}...")
final_params = {k: v for k, v in base_params.items() if k != "early_stopping_rounds"}
final_params["n_estimators"] = best_n

final_model = xgb.XGBClassifier(**final_params, feature_names=feature_names)
final_model.fit(X_enc, y.values)

# ── Feature importance from final model ───────────────────────────────────────
fi_scores_final = final_model.get_booster().get_score(importance_type="gain")
fi_named_final = {}
for k, v in fi_scores_final.items():
    # feature names were set in the constructor
    fi_named_final[k] = v

fi_df = (
    pd.DataFrame({"feature": list(fi_named_final.keys()), "importance": list(fi_named_final.values())})
    .sort_values("importance", ascending=False)
    .head(30)
    .reset_index(drop=True)
)

print("\nTop 30 features by gain (final model):")
print(fi_df.head(10).to_string(index=False))

# ── Plot feature importance ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 12))
fi_sorted = fi_df.sort_values("importance")
ax.barh(fi_sorted["feature"], fi_sorted["importance"], color="darkorange")
ax.set_xlabel("Gain")
ax.set_title("XGBoost — Top 30 Feature Importance (Gain)")
plt.tight_layout()
fig_path = FIGURES_DIR / "xgboost_feature_importance.png"
fig.savefig(fig_path, dpi=150)
plt.close()
print(f"\nFeature importance plot saved to: {fig_path}")

# ── Save model ─────────────────────────────────────────────────────────────────
model_path = MODEL_DIR / "xgboost_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(final_model, f)
print(f"Model saved to: {model_path}")

# ── MLflow logging ─────────────────────────────────────────────────────────────
from src.models.mlflow_setup import setup_mlflow, log_cv_results, check_targets

setup_mlflow("pod-bank-credit-score")
log_params = {k: v for k, v in final_params.items()}
log_params["best_n_estimators"] = best_n

run_id = log_cv_results(
    run_name=f"xgboost_v1_{datetime.now().strftime('%Y%m%d_%H%M')}",
    model=final_model,
    params=log_params,
    cv_metrics={"auc_roc": auc_list, "ks": ks_list, "gini": gini_list},
    feature_importance=fi_df,
)
result = check_targets(
    {"auc_roc": np.mean(auc_list), "ks": np.mean(ks_list), "gini": np.mean(gini_list)}
)
print(f"\nXGBoost status: {result['status']}")
print(f"MLflow run_id: {run_id}")
print(f"Details: {result['details']}")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n=== XGBOOST SUMMARY ===")
print(f"AUC-ROC: {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")
print(f"KS:      {np.mean(ks_list):.4f} ± {np.std(ks_list):.4f}")
print(f"Gini:    {np.mean(gini_list):.4f} ± {np.std(gini_list):.4f}")
print(f"Status:  {result['status']}")
print(f"Run ID:  {run_id}")
