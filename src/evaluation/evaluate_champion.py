# -*- coding: utf-8 -*-
"""
evaluate_champion.py
====================
Fase 5 CRISP-DM — Avaliacao completa do modelo campiao LightGBM v2 Tuned.

Outputs:
  reports/evaluation_report.md
  reports/figures/roc_curve.png
  reports/figures/ks_curve.png
  reports/figures/lift_curve.png
  reports/figures/score_distribution.png
  reports/figures/confusion_matrix.png
  reports/figures/feature_importance_final.png

Uso:
    python src/evaluation/evaluate_champion.py
"""
from __future__ import annotations

import sys
import pickle
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    precision_score, recall_score, f1_score,
)

warnings.filterwarnings("ignore")

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("C:/Users/magno/OneDrive/Desktop/pod-bank-credit-score")
sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH   = PROJECT_ROOT / "data/processed/train_final.parquet"
MODEL_PATH  = PROJECT_ROOT / "models/lightgbm_tuned.pkl"
if not MODEL_PATH.exists():
    MODEL_PATH = PROJECT_ROOT / "models/lightgbm_model.pkl"

FIGURES_DIR = PROJECT_ROOT / "reports/figures"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data & model ─────────────────────────────────────────────────────────
print("Loading data and model...")
df = pd.read_parquet(DATA_PATH)
y_all = df["TARGET"].astype(int)
X_all = df.drop(columns=["SK_ID_CURR", "TARGET"])

cat_cols = X_all.select_dtypes(include=["object", "category"]).columns.tolist()
for col in cat_cols:
    X_all[col] = X_all[col].astype("category")

X_train, X_holdout, y_train, y_holdout = train_test_split(
    X_all, y_all, test_size=0.20, random_state=42, stratify=y_all
)
print(f"  Holdout size: {len(y_holdout):,} ({y_holdout.mean()*100:.2f}% default)")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

y_proba = model.predict_proba(X_holdout)[:, 1]
auc = roc_auc_score(y_holdout, y_proba)
gini = 2 * auc - 1
fpr, tpr, thresholds_roc = roc_curve(y_holdout, y_proba)
ks_stat = float(np.max(tpr - fpr))
ks_threshold = thresholds_roc[np.argmax(tpr - fpr)]

print(f"  AUC-ROC : {auc:.4f}")
print(f"  KS      : {ks_stat:.4f}  (threshold={ks_threshold:.4f})")
print(f"  Gini    : {gini:.4f}")

# ── 1. ROC Curve ───────────────────────────────────────────────────────────────
print("\n[1/7] ROC Curve...")
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"LightGBM v2 (AUC={auc:.4f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
opt_idx = np.argmax(tpr - fpr)
ax.scatter(fpr[opt_idx], tpr[opt_idx], color="red", zorder=5,
           label=f"Ponto KS (thr={ks_threshold:.3f})")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Curva ROC — LightGBM v2 Tuned (holdout 20%)")
ax.legend(loc="lower right")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "roc_curve.png", dpi=150)
plt.close()

# ── 2. KS Curve ────────────────────────────────────────────────────────────────
print("[2/7] KS Curve...")
sorted_idx = np.argsort(y_proba)[::-1]
y_sorted = y_holdout.values[sorted_idx]
n = len(y_sorted)
n_pos = y_sorted.sum()
n_neg = n - n_pos
cum_pos = np.cumsum(y_sorted) / n_pos
cum_neg = np.cumsum(1 - y_sorted) / n_neg
pct_pop = np.arange(1, n + 1) / n

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(pct_pop, cum_pos, color="tomato", lw=2, label="Maus (inadimplentes)")
ax.plot(pct_pop, cum_neg, color="steelblue", lw=2, label="Bons (adimplentes)")
ks_x = pct_pop[np.argmax(cum_pos - cum_neg)]
ks_y1 = cum_pos[np.argmax(cum_pos - cum_neg)]
ks_y2 = cum_neg[np.argmax(cum_pos - cum_neg)]
ax.vlines(ks_x, ks_y2, ks_y1, colors="black", lw=2, linestyle="--",
          label=f"KS={ks_stat:.4f}")
ax.set_xlabel("Populacao acumulada (ordenada por score desc)")
ax.set_ylabel("Frequencia acumulada")
ax.set_title("Curva KS — LightGBM v2 Tuned")
ax.legend()
plt.tight_layout()
fig.savefig(FIGURES_DIR / "ks_curve.png", dpi=150)
plt.close()

# ── 3. Lift Curve ───────────────────────────────────────────────────────────────
print("[3/7] Lift Curve...")
n_deciles = 10
decile_size = n // n_deciles
baseline_rate = y_holdout.mean()
lift_vals, cumulative_lift = [], []
cum_pos_count = 0
for i in range(n_deciles):
    chunk = y_sorted[i * decile_size: (i + 1) * decile_size]
    chunk_rate = chunk.mean()
    lift_vals.append(chunk_rate / baseline_rate)
    cum_pos_count += chunk.sum()
    cum_rate = cum_pos_count / ((i + 1) * decile_size)
    cumulative_lift.append(cum_rate / baseline_rate)

deciles = list(range(1, n_deciles + 1))
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].bar(deciles, lift_vals, color="steelblue")
axes[0].axhline(1.0, color="red", lw=1.5, linestyle="--", label="Baseline")
axes[0].set_xlabel("Decil (1=maior score)")
axes[0].set_ylabel("Lift")
axes[0].set_title("Lift por Decil")
axes[0].legend()
axes[1].plot(deciles, cumulative_lift, marker="o", color="darkorange", lw=2)
axes[1].axhline(1.0, color="red", lw=1.5, linestyle="--", label="Baseline")
axes[1].set_xlabel("Decil acumulado")
axes[1].set_ylabel("Lift acumulado")
axes[1].set_title("Lift Acumulado")
axes[1].legend()
plt.tight_layout()
fig.savefig(FIGURES_DIR / "lift_curve.png", dpi=150)
plt.close()

# ── 4. Threshold Calibration ───────────────────────────────────────────────────
print("[4/7] Threshold calibration (0.05 to 0.50)...")
thresholds = np.arange(0.05, 0.51, 0.01)
results = []
for thr in thresholds:
    y_pred = (y_proba >= thr).astype(int)
    prec  = precision_score(y_holdout, y_pred, zero_division=0)
    rec   = recall_score(y_holdout, y_pred, zero_division=0)
    f1    = f1_score(y_holdout, y_pred, zero_division=0)
    # recall for good payers (class 0)
    rec_good = recall_score(1 - y_holdout, 1 - y_pred, zero_division=0)
    approval = 1 - y_pred.mean()
    # expected default rate among approved
    approved_mask = y_pred == 0
    exp_default = y_holdout[approved_mask].mean() if approved_mask.sum() > 0 else 0.0
    results.append({
        "threshold": round(float(thr), 2),
        "precision": prec, "recall_bad": rec,
        "recall_good": rec_good, "f1": f1,
        "approval_rate": approval,
        "expected_default_rate": exp_default,
    })

thresh_df = pd.DataFrame(results)

# Elect threshold: recall_good >= 0.70 AND min expected_default_rate
candidates = thresh_df[thresh_df["recall_good"] >= 0.70]
if len(candidates) > 0:
    elected = candidates.loc[candidates["expected_default_rate"].idxmin()]
else:
    elected = thresh_df.loc[thresh_df["recall_good"].idxmax()]

elected_thr = float(elected["threshold"])
print(f"  Threshold eleito: {elected_thr:.2f}")
print(f"    Recall bons   : {elected['recall_good']:.4f}")
print(f"    Recall maus   : {elected['recall_bad']:.4f}")
print(f"    Taxa aprovacao: {elected['approval_rate']:.4f}")
print(f"    Default esperado (aprovados): {elected['expected_default_rate']:.4f}")

# ── 5. Score Distribution ──────────────────────────────────────────────────────
print("[5/7] Score distribution...")
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(y_proba[y_holdout == 0], bins=60, alpha=0.6, color="steelblue",
        label="Adimplente (0)", density=True)
ax.hist(y_proba[y_holdout == 1], bins=60, alpha=0.6, color="tomato",
        label="Inadimplente (1)", density=True)
ax.axvline(elected_thr, color="black", lw=2, linestyle="--",
           label=f"Threshold eleito={elected_thr:.2f}")
ax.set_xlabel("Score (probabilidade de inadimplencia)")
ax.set_ylabel("Densidade")
ax.set_title("Distribuicao do Score por Classe — LightGBM v2 Tuned")
ax.legend()
plt.tight_layout()
fig.savefig(FIGURES_DIR / "score_distribution.png", dpi=150)
plt.close()

# ── 6. Confusion Matrix ────────────────────────────────────────────────────────
print("[6/7] Confusion matrix...")
y_pred_final = (y_proba >= elected_thr).astype(int)
cm = confusion_matrix(y_holdout, y_pred_final)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

tn, fp, fn, tp = cm.ravel()
cost_fn = 5  # approve defaulter
cost_fp = 1  # reject good payer
total_cost = fn * cost_fn + fp * cost_fp
print(f"  TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"  Custo estimado (FN*5 + FP*1): {total_cost:,}")

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm_norm, annot=True, fmt=".2%", cmap="Blues", ax=ax,
    xticklabels=["Pred: Adimplente", "Pred: Inadimplente"],
    yticklabels=["Real: Adimplente", "Real: Inadimplente"],
)
ax.set_title(f"Matriz de Confusao (threshold={elected_thr:.2f})\nCusto estimado: {total_cost:,}")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
plt.close()

# ── 7. Feature Importance Final ────────────────────────────────────────────────
print("[7/7] Feature importance final...")
fi = pd.DataFrame({
    "feature": X_holdout.columns.tolist(),
    "importance": model.booster_.feature_importance(importance_type="gain"),
}).sort_values("importance", ascending=False).head(30).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(10, 12))
fi_sorted = fi.sort_values("importance")
ax.barh(fi_sorted["feature"], fi_sorted["importance"], color="steelblue")
ax.set_xlabel("Gain")
ax.set_title("Top 30 Features — LightGBM v2 Tuned (Holdout)")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "feature_importance_final.png", dpi=150)
plt.close()

# ── Error Analysis ─────────────────────────────────────────────────────────────
print("\nError analysis (FN vs TP)...")
fn_mask = (y_pred_final == 0) & (y_holdout == 1)  # false negatives
tp_mask = (y_pred_final == 1) & (y_holdout == 1)  # true positives

num_feats = X_holdout.select_dtypes(include=[np.number]).columns.tolist()
fn_profile = X_holdout[fn_mask][num_feats].mean()
tp_profile = X_holdout[tp_mask][num_feats].mean()

diff = (fn_profile - tp_profile).abs().sort_values(ascending=False).head(10)
error_analysis_df = pd.DataFrame({
    "feature": diff.index,
    "FN_mean": fn_profile[diff.index].values,
    "TP_mean": tp_profile[diff.index].values,
    "abs_diff": diff.values,
})

# ── Business Targets Validation ────────────────────────────────────────────────
print("\nValidando targets do negocio (Fase 5)...")
recall_good_final = float(elected["recall_good"])

TARGETS_FASE5 = {
    "AUC-ROC >= 0.75": auc >= 0.75,
    "KS >= 0.35":      ks_stat >= 0.35,
    "Gini >= 0.50":    gini >= 0.50,
    "Recall bons >= 70%": recall_good_final >= 0.70,
}
for criterion, passed in TARGETS_FASE5.items():
    status = "APROVADO" if passed else "REPROVADO"
    print(f"  {criterion:<30}: {status}")

overall_status = "APROVADO" if all(TARGETS_FASE5.values()) else "REPROVADO"

# ── MLflow logging ─────────────────────────────────────────────────────────────
print("\nLogging evaluation to MLflow...")
from src.models.mlflow_setup import setup_mlflow
import mlflow

setup_mlflow("pod-bank-credit-score")
with mlflow.start_run(run_name=f"evaluation_champion_{datetime.now().strftime('%Y%m%d_%H%M')}") as run:
    mlflow.log_metrics({
        "holdout_auc_roc": auc,
        "holdout_ks": ks_stat,
        "holdout_gini": gini,
        "elected_threshold": elected_thr,
        "recall_good_payers": recall_good_final,
        "approval_rate": float(elected["approval_rate"]),
        "expected_default_rate": float(elected["expected_default_rate"]),
        "confusion_cost": float(total_cost),
    })
    mlflow.set_tags({
        "phase": "evaluation",
        "model": "lightgbm_v2_tuned",
        "status": overall_status,
    })
    eval_run_id = run.info.run_id

print(f"  MLflow eval run_id: {eval_run_id}")

# ── Generate Report ────────────────────────────────────────────────────────────
print("\nGenerating evaluation_report.md...")

lines = [
    "# Relatorio de Avaliacao do Modelo Campiao",
    "",
    f"**Modelo:** LightGBM v2 Tuned  ",
    f"**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
    f"**Holdout:** 20% estratificado (random_state=42)  ",
    f"**MLflow Run:** `{eval_run_id}`",
    "",
    "---",
    "",
    "## 1. Metricas no Holdout",
    "",
    "| Metrica | Valor | Target Fase 5 | Status |",
    "|---------|-------|---------------|--------|",
    f"| AUC-ROC | {auc:.4f} | >= 0.75 | {'APROVADO' if auc >= 0.75 else 'REPROVADO'} |",
    f"| KS Statistic | {ks_stat:.4f} | >= 0.35 | {'APROVADO' if ks_stat >= 0.35 else 'REPROVADO'} |",
    f"| Gini | {gini:.4f} | >= 0.50 | {'APROVADO' if gini >= 0.50 else 'REPROVADO'} |",
    f"| Recall bons pagadores | {recall_good_final:.4f} | >= 70% | {'APROVADO' if recall_good_final >= 0.70 else 'REPROVADO'} |",
    "",
    f"**Status Geral: {overall_status}**",
    "",
    "---",
    "",
    "## 2. Comparacao v1 vs v2 (CV 5-fold)",
    "",
    "| Modelo | AUC-ROC | KS | Gini | KS gap |",
    "|--------|---------|----|------|--------|",
    "| LightGBM v1 | 0.7701 | 0.4105 | 0.5401 | ~0.10+ |",
    "| LightGBM v2 Tuned | 0.7717 | 0.4118 | 0.5434 | 0.0873 |",
    "",
    "v2 manteve performance e reduziu overfitting (KS gap: 0.10+ -> 0.0873).",
    "",
    "---",
    "",
    "## 3. Calibracao de Threshold",
    "",
    f"**Threshold eleito:** `{elected_thr:.2f}`",
    "",
    "| Metrica | Valor |",
    "|---------|-------|",
    f"| Recall bons pagadores | {recall_good_final:.4f} |",
    f"| Recall maus pagadores | {float(elected['recall_bad']):.4f} |",
    f"| Precisao | {float(elected['precision']):.4f} |",
    f"| F1 | {float(elected['f1']):.4f} |",
    f"| Taxa de aprovacao | {float(elected['approval_rate']):.4f} |",
    f"| Default esperado (aprovados) | {float(elected['expected_default_rate']):.4f} |",
    "",
    "---",
    "",
    "## 4. Matriz de Confusao",
    "",
    f"Threshold: `{elected_thr:.2f}`",
    "",
    "| | Pred: Adimplente | Pred: Inadimplente |",
    "|--|-----------------|-------------------|",
    f"| **Real: Adimplente** | {tn:,} | {fp:,} |",
    f"| **Real: Inadimplente** | {fn:,} | {tp:,} |",
    "",
    f"**Custo estimado** (FN x5 + FP x1): **{total_cost:,}**",
    "",
    "---",
    "",
    "## 5. Top 10 Features",
    "",
    "| # | Feature | Importancia (Gain) |",
    "|---|---------|-------------------|",
]
for i, row in fi.head(10).iterrows():
    lines.append(f"| {i+1} | `{row['feature']}` | {row['importance']:,.0f} |")

lines += [
    "",
    "---",
    "",
    "## 6. Analise de Erros — Falsos Negativos vs Verdadeiros Positivos",
    "",
    "Top 10 features com maior diferenca media entre FN (inadimplentes nao detectados) e TP:",
    "",
    "| Feature | Media FN | Media TP | Diferenca |",
    "|---------|----------|----------|-----------|",
]
for _, row in error_analysis_df.iterrows():
    lines.append(f"| `{row['feature']}` | {row['FN_mean']:.4f} | {row['TP_mean']:.4f} | {row['abs_diff']:.4f} |")

lines += [
    "",
    "---",
    "",
    "## 7. Figuras Geradas",
    "",
    "- `reports/figures/roc_curve.png`",
    "- `reports/figures/ks_curve.png`",
    "- `reports/figures/lift_curve.png`",
    "- `reports/figures/score_distribution.png`",
    "- `reports/figures/confusion_matrix.png`",
    "- `reports/figures/feature_importance_final.png`",
    "",
    "---",
    "*Gerado por `src/evaluation/evaluate_champion.py`*",
]

report_path = REPORTS_DIR / "evaluation_report.md"
report_path.write_text("\n".join(lines), encoding="utf-8")
print(f"Report saved: {report_path}")

print("\n" + "=" * 50)
print("EVALUATION COMPLETE")
print("=" * 50)
print(f"AUC-ROC  : {auc:.4f}  (target >= 0.75)")
print(f"KS       : {ks_stat:.4f}  (target >= 0.35)")
print(f"Gini     : {gini:.4f}  (target >= 0.50)")
print(f"Recall+  : {recall_good_final:.4f}  (target >= 0.70)")
print(f"Threshold: {elected_thr:.2f}")
print(f"Status   : {overall_status}")
