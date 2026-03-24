"""
test_pipeline.py — Smoke test for the production scoring pipeline.

Selects 5 real clients from train_final.parquet:
  - 2 with TARGET=0 (good payers)
  - 2 with TARGET=1 (defaulters)
  - 1 borderline: predicted score between 0.40 and 0.55

Measures inference time for each call and reports average latency.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("C:/Users/magno/OneDrive/Desktop/pod-bank-credit-score")
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predict import predict_score, _load_pipeline

DATA_PATH = PROJECT_ROOT / "data/processed/train_final.parquet"

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading train_final.parquet ...")
df = pd.read_parquet(DATA_PATH)
print(f"  Shape: {df.shape}")

# Pre-warm the pipeline (load model once before timing)
print("\nPre-warming pipeline (loading model) ...")
pipeline = _load_pipeline()
print(f"  Pipeline version: {pipeline['version']}")
print(f"  Threshold       : {pipeline['threshold']}")
print(f"  Features        : {len(pipeline['feature_columns'])}")

feature_columns = pipeline["feature_columns"]
model = pipeline["model"]

# ── Score full sample to find borderline client ────────────────────────────────
print("\nScoring full dataset to find borderline client ...")
X_all = df[feature_columns].copy()
cat_cols = X_all.select_dtypes(include=["object"]).columns.tolist()
for col in cat_cols:
    X_all[col] = X_all[col].astype("category")

probas = model.predict_proba(X_all)[:, 1]
df["_proba"] = probas

# ── Select 5 clients ──────────────────────────────────────────────────────────
rng = np.random.default_rng(42)

good_payers  = df[df["TARGET"] == 0].sample(2, random_state=42)
defaulters   = df[df["TARGET"] == 1].sample(2, random_state=42)

borderline   = df[(df["_proba"] >= 0.40) & (df["_proba"] <= 0.55)]
if len(borderline) == 0:
    # Fallback: closest to threshold
    df["_dist"] = (df["_proba"] - 0.48).abs()
    borderline_row = df.nsmallest(1, "_dist")
else:
    borderline_row = borderline.sample(1, random_state=7)

selected = pd.concat([good_payers, defaulters, borderline_row], ignore_index=True)
print(f"  Selected {len(selected)} clients")
print(f"  Borderline score: {borderline_row['_proba'].iloc[0]:.4f}")

# ── Run predict_score and collect results ─────────────────────────────────────
print("\n" + "=" * 70)
print(f"{'SK_ID_CURR':>12}  {'TARGET':>6}  {'Score':>6}  {'Decision':>10}  {'Band':>6}  {'ms':>6}")
print("=" * 70)

results = []
for _, row in selected.iterrows():
    sk_id     = int(row["SK_ID_CURR"])
    target    = int(row["TARGET"])
    applicant = row[feature_columns].to_dict()

    t0     = time.perf_counter()
    result = predict_score(applicant)
    wall   = (time.perf_counter() - t0) * 1000.0

    results.append({
        "SK_ID_CURR"    : sk_id,
        "TARGET_real"   : target,
        "score"         : result["score"],
        "decision"      : result["decision"],
        "risk_band"     : result["risk_band"],
        "inference_ms"  : result["inference_ms"],
        "wall_ms"       : wall,
        "top_factors"   : result["top_factors"],
    })

    print(
        f"  {sk_id:>10}  {target:>6}  {result['score']:>6.4f}"
        f"  {result['decision']:>10}  {result['risk_band']:>6}  {result['inference_ms']:>6.2f}ms"
    )

print("=" * 70)

avg_ms = np.mean([r["inference_ms"] for r in results])
max_ms = np.max([r["inference_ms"] for r in results])
print(f"\nAverage inference time : {avg_ms:.2f} ms")
print(f"Max inference time     : {max_ms:.2f} ms")
sla_ok = avg_ms < 500
print(f"SLA < 500ms            : {'PASS' if sla_ok else 'FAIL'}")

# ── Top factors for each client ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("TOP FACTORS PER CLIENT")
print("=" * 70)
for r in results:
    print(f"\n  SK_ID_CURR={r['SK_ID_CURR']}  TARGET={r['TARGET_real']}  score={r['score']:.4f}")
    for f in r["top_factors"]:
        print(f"    {f['feature']:<35} value={f['value']}  importance={f['importance']:.1f}")

# ── Export summary for report ─────────────────────────────────────────────────
summary_rows = []
for r in results:
    summary_rows.append({
        "SK_ID_CURR"  : r["SK_ID_CURR"],
        "TARGET_real" : r["TARGET_real"],
        "score"       : round(r["score"], 4),
        "decision"    : r["decision"],
        "risk_band"   : r["risk_band"],
        "inference_ms": round(r["inference_ms"], 2),
    })

summary_df = pd.DataFrame(summary_rows)
print("\n\nSUMMARY TABLE (for report)")
print(summary_df.to_string(index=False))

# Save summary for the report-generation step
import json
out = {
    "avg_inference_ms": round(avg_ms, 2),
    "max_inference_ms": round(max_ms, 2),
    "sla_pass"        : sla_ok,
    "rows"            : summary_rows,
    "top_factors"     : {
        r["SK_ID_CURR"]: r["top_factors"] for r in results
    },
}
out_path = PROJECT_ROOT / "reports" / "_pipeline_test_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\nTest results saved to: {out_path}")
