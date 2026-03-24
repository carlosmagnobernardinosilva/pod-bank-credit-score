"""
predict.py — Production scoring pipeline for PoD Bank credit scoring.

Usage:
    from src.models.predict import predict_score
    result = predict_score(applicant_dict)

Returns a dict with:
    score      : float — probability of default (0.0 to 1.0)
    decision   : str   — 'APROVADO' | 'REPROVADO'
    risk_band  : str   — 'BAIXO' | 'MEDIO' | 'ALTO'
    top_factors: list  — top 5 features influencing this score
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_PATH = PROJECT_ROOT / "models" / "scoring_pipeline.pkl"

# ── Singleton state ────────────────────────────────────────────────────────────
_pipeline: dict | None = None


def _load_pipeline() -> dict:
    """Load pipeline from disk (lazy singleton)."""
    global _pipeline
    if _pipeline is None:
        _pipeline = joblib.load(PIPELINE_PATH)
    return _pipeline


# ── Preprocessing ─────────────────────────────────────────────────────────────
_DAYS_SENTINEL = 365243


def _preprocess(raw: dict[str, Any], feature_columns: list[str]) -> pd.DataFrame:
    """
    Apply preprocessing rules and align columns with the trained model.

    Rules:
    - Replace 365243 (sentinel for missing) with NaN in DAYS_* columns.
    - Replace 'XNA' and 'XAP' categorical values with 'Unknown'.
    - Fill missing columns (not provided by caller) with NaN.
    - Select and order columns exactly as the model expects.
    """
    row = dict(raw)

    # Replace DAYS sentinel
    for key, val in row.items():
        if key.startswith("DAYS_") and val == _DAYS_SENTINEL:
            row[key] = np.nan

    # Replace XNA/XAP in string fields
    for key, val in row.items():
        if isinstance(val, str) and val in ("XNA", "XAP"):
            row[key] = "Unknown"

    df = pd.DataFrame([row])

    # Align to expected feature columns — missing cols become NaN
    for col in feature_columns:
        if col not in df.columns:
            df[col] = np.nan

    # Keep only model columns, in correct order
    df = df[feature_columns]

    # Cast object columns to category for LightGBM
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        df[col] = df[col].astype("category")

    return df


# ── Top factors ───────────────────────────────────────────────────────────────

def _get_top_factors(
    model: Any,
    feature_columns: list[str],
    row_df: pd.DataFrame,
    top_n: int = 5,
) -> list[dict]:
    """
    Return top_n features ranked by model feature importance (gain),
    together with their actual values for this applicant.
    """
    try:
        importances = model.booster_.feature_importance(importance_type="gain")
    except AttributeError:
        # Fallback for sklearn-style models
        try:
            importances = model.feature_importances_
        except AttributeError:
            return []

    fi_series = pd.Series(importances, index=feature_columns)
    top_features = fi_series.nlargest(top_n).index.tolist()

    factors = []
    for feat in top_features:
        val = row_df[feat].iloc[0]
        factors.append({
            "feature": feat,
            "value": None if pd.isna(val) else val,
            "importance": float(fi_series[feat]),
        })
    return factors


# ── Main scoring function ──────────────────────────────────────────────────────

def predict_score(applicant: dict[str, Any]) -> dict:
    """
    Score a single loan applicant.

    Parameters
    ----------
    applicant : dict
        Raw feature values for one applicant. Keys are column names;
        only known features are used — extras are silently ignored.

    Returns
    -------
    dict with keys:
        score       : float   — P(default), range [0, 1]
        decision    : str     — 'APROVADO' if score < threshold, else 'REPROVADO'
        risk_band   : str     — 'BAIXO' (<0.20), 'MEDIO' (0.20-0.48), 'ALTO' (>=0.48)
        top_factors : list    — top 5 features [{feature, value, importance}]
        inference_ms: float   — latency in milliseconds
    """
    t0 = time.perf_counter()

    pipeline = _load_pipeline()
    model = pipeline["model"]
    feature_columns: list[str] = pipeline["feature_columns"]
    threshold: float = pipeline["threshold"]

    row_df = _preprocess(applicant, feature_columns)
    score = float(model.predict_proba(row_df)[0, 1])

    # Decision
    decision = "APROVADO" if score < threshold else "REPROVADO"

    # Risk band
    if score < 0.20:
        risk_band = "BAIXO"
    elif score < 0.48:
        risk_band = "MEDIO"
    else:
        risk_band = "ALTO"

    top_factors = _get_top_factors(model, feature_columns, row_df)

    inference_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "score": score,
        "decision": decision,
        "risk_band": risk_band,
        "top_factors": top_factors,
        "inference_ms": inference_ms,
    }
