"""
MLflow setup and logging utilities for PoD Bank credit score experiments.

Provides helpers to initialize tracking, log cross-validation results,
and check whether model metrics meet project targets.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

EXPERIMENT_NAME = "pod-bank-credit-score"

# Minimum thresholds defined in CLAUDE.md success metrics
_TARGETS: dict[str, float] = {
    "auc_roc": 0.75,
    "ks": 0.35,
    "gini": 0.50,
}

# Root of the repository (two levels up from this file: src/models → src → root)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TRACKING_URI = (_PROJECT_ROOT / "mlruns").as_uri()  # file:// URI for Windows compat


# ── Setup ─────────────────────────────────────────────────────────────────────


def setup_mlflow(experiment_name: str = EXPERIMENT_NAME) -> str:
    """Configure MLflow tracking and return the experiment ID.

    Sets the tracking URI to ``<project_root>/mlruns/`` and creates the
    experiment if it does not exist yet.

    Parameters
    ----------
    experiment_name:
        Human-readable name for the MLflow experiment.

    Returns
    -------
    str
        The MLflow experiment ID.
    """
    mlflow.set_tracking_uri(_TRACKING_URI)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)
    return experiment_id


# ── Target checker ────────────────────────────────────────────────────────────


def check_targets(metrics: dict[str, float]) -> dict[str, Any]:
    """Verify whether mean CV metrics meet the project success targets.

    Parameters
    ----------
    metrics:
        Dictionary that must contain the keys ``auc_roc``, ``ks``, and
        ``gini`` with their mean values from cross-validation.

    Returns
    -------
    dict
        ``{
            "status": "APROVADO" | "REPROVADO",
            "checks": {"auc_roc": bool, "ks": bool, "gini": bool},
            "details": {"auc_roc": {"value": float, "target": float, "passed": bool}, ...}
        }``
    """
    checks: dict[str, bool] = {}
    details: dict[str, dict[str, Any]] = {}

    for metric, threshold in _TARGETS.items():
        value = metrics.get(metric, 0.0)
        passed = value >= threshold
        checks[metric] = passed
        details[metric] = {"value": value, "target": threshold, "passed": passed}

    status = "APROVADO" if all(checks.values()) else "REPROVADO"
    return {"status": status, "checks": checks, "details": details}


# ── CV logging ────────────────────────────────────────────────────────────────


def log_cv_results(
    run_name: str,
    model: Any,
    params: dict[str, Any],
    cv_metrics: dict[str, list[float]],
    feature_importance: pd.DataFrame | None = None,
) -> str:
    """Log a cross-validated model run to MLflow and return the run ID.

    Expected keys in ``cv_metrics``
    --------------------------------
    ``auc_roc``, ``ks``, ``gini`` — each a list of per-fold scalar values.

    What gets logged
    ----------------
    * All ``params`` entries as MLflow parameters.
    * Mean and std for each metric (e.g. ``auc_roc_mean``, ``auc_roc_std``).
    * Per-fold values (e.g. ``fold_1_auc``, …, ``fold_5_auc``).
    * Tags: ``model_type`` (class name), ``phase`` = "modeling",
      ``status`` = "APROVADO" or "REPROVADO".
    * If ``feature_importance`` is provided, saves it as ``feature_importance.csv``
      artifact.
    * The model artifact via ``mlflow.sklearn.log_model`` (or
      ``mlflow.lightgbm.log_model`` when the model is a LightGBM Booster/
      scikit-learn wrapper).

    Parameters
    ----------
    run_name:
        Descriptive label shown in the MLflow UI for this run.
    model:
        Fitted scikit-learn–compatible estimator or LightGBM model.
    params:
        Hyperparameters used to train ``model``.
    cv_metrics:
        Per-fold metric values keyed by metric name.
    feature_importance:
        Optional DataFrame with columns ``feature`` and ``importance``.
        Saved as a CSV artifact when provided.

    Returns
    -------
    str
        MLflow run ID.
    """
    import numpy as np

    mean_metrics = {k: float(np.mean(v)) for k, v in cv_metrics.items()}
    target_result = check_targets(mean_metrics)

    with mlflow.start_run(run_name=run_name) as run:
        # ── Parameters ────────────────────────────────────────────────────
        mlflow.log_params(params)

        # ── Aggregate metrics ─────────────────────────────────────────────
        for metric, values in cv_metrics.items():
            arr = np.array(values, dtype=float)
            mlflow.log_metric(f"{metric}_mean", float(arr.mean()))
            mlflow.log_metric(f"{metric}_std", float(arr.std()))

        # ── Per-fold AUC ──────────────────────────────────────────────────
        auc_folds = cv_metrics.get("auc_roc", [])
        for i, val in enumerate(auc_folds, start=1):
            mlflow.log_metric(f"fold_{i}_auc", float(val))

        # ── Tags ──────────────────────────────────────────────────────────
        mlflow.set_tags(
            {
                "model_type": type(model).__name__,
                "phase": "modeling",
                "status": target_result["status"],
            }
        )

        # ── Feature importance artifact ───────────────────────────────────
        if feature_importance is not None:
            # Use tempfile to avoid Windows path/URL confusion with pandas
            fd, tmp_path = tempfile.mkstemp(suffix=".csv", prefix="fi_")
            try:
                os.close(fd)
                feature_importance.to_csv(tmp_path, index=False)
                mlflow.log_artifact(tmp_path, artifact_path="feature_importance")
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        # ── Model artifact ────────────────────────────────────────────────
        _log_model(model)

        return run.info.run_id


# ── Internal helpers ──────────────────────────────────────────────────────────


def _log_model(model: Any) -> None:
    """Dispatch model logging to the right MLflow flavour."""
    model_cls = type(model).__name__

    # LightGBM native Booster or sklearn wrapper
    if "lightgbm" in type(model).__module__.lower():
        try:
            import mlflow.lightgbm as mlflow_lgb

            mlflow_lgb.log_model(model, artifact_path="model")
            return
        except Exception:
            pass  # fall through to sklearn flavour

    mlflow.sklearn.log_model(model, artifact_path="model")
