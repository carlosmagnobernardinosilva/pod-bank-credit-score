"""
Fase 6 — Deployment: Registro do pipeline de producao no MLflow.

Registra o scoring_pipeline.pkl como artefato de producao no experimento
"pod-bank-deployment", com metricas, parametros e artefatos associados.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import mlflow

from models.mlflow_setup import setup_mlflow

# ── Paths ──────────────────────────────────────────────────────────────────────

_PIPELINE_PKL = _PROJECT_ROOT / "models" / "scoring_pipeline.pkl"
_PIPELINE_REPORT = _PROJECT_ROOT / "reports" / "pipeline_report.md"
_PREDICT_PY = _PROJECT_ROOT / "src" / "models" / "predict.py"

EXPERIMENT_NAME = "pod-bank-deployment"


def main() -> None:
    # Configure tracking URI (same mlruns/ directory used by all other tasks)
    setup_mlflow(EXPERIMENT_NAME)

    # Ensure the deployment experiment exists (setup_mlflow already set it,
    # but we call get_experiment_by_name to be explicit)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="scoring_pipeline_v1") as run:
        # ── Tags ──────────────────────────────────────────────────────────────
        mlflow.set_tags(
            {
                "stage": "production",
                "pipeline_version": "1.0-tuned",
                "model_type": "lightgbm_tuned",
            }
        )

        # ── Parameters ────────────────────────────────────────────────────────
        mlflow.log_params(
            {
                "threshold": 0.48,
                "feature_count": 62,
                "model_version": "lightgbm_tuned",
                "pipeline_version": "1.0-tuned",
            }
        )

        # ── Metrics ───────────────────────────────────────────────────────────
        mlflow.log_metrics(
            {
                "auc_roc": 0.7701,
                "ks": 0.4105,
                "gini": 0.5401,
                "avg_inference_ms": 7.76,
                "max_inference_ms": 8.02,
                "sla_pass": 1,
            }
        )

        # ── Artifacts ─────────────────────────────────────────────────────────
        if _PIPELINE_PKL.exists():
            mlflow.log_artifact(str(_PIPELINE_PKL), artifact_path="pipeline")
        else:
            print(f"WARNING: pipeline artifact not found at {_PIPELINE_PKL}")

        if _PIPELINE_REPORT.exists():
            mlflow.log_artifact(str(_PIPELINE_REPORT), artifact_path="reports")
        else:
            print(f"WARNING: pipeline report not found at {_PIPELINE_REPORT}")

        if _PREDICT_PY.exists():
            mlflow.log_artifact(str(_PREDICT_PY), artifact_path="src")
        else:
            print(f"WARNING: predict.py not found at {_PREDICT_PY}")

        run_id = run.info.run_id

    print(f"MLflow run registered successfully.")
    print(f"Experiment : {EXPERIMENT_NAME}")
    print(f"Run ID     : {run_id}")
    return run_id


if __name__ == "__main__":
    main()
