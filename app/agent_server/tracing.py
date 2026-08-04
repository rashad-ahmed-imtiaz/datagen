from __future__ import annotations

import os


def configure_tracing() -> None:
    try:  # pragma: no cover - depends on Databricks MLflow runtime
        import mlflow
    except Exception:
        return
    experiment_id = os.getenv("SDF_MLFLOW_EXPERIMENT_ID")
    if experiment_id:
        mlflow.set_experiment(experiment_id=experiment_id)
    try:
        mlflow.openai.autolog()
    except Exception:
        pass
