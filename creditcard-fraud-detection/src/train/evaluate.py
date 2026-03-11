"""Evaluate and promote models in MLflow registry."""

import os
import logging
import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def promote_if_better(
    model_name: str = "paysim-fraud-detector",
    metric: str = "f1",
    min_threshold: float = 0.5,
) -> bool:
    """
    Compare latest model with production. Promote if better.

    Returns True if a new model was promoted.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    try:
        latest_versions = client.get_latest_versions(model_name, stages=["None"])
    except Exception as e:
        logger.error(f"No registered model found: {e}")
        return False

    if not latest_versions:
        logger.info("No new model versions to evaluate.")
        return False

    latest_version = latest_versions[0]
    latest_run = client.get_run(latest_version.run_id)
    latest_score = latest_run.data.metrics.get(metric, 0)

    logger.info(f"Latest v{latest_version.version}: {metric}={latest_score:.4f}")

    if latest_score < min_threshold:
        logger.warning(f"Score {latest_score:.4f} below threshold {min_threshold}.")
        return False

    prod_versions = client.get_latest_versions(model_name, stages=["Production"])

    if not prod_versions:
        logger.info("No production model. Promoting latest.")
        client.transition_model_version_stage(
            name=model_name, version=latest_version.version, stage="Production",
        )
        return True

    prod_version = prod_versions[0]
    prod_run = client.get_run(prod_version.run_id)
    prod_score = prod_run.data.metrics.get(metric, 0)

    logger.info(f"Production v{prod_version.version}: {metric}={prod_score:.4f}")

    if latest_score > prod_score:
        logger.info(f"Promoting: {latest_score:.4f} > {prod_score:.4f}")
        client.transition_model_version_stage(
            name=model_name, version=prod_version.version, stage="Archived",
        )
        client.transition_model_version_stage(
            name=model_name, version=latest_version.version, stage="Production",
        )
        return True

    logger.info(f"No improvement: {latest_score:.4f} <= {prod_score:.4f}")
    return False


def rollback_model(model_name: str = "paysim-fraud-detector") -> bool:
    """Rollback to previous production model."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    archived_versions = client.get_latest_versions(model_name, stages=["Archived"])

    if not prod_versions or not archived_versions:
        logger.error("Cannot rollback: missing production or archived model.")
        return False

    current_prod = prod_versions[0]
    rollback_target = archived_versions[0]

    logger.info(f"Rolling back v{current_prod.version} -> v{rollback_target.version}")

    client.transition_model_version_stage(
        name=model_name, version=current_prod.version, stage="Archived",
    )
    client.transition_model_version_stage(
        name=model_name, version=rollback_target.version, stage="Production",
    )
    return True
