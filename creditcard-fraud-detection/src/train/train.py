"""Train fraud detection model on PaySim data with MLflow tracking."""

import os
import logging
import pandas as pd
import mlflow
import mlflow.sklearn
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def train_model(
    data_dir: str,
    experiment_name: str = "paysim-fraud-detection",
    model_name: str = "paysim-fraud-detector",
) -> dict:
    """
    Train a RandomForest on PaySim features and log to MLflow.

    Args:
        data_dir: Directory with processed parquet files
        experiment_name: MLflow experiment name
        model_name: Registered model name

    Returns:
        Dict of evaluation metrics
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    X_train = pd.read_parquet(os.path.join(data_dir, "X_train.parquet"))
    X_test = pd.read_parquet(os.path.join(data_dir, "X_test.parquet"))
    y_train = pd.read_parquet(os.path.join(data_dir, "y_train.parquet")).squeeze()
    y_test = pd.read_parquet(os.path.join(data_dir, "y_test.parquet")).squeeze()

    logger.info(f"Training: {X_train.shape}, Test: {X_test.shape}")
    logger.info(f"Fraud in train (before SMOTE): {y_train.sum()} ({y_train.mean():.4%})")

    # --- SMOTE: oversample minority class to handle imbalanced data ---
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    logger.info(
        f"After SMOTE: {X_train_res.shape[0]} samples, "
        f"fraud={y_train_res.sum()} ({y_train_res.mean():.4%})"
    )

    with mlflow.start_run() as run:
        params = {
            "n_estimators": 100,
            "max_depth": 12,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
        }
        mlflow.log_params(params)
        mlflow.log_param("train_size_original", len(X_train))
        mlflow.log_param("train_size_after_smote", len(X_train_res))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("fraud_ratio_train_original", float(y_train.mean()))
        mlflow.log_param("fraud_ratio_train_smote", float(y_train_res.mean()))
        mlflow.log_param("dataset", "PaySim")
        mlflow.log_param("oversampling", "SMOTE")

        logger.info("Training RandomForest on SMOTE-resampled data...")
        model = RandomForestClassifier(**params)
        model.fit(X_train_res, y_train_res)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "f1": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "avg_precision": average_precision_score(y_test, y_proba),
        }
        mlflow.log_metrics(metrics)

        cm = confusion_matrix(y_test, y_pred)
        mlflow.log_metric("true_negatives", int(cm[0][0]))
        mlflow.log_metric("false_positives", int(cm[0][1]))
        mlflow.log_metric("false_negatives", int(cm[1][0]))
        mlflow.log_metric("true_positives", int(cm[1][1]))

        # Feature importances
        feature_importance = dict(zip(X_train.columns, model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        for feat_name, importance in top_features:
            mlflow.log_metric(f"importance_{feat_name}", importance)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=model_name,
        )

        logger.info(f"Run ID: {run.info.run_id}")
        logger.info(f"Metrics: {metrics}")
        return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_model(data_dir="/tmp/data/processed")
