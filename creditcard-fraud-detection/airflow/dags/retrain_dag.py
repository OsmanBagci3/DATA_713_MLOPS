"""DAG: Continuous Training pipeline."""
import sys, os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.email import EmailOperator
from airflow.operators.empty import EmptyOperator

sys.path.insert(0, "/opt/airflow/src")

default_args = {
    "owner": "mlops-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": True,
    "email": ["team@paysim-fraud.local"],
}


def task_check_trigger(**kwargs):
    import json
    from pathlib import Path

    feedback_file = "/tmp/data/feedback/feedback.jsonl"
    drift_report_file = "/tmp/data/processed/drift_report.json"
    reasons = []

    # --- Feedback-based trigger ---
    if os.path.exists(feedback_file):
        with open(feedback_file) as f:
            count = sum(1 for _ in f)
        if count >= 10:
            reasons.append(f"feedback ({count} entries)")

    # --- Drift-based trigger (KL divergence) ---
    if os.path.exists(drift_report_file):
        with open(drift_report_file) as f:
            report = json.load(f)
        if report.get("drift_detected"):
            drifted = report.get("drifted_features", [])
            max_kl = report.get("max_kl", 0)
            reasons.append(
                f"data_drift ({len(drifted)} feature(s) drifted, max KL={max_kl:.4f})"
            )

    reason = "; ".join(reasons) if reasons else "scheduled"
    kwargs["ti"].xcom_push(key="reason", value=reason)


def task_train(**kwargs):
    from train.train import train_model
    metrics = train_model(data_dir="/tmp/data/processed")
    kwargs["ti"].xcom_push(key="metrics", value=metrics)


def task_evaluate(**kwargs):
    from train.evaluate import promote_if_better
    promoted = promote_if_better()
    return "reload_api" if promoted else "join"


def task_reload_api(**kwargs):
    import requests
    api_url = os.getenv("API_URL", "http://api:8000")
    try:
        requests.post(f"{api_url}/model/reload",
                       headers={"X-API-Key": "admin-key-123"}, timeout=30)
    except Exception as e:
        return f"Reload failed: {e}"


def task_cleanup(**kwargs):
    import shutil
    f = "/tmp/data/feedback/feedback.jsonl"
    if os.path.exists(f):
        shutil.move(f, f.replace(".jsonl", f"_{datetime.now():%Y%m%d_%H%M%S}.jsonl"))


with DAG(
    "retrain_pipeline",
    default_args=default_args,
    description="Continuous Training with model promotion",
    schedule_interval="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["training", "CT"],
) as dag:
    t_check = PythonOperator(task_id="check_trigger", python_callable=task_check_trigger)
    t_train = PythonOperator(task_id="train", python_callable=task_train)
    t_eval = BranchPythonOperator(task_id="evaluate", python_callable=task_evaluate)
    t_reload = PythonOperator(task_id="reload_api", python_callable=task_reload_api,
                               trigger_rule="none_failed_min_one_success")
    t_cleanup = PythonOperator(task_id="cleanup", python_callable=task_cleanup,
                                trigger_rule="none_failed")
    t_join = EmptyOperator(task_id="join", trigger_rule="none_failed_min_one_success")

    t_check >> t_train >> t_eval
    t_eval >> t_reload >> t_join
    t_join >> t_cleanup
