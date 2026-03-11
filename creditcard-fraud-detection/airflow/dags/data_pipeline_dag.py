"""DAG: Extract and preprocess PaySim data.

In this project, the PaySim CSV is expected to be uploaded to MinIO (S3-compatible)
in bucket `data` with key `paysim.csv` (configurable via env).

The DAG downloads the CSV from MinIO into the shared Airflow volume under
`/tmp/data/raw/` so subsequent tasks (preprocess/train/live simulation) can reuse it.
"""

import os
import sys
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, "/opt/airflow/src")

default_args = {
    "owner": "mlops-team",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": True,
    "email": ["team@paysim-fraud.local"],
}

DATA_RAW = os.getenv("PAYSIM_LOCAL_PATH", "/tmp/data/raw/paysim.csv")
DATA_PROCESSED = os.getenv("PAYSIM_PROCESSED_DIR", "/tmp/data/processed")


def task_download(**kwargs):
    from data.download import download_dataset
    os.makedirs(os.path.dirname(DATA_RAW), exist_ok=True)
    download_dataset(output_path=DATA_RAW)


def task_preprocess(**kwargs):
    from data.preprocess import load_and_preprocess
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    stats = load_and_preprocess(input_path=DATA_RAW, output_dir=DATA_PROCESSED)
    kwargs["ti"].xcom_push(key="stats", value=stats)


def task_validate(**kwargs):
    import pandas as pd
    stats = kwargs["ti"].xcom_pull(task_ids="preprocess", key="stats")
    assert stats["total_samples"] > 10000, "Dataset too small"
    assert stats["n_features"] == 15, f"Expected 15 features, got {stats['n_features']}"
    assert 0 < stats["fraud_ratio"] < 0.05, "Unexpected fraud ratio"
    X = pd.read_parquet(f"{DATA_PROCESSED}/X_train.parquet")
    assert not X.isnull().any().any(), "Null values found"


with DAG(
    "data_pipeline",
    default_args=default_args,
    description="Download and preprocess PaySim dataset",
    schedule_interval="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["data", "paysim"],
) as dag:
    t1 = PythonOperator(task_id="download", python_callable=task_download)
    t2 = PythonOperator(task_id="preprocess", python_callable=task_preprocess)
    t3 = PythonOperator(task_id="validate", python_callable=task_validate)
    t1 >> t2 >> t3
