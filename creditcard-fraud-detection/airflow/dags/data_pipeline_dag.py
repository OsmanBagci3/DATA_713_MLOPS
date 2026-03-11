"""DAG: Extract and preprocess PaySim data."""
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

DATA_RAW = "/tmp/data/raw/paysim.csv"
DATA_PROCESSED = "/tmp/data/processed"


def task_download(**kwargs):
    from data.download import download_dataset
    download_dataset(output_path=DATA_RAW)


def task_preprocess(**kwargs):
    from data.preprocess import load_and_preprocess
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
