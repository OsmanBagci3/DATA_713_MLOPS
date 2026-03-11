"""DAG: Simulate live transaction streaming."""
import sys, os, json
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, "/opt/airflow/src")

default_args = {"owner": "mlops-team", "retries": 1, "retry_delay": timedelta(minutes=2)}


def task_simulate(**kwargs):
    from data.preprocess import simulate_live_data
    raw_path = "/tmp/data/raw/paysim.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset not at {raw_path}")

    batch = simulate_live_data(raw_path, batch_size=50)
    staging = "/tmp/data/staging"
    os.makedirs(staging, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_file = os.path.join(staging, f"batch_{ts}.jsonl")
    with open(batch_file, "w") as f:
        for _, row in batch.iterrows():
            f.write(json.dumps(row.to_dict()) + "\n")
    return {"batch_file": batch_file, "n": len(batch)}


def task_predict(**kwargs):
    import requests
    info = kwargs["ti"].xcom_pull(task_ids="simulate")
    api = os.getenv("API_URL", "http://api:8000")
    headers = {"X-API-Key": "admin-key-123"}
    results = []
    with open(info["batch_file"]) as f:
        for line in f:
            tx = json.loads(line)
            payload = {
                "step": int(tx.get("step", 1)),
                "type": tx.get("type", "TRANSFER"),
                "amount": float(tx.get("amount", 0)),
                "oldbalanceOrg": float(tx.get("oldbalanceOrg", 0)),
                "newbalanceOrig": float(tx.get("newbalanceOrig", 0)),
                "oldbalanceDest": float(tx.get("oldbalanceDest", 0)),
                "newbalanceDest": float(tx.get("newbalanceDest", 0)),
            }
            try:
                r = requests.post(f"{api}/predict/raw", json=payload, headers=headers, timeout=10)
                res = r.json()
                res["actual"] = int(tx.get("isFraud", 0))
                results.append(res)
            except Exception as e:
                results.append({"error": str(e)})
    n_fraud = sum(1 for r in results if r.get("is_fraud"))
    return {"total": len(results), "predicted_fraud": n_fraud}


with DAG(
    "live_data_simulation",
    default_args=default_args,
    description="Simulate live PaySim transactions",
    schedule_interval="*/30 * * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["live", "simulation"],
) as dag:
    t1 = PythonOperator(task_id="simulate", python_callable=task_simulate)
    t2 = PythonOperator(task_id="predict", python_callable=task_predict)
    t1 >> t2
