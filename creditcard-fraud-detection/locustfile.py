"""Load testing for PaySim Fraud Detection API."""
import random
from locust import HttpUser, task, between

TX_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

class FraudAPIUser(HttpUser):
    wait_time = between(0.5, 2)
    host = "http://localhost:8000"

    def on_start(self):
        self.headers = {"X-API-Key": "user-key-456"}

    @task(10)
    def predict_raw(self):
        self.client.post("/predict/raw", json={
            "step": random.randint(1, 744),
            "type": random.choice(TX_TYPES),
            "amount": random.uniform(100, 500000),
            "oldbalanceOrg": random.uniform(0, 1000000),
            "newbalanceOrig": random.uniform(0, 500000),
            "oldbalanceDest": random.uniform(0, 500000),
            "newbalanceDest": random.uniform(0, 1000000),
        }, headers=self.headers)

    @task(2)
    def predict_batch(self):
        txs = [{"step": 1, "type": "TRANSFER", "amount": random.uniform(100, 50000),
                "oldbalanceOrg": 100000, "newbalanceOrig": 0,
                "oldbalanceDest": 0, "newbalanceDest": 0} for _ in range(10)]
        self.client.post("/predict/batch", json={"transactions": txs}, headers=self.headers)

    @task(1)
    def health(self):
        self.client.get("/health")
