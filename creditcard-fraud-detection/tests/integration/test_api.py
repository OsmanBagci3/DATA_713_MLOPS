"""Integration tests for the PaySim Fraud Detection API."""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

mock_model = MagicMock()
mock_model.predict_proba.return_value = np.array([[0.85, 0.15]])
mock_model.predict.return_value = np.array([0])

ADMIN_KEY = "admin-key-123"
USER_KEY = "user-key-456"


@pytest.fixture
def client():
    with patch("api.main.MODEL", mock_model):
        from api.main import app
        with TestClient(app) as c:
            yield c


SAMPLE_RAW = {
    "step": 1, "type": "TRANSFER", "amount": 181000,
    "oldbalanceOrg": 181000, "newbalanceOrig": 0,
    "oldbalanceDest": 0, "newbalanceDest": 0,
}

SAMPLE_FEATURES = [float(x) for x in np.random.randn(15)]


class TestHealth:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestPredictRaw:
    def test_valid(self, client):
        r = client.post("/predict/raw", json=SAMPLE_RAW, headers={"X-API-Key": ADMIN_KEY})
        assert r.status_code == 200
        d = r.json()
        assert "is_fraud" in d
        assert "confidence" in d
        assert "risk_level" in d

    def test_no_key(self, client):
        r = client.post("/predict/raw", json=SAMPLE_RAW)
        assert r.status_code == 403

    def test_invalid_key(self, client):
        r = client.post("/predict/raw", json=SAMPLE_RAW, headers={"X-API-Key": "bad"})
        assert r.status_code == 403


class TestPredictFeatures:
    def test_valid(self, client):
        r = client.post("/predict", json={"features": SAMPLE_FEATURES}, headers={"X-API-Key": ADMIN_KEY})
        assert r.status_code == 200

    def test_wrong_count(self, client):
        r = client.post("/predict", json={"features": [1.0, 2.0]}, headers={"X-API-Key": ADMIN_KEY})
        assert r.status_code == 422


class TestBatch:
    def test_batch(self, client):
        r = client.post("/predict/batch",
                         json={"transactions": [SAMPLE_RAW, SAMPLE_RAW]},
                         headers={"X-API-Key": ADMIN_KEY})
        assert r.status_code == 200
        assert len(r.json()["predictions"]) == 2


class TestFeedback:
    def test_submit(self, client):
        r = client.post("/feedback",
                         json={"transaction": SAMPLE_RAW, "correct_label": 1, "comment": "test"},
                         headers={"X-API-Key": ADMIN_KEY})
        assert r.status_code == 200


class TestReload:
    def test_user_forbidden(self, client):
        r = client.post("/model/reload", headers={"X-API-Key": USER_KEY})
        assert r.status_code == 403
