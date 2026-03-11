"""FastAPI application for PaySim fraud detection."""

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import mlflow.sklearn
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

logger = logging.getLogger(__name__)

# ---- Prometheus Metrics ----
PREDICTIONS_TOTAL = Counter("predictions_total", "Total predictions", ["result"])
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency")
PREDICTION_CONFIDENCE = Histogram("prediction_confidence", "Confidence scores")
PREDICTIONS_BY_TYPE = Counter("predictions_by_type", "Predictions by tx type", ["tx_type"])

MODEL = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    try:
        MODEL = mlflow.sklearn.load_model("models:/paysim-fraud-detector/Production")
        logger.info("Model loaded from MLflow registry.")
    except Exception as e:
        logger.warning(f"Could not load model: {e}. API starts without model.")
    yield


app = FastAPI(
    title="PaySim Fraud Detection API",
    description="Detect fraudulent mobile money transactions (PaySim dataset).",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---- Auth ----
API_KEYS = {
    "admin-key-123": {"user": "admin", "role": "admin"},
    "user-key-456": {"user": "analyst", "role": "user"},
}
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key is None or api_key not in API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return API_KEYS[api_key]


# ---- Schemas ----
class RawTransaction(BaseModel):
    """A raw PaySim transaction before feature engineering."""
    step: int = Field(1, description="Hour of simulation (1-744)")
    type: str = Field(..., description="CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER")
    amount: float = Field(..., ge=0, description="Transaction amount")
    oldbalanceOrg: float = Field(..., ge=0, description="Sender balance before")
    newbalanceOrig: float = Field(..., ge=0, description="Sender balance after")
    oldbalanceDest: float = Field(..., ge=0, description="Receiver balance before")
    newbalanceDest: float = Field(..., ge=0, description="Receiver balance after")
    nameOrig: Optional[str] = Field(None, description="Sender ID (optional)")
    nameDest: Optional[str] = Field(None, description="Receiver ID (optional)")


class FeatureTransaction(BaseModel):
    """Pre-processed features (15 floats) for direct prediction."""
    features: list[float] = Field(..., min_length=15, max_length=15)


class PredictionResponse(BaseModel):
    is_fraud: bool
    confidence: float
    prediction_time_ms: float
    risk_level: str


class BatchRawTransaction(BaseModel):
    transactions: list[RawTransaction]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class FeedbackRequest(BaseModel):
    transaction: RawTransaction
    correct_label: int = Field(..., ge=0, le=1)
    comment: str = ""


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    environment: str


# ---- Feature engineering (same logic as preprocessing) ----
TYPE_MAPPING = {"CASH_IN": 0, "CASH_OUT": 1, "DEBIT": 2, "PAYMENT": 3, "TRANSFER": 4}


def raw_to_features(tx: RawTransaction) -> np.ndarray:
    """Convert a raw transaction to 15 engineered features."""
    type_encoded = TYPE_MAPPING.get(tx.type, 4)
    orig_diff = tx.oldbalanceOrg - tx.newbalanceOrig
    dest_diff = tx.newbalanceDest - tx.oldbalanceDest
    orig_error = orig_diff - tx.amount
    dest_error = dest_diff - tx.amount
    amount_ratio = tx.amount / tx.oldbalanceOrg if tx.oldbalanceOrg > 0 else 0
    is_orig_empty = int(tx.newbalanceOrig == 0)
    is_dest_empty = int(tx.oldbalanceDest == 0)
    step_hour = tx.step % 24
    step_day = tx.step // 24

    return np.array([[
        type_encoded, tx.amount, tx.oldbalanceOrg, tx.newbalanceOrig,
        tx.oldbalanceDest, tx.newbalanceDest,
        orig_diff, dest_diff, orig_error, dest_error, amount_ratio,
        is_orig_empty, is_dest_empty, step_hour, step_day,
    ]])


def get_risk_level(confidence: float) -> str:
    if confidence >= 0.8:
        return "CRITICAL"
    elif confidence >= 0.5:
        return "HIGH"
    elif confidence >= 0.3:
        return "MEDIUM"
    return "LOW"


def _predict(X: np.ndarray, tx_type: str = "UNKNOWN") -> PredictionResponse:
    start = time.time()
    proba = MODEL.predict_proba(X)[0][1]
    is_fraud = bool(proba > 0.5)
    latency_ms = (time.time() - start) * 1000

    PREDICTIONS_TOTAL.labels(result="fraud" if is_fraud else "legit").inc()
    PREDICTION_LATENCY.observe(latency_ms / 1000)
    PREDICTION_CONFIDENCE.observe(proba)
    PREDICTIONS_BY_TYPE.labels(tx_type=tx_type).inc()

    return PredictionResponse(
        is_fraud=is_fraud,
        confidence=float(proba),
        prediction_time_ms=round(latency_ms, 2),
        risk_level=get_risk_level(proba),
    )


# ---- Endpoints ----
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok", model_loaded=MODEL is not None,
        environment=os.getenv("ENV", "unknown"),
    )


@app.post("/predict/raw", response_model=PredictionResponse)
def predict_raw(tx: RawTransaction, user: dict = Depends(verify_api_key)):
    """Predict fraud from a raw PaySim transaction."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    X = raw_to_features(tx)
    result = _predict(X, tx.type)
    logger.info(f"Predict: type={tx.type}, amount={tx.amount}, fraud={result.is_fraud}, user={user['user']}")
    return result


@app.post("/predict", response_model=PredictionResponse)
def predict_features(tx: FeatureTransaction, user: dict = Depends(verify_api_key)):
    """Predict fraud from pre-processed features (15 floats)."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    X = np.array(tx.features).reshape(1, -1)
    return _predict(X)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(batch: BatchRawTransaction, user: dict = Depends(verify_api_key)):
    """Predict fraud for a batch of raw transactions."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    results = []
    for tx in batch.transactions:
        X = raw_to_features(tx)
        results.append(_predict(X, tx.type))
    return BatchPredictionResponse(predictions=results)


@app.post("/feedback")
def submit_feedback(feedback: FeedbackRequest, user: dict = Depends(verify_api_key)):
    """Submit human feedback for a prediction."""
    import json
    from datetime import datetime

    feedback_dir = "/tmp/data/feedback"
    os.makedirs(feedback_dir, exist_ok=True)

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "transaction": feedback.transaction.model_dump(),
        "correct_label": feedback.correct_label,
        "comment": feedback.comment,
        "submitted_by": user["user"],
    }

    with open(os.path.join(feedback_dir, "feedback.jsonl"), "a") as f:
        f.write(json.dumps(entry) + "\n")

    return {"status": "ok", "message": "Feedback recorded"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/model/reload")
def reload_model(user: dict = Depends(verify_api_key)):
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    global MODEL
    try:
        MODEL = mlflow.sklearn.load_model("models:/paysim-fraud-detector/Production")
        return {"status": "ok", "message": "Model reloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}")
