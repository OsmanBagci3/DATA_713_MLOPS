"""PaySim preprocessing and feature engineering.

Outputs are designed to be consumed by:
- Airflow DAGs writing to `/tmp/data/processed`
- Training code in `src/train/train.py`
- Unit tests in `tests/unit/test_preprocess.py`

The engineered feature set matches the API's `raw_to_features()` logic.
"""

from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TYPE_MAPPING: dict[str, int] = {
    "CASH_IN": 0,
    "CASH_OUT": 1,
    "DEBIT": 2,
    "PAYMENT": 3,
    "TRANSFER": 4,
}

FEATURE_COLUMNS = [
    "type_encoded",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "orig_balance_diff",
    "dest_balance_diff",
    "orig_balance_error",
    "dest_balance_error",
    "amount_ratio_orig",
    "is_orig_empty_after",
    "is_dest_empty_before",
    "step_hour",
    "step_day",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 15 engineered features expected by training and serving."""
    required = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for feature engineering: {missing}")

    out = df.copy()

    out["type_encoded"] = out["type"].map(TYPE_MAPPING).fillna(TYPE_MAPPING["TRANSFER"]).astype(int)

    out["orig_balance_diff"] = out["oldbalanceOrg"] - out["newbalanceOrig"]
    out["dest_balance_diff"] = out["newbalanceDest"] - out["oldbalanceDest"]

    out["orig_balance_error"] = out["orig_balance_diff"] - out["amount"]
    out["dest_balance_error"] = out["dest_balance_diff"] - out["amount"]

    out["amount_ratio_orig"] = np.where(out["oldbalanceOrg"] > 0, out["amount"] / out["oldbalanceOrg"], 0.0)

    out["is_orig_empty_after"] = (out["newbalanceOrig"] == 0).astype(int)
    out["is_dest_empty_before"] = (out["oldbalanceDest"] == 0).astype(int)

    out["step_hour"] = (out["step"] % 24).astype(int)
    out["step_day"] = (out["step"] // 24).astype(int)

    # Ensure numeric dtypes (scaler needs floats)
    features = out[FEATURE_COLUMNS].copy()
    for col in FEATURE_COLUMNS:
        features[col] = pd.to_numeric(features[col], errors="coerce")

    return features


def load_and_preprocess(
    input_path: str,
    output_dir: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Load PaySim CSV, engineer features, split, scale, and save parquet artifacts."""
    df = pd.read_csv(input_path)

    # --- DEV MODE: reduce dataset for fast DevOps testing ---
    sample_size = int(os.getenv("DEV_SAMPLE_SIZE", "0"))
    if sample_size > 0 and len(df) > sample_size:
        # Stratified sample to keep fraud ratio representative
        fraud = df[df["isFraud"] == 1]
        legit = df[df["isFraud"] == 0]
        n_fraud = max(1, int(sample_size * len(fraud) / len(df)))
        n_legit = sample_size - n_fraud
        df = pd.concat([
            fraud.sample(n=min(n_fraud, len(fraud)), random_state=42),
            legit.sample(n=min(n_legit, len(legit)), random_state=42),
        ]).reset_index(drop=True)
        print(f"[DEV] Sampled to {len(df)} rows (fraud={df['isFraud'].sum()})")

    if "isFraud" not in df.columns:
        raise ValueError("Input CSV must contain 'isFraud' column")

    y = df["isFraud"].astype(int)
    X = engineer_features(df)

    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURE_COLUMNS, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=FEATURE_COLUMNS, index=X_test.index)

    os.makedirs(output_dir, exist_ok=True)

    X_train_scaled.to_parquet(os.path.join(output_dir, "X_train.parquet"))
    X_test_scaled.to_parquet(os.path.join(output_dir, "X_test.parquet"))
    pd.DataFrame({"isFraud": y_train}).to_parquet(os.path.join(output_dir, "y_train.parquet"))
    pd.DataFrame({"isFraud": y_test}).to_parquet(os.path.join(output_dir, "y_test.parquet"))

    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
    joblib.dump(TYPE_MAPPING, os.path.join(output_dir, "type_mapping.joblib"))

    fraud_ratio = float(y.mean())
    fraud_by_type = {}
    if "type" in df.columns:
        fraud_by_type = (
            df.groupby("type")["isFraud"].sum().sort_values(ascending=False).to_dict()
        )

    stats = {
        "total_samples": int(len(df)),
        "train_size": int(len(X_train_scaled)),
        "test_size": int(len(X_test_scaled)),
        "n_features": int(X_train_scaled.shape[1]),
        "fraud_ratio": fraud_ratio,
        "fraud_by_type": fraud_by_type,
    }
    return stats


def simulate_live_data(raw_csv_path: str, batch_size: int = 50, random_state: int | None = None) -> pd.DataFrame:
    """Sample a small batch of raw transactions from the PaySim CSV."""
    df = pd.read_csv(raw_csv_path)
    n = min(batch_size, len(df))
    batch = df.sample(n=n, random_state=random_state)

    # Keep only the fields expected by the live simulation DAG.
    cols = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]
    if "isFraud" in batch.columns:
        cols.append("isFraud")

    return batch[cols].reset_index(drop=True)
