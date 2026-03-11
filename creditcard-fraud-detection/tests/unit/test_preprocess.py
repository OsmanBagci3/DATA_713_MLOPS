"""Unit tests for PaySim preprocessing."""
import os
import pandas as pd
import numpy as np
import pytest
from src.data.preprocess import load_and_preprocess, engineer_features, TYPE_MAPPING


@pytest.fixture
def sample_csv(tmp_path):
    np.random.seed(42)
    n = 1000
    types = np.random.choice(["CASH_IN", "CASH_OUT", "TRANSFER", "PAYMENT", "DEBIT"], n, p=[0.2, 0.3, 0.1, 0.3, 0.1])
    amounts = np.random.exponential(5000, n)
    old_orig = np.random.exponential(10000, n)
    new_orig = np.maximum(0, old_orig - amounts)
    old_dest = np.random.exponential(5000, n)
    new_dest = old_dest + amounts
    fraud = np.zeros(n, dtype=int)
    # Make some TRANSFER/CASH_OUT fraudulent
    transfer_idx = np.where((types == "TRANSFER") | (types == "CASH_OUT"))[0][:20]
    fraud[transfer_idx] = 1

    df = pd.DataFrame({
        "step": np.random.randint(1, 744, n),
        "type": types,
        "amount": amounts,
        "nameOrig": [f"C{i}" for i in range(n)],
        "oldbalanceOrg": old_orig,
        "newbalanceOrig": new_orig,
        "nameDest": [f"C{i+n}" for i in range(n)],
        "oldbalanceDest": old_dest,
        "newbalanceDest": new_dest,
        "isFraud": fraud,
        "isFlaggedFraud": 0,
    })
    path = tmp_path / "paysim_test.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_creates_output_files(sample_csv, tmp_path):
    out = str(tmp_path / "processed")
    load_and_preprocess(sample_csv, out)
    assert os.path.exists(os.path.join(out, "X_train.parquet"))
    assert os.path.exists(os.path.join(out, "X_test.parquet"))
    assert os.path.exists(os.path.join(out, "scaler.joblib"))
    assert os.path.exists(os.path.join(out, "type_mapping.joblib"))


def test_returns_stats(sample_csv, tmp_path):
    stats = load_and_preprocess(sample_csv, str(tmp_path / "p"))
    assert "total_samples" in stats
    assert "fraud_by_type" in stats
    assert stats["n_features"] == 15
    assert stats["train_size"] + stats["test_size"] == stats["total_samples"]


def test_feature_engineering():
    df = pd.DataFrame({
        "step": [1], "type": ["TRANSFER"], "amount": [10000],
        "oldbalanceOrg": [10000], "newbalanceOrig": [0],
        "oldbalanceDest": [0], "newbalanceDest": [10000],
    })
    df = engineer_features(df)
    assert df["type_encoded"].iloc[0] == TYPE_MAPPING["TRANSFER"]
    assert df["orig_balance_diff"].iloc[0] == 10000
    assert df["is_orig_empty_after"].iloc[0] == 1
    assert df["amount_ratio_orig"].iloc[0] == 1.0


def test_no_nulls(sample_csv, tmp_path):
    out = str(tmp_path / "p")
    load_and_preprocess(sample_csv, out)
    X = pd.read_parquet(os.path.join(out, "X_train.parquet"))
    assert not X.isnull().any().any()


def test_stratified_split(sample_csv, tmp_path):
    out = str(tmp_path / "p")
    load_and_preprocess(sample_csv, out)
    y_train = pd.read_parquet(os.path.join(out, "y_train.parquet")).squeeze()
    y_test = pd.read_parquet(os.path.join(out, "y_test.parquet")).squeeze()
    assert abs(y_train.mean() - y_test.mean()) < 0.05
