"""Unit tests for model training."""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture
def sample_data(tmp_path):
    np.random.seed(42)
    cols = ["type_encoded", "amount", "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest", "orig_balance_diff",
            "dest_balance_diff", "orig_balance_error", "dest_balance_error",
            "amount_ratio_orig", "is_orig_empty_after", "is_dest_empty_before",
            "step_hour", "step_day"]

    def make(n):
        X = pd.DataFrame(np.random.randn(n, 15), columns=cols)
        y = pd.DataFrame({"isFraud": np.random.choice([0, 1], n, p=[0.97, 0.03])})
        return X, y

    X_train, y_train = make(500)
    X_test, y_test = make(100)
    X_train.to_parquet(tmp_path / "X_train.parquet")
    X_test.to_parquet(tmp_path / "X_test.parquet")
    y_train.to_parquet(tmp_path / "y_train.parquet")
    y_test.to_parquet(tmp_path / "y_test.parquet")
    return str(tmp_path)


def test_trains_successfully(sample_data):
    X = pd.read_parquet(f"{sample_data}/X_train.parquet")
    y = pd.read_parquet(f"{sample_data}/y_train.parquet").squeeze()
    model = RandomForestClassifier(n_estimators=10, class_weight="balanced", random_state=42)
    model.fit(X, y)
    assert len(model.classes_) == 2


def test_probabilities_valid(sample_data):
    X_train = pd.read_parquet(f"{sample_data}/X_train.parquet")
    y_train = pd.read_parquet(f"{sample_data}/y_train.parquet").squeeze()
    X_test = pd.read_parquet(f"{sample_data}/X_test.parquet")
    model = RandomForestClassifier(n_estimators=10, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_feature_importances(sample_data):
    X = pd.read_parquet(f"{sample_data}/X_train.parquet")
    y = pd.read_parquet(f"{sample_data}/y_train.parquet").squeeze()
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    assert len(model.feature_importances_) == 15
    assert abs(sum(model.feature_importances_) - 1.0) < 0.01
