"""Unit tests for KL-divergence data drift detection."""

import json
import os

import numpy as np
import pandas as pd
import pytest

from src.data.drift import (
    compute_drift_report,
    compute_kl_divergence,
    run_drift_detection,
    save_current_distributions,
    save_reference_distributions,
    DRIFT_THRESHOLD,
    REFERENCE_FILE,
    CURRENT_FILE,
    REPORT_FILE,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

FEATURES = ["amount", "type_encoded", "orig_balance_diff", "amount_ratio_orig"]


def _make_df(n: int = 500, scale: float = 1.0, shift: float = 0.0) -> pd.DataFrame:
    """Build a small feature DataFrame for testing."""
    return pd.DataFrame(
        {
            "amount": RNG.exponential(5000 * scale, n) + shift,
            "type_encoded": RNG.integers(0, 5, n).astype(float),
            "orig_balance_diff": RNG.normal(0, 1000 * scale, n),
            "amount_ratio_orig": RNG.uniform(0, 1, n),
        }
    )


@pytest.fixture
def reference_df():
    return _make_df(n=500)


@pytest.fixture
def similar_df(reference_df):
    """Exact copy of the reference — KL divergence should be 0 (no drift)."""
    return reference_df.copy()


@pytest.fixture
def drifted_df():
    """Heavily shifted data — should trigger drift."""
    return _make_df(n=500, scale=10.0, shift=50000)


# ---------------------------------------------------------------------------
# 1. KL divergence maths
# ---------------------------------------------------------------------------


class TestKLDivergence:
    def test_identical_distributions_zero(self):
        hist = np.array([10.0, 20.0, 30.0, 20.0, 10.0])
        assert compute_kl_divergence(hist, hist) == pytest.approx(0.0, abs=1e-6)

    def test_kl_is_non_negative(self):
        p = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        q = np.array([3.0, 1.0, 2.0, 4.0, 1.0])
        assert compute_kl_divergence(p, q) >= 0.0

    def test_kl_asymmetric(self):
        """KL(P‖Q) ≠ KL(Q‖P) in general."""
        p = np.array([1.0, 9.0])
        q = np.array([5.0, 5.0])
        assert compute_kl_divergence(p, q) != pytest.approx(
            compute_kl_divergence(q, p), abs=1e-6
        )

    def test_kl_increases_with_divergence(self):
        """More different distributions → higher KL."""
        base = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        small_shift = np.array([1.0, 2.0, 3.0, 3.0, 1.0])
        large_shift = np.array([3.0, 3.0, 1.0, 1.0, 1.0])
        kl_small = compute_kl_divergence(base, small_shift)
        kl_large = compute_kl_divergence(base, large_shift)
        assert kl_large > kl_small

    def test_zero_bins_handled(self):
        """Epsilon smoothing must prevent division-by-zero."""
        p = np.array([10.0, 0.0, 5.0])
        q = np.array([0.0, 10.0, 5.0])
        result = compute_kl_divergence(p, q)
        assert np.isfinite(result)


# ---------------------------------------------------------------------------
# 2. Save / load reference
# ---------------------------------------------------------------------------


class TestSaveReferenceDistributions:
    def test_creates_json_file(self, tmp_path, reference_df):
        path = save_reference_distributions(reference_df, str(tmp_path))
        assert os.path.exists(path)
        assert path.endswith(REFERENCE_FILE)

    def test_json_contains_all_features(self, tmp_path, reference_df):
        save_reference_distributions(reference_df, str(tmp_path))
        with open(os.path.join(tmp_path, REFERENCE_FILE)) as f:
            data = json.load(f)
        for col in reference_df.columns:
            assert col in data
            assert "bin_edges" in data[col]
            assert "hist" in data[col]
            assert "mean" in data[col]
            assert "std" in data[col]

    def test_not_overwritten_by_default(self, tmp_path, reference_df, similar_df):
        save_reference_distributions(reference_df, str(tmp_path))
        original_mean = json.load(open(os.path.join(tmp_path, REFERENCE_FILE)))["amount"]["mean"]

        save_reference_distributions(similar_df, str(tmp_path))  # should be ignored
        new_mean = json.load(open(os.path.join(tmp_path, REFERENCE_FILE)))["amount"]["mean"]

        assert original_mean == new_mean

    def test_overwrite_flag_replaces_reference(self, tmp_path, reference_df, drifted_df):
        save_reference_distributions(reference_df, str(tmp_path))
        old_mean = json.load(open(os.path.join(tmp_path, REFERENCE_FILE)))["amount"]["mean"]

        save_reference_distributions(drifted_df, str(tmp_path), overwrite=True)
        new_mean = json.load(open(os.path.join(tmp_path, REFERENCE_FILE)))["amount"]["mean"]

        assert old_mean != new_mean


# ---------------------------------------------------------------------------
# 3. Current distributions reuse reference bin edges
# ---------------------------------------------------------------------------


class TestSaveCurrentDistributions:
    def test_creates_json_file(self, tmp_path, reference_df, similar_df):
        save_reference_distributions(reference_df, str(tmp_path))
        path = save_current_distributions(similar_df, str(tmp_path))
        assert os.path.exists(path)

    def test_same_bin_edges_as_reference(self, tmp_path, reference_df, similar_df):
        save_reference_distributions(reference_df, str(tmp_path))
        save_current_distributions(similar_df, str(tmp_path))

        with open(os.path.join(tmp_path, REFERENCE_FILE)) as f:
            ref = json.load(f)
        with open(os.path.join(tmp_path, CURRENT_FILE)) as f:
            cur = json.load(f)

        for col in ref:
            assert ref[col]["bin_edges"] == cur[col]["bin_edges"], (
                f"Bin edges differ for feature '{col}' — KL comparison would be invalid"
            )


# ---------------------------------------------------------------------------
# 4. Drift report
# ---------------------------------------------------------------------------


class TestComputeDriftReport:
    def test_no_drift_similar_data(self, tmp_path, reference_df, similar_df):
        save_reference_distributions(reference_df, str(tmp_path))
        save_current_distributions(similar_df, str(tmp_path))
        report = compute_drift_report(str(tmp_path))

        assert not report["drift_detected"]
        assert report["drifted_features"] == []
        assert report["n_features_checked"] == len(reference_df.columns)

    def test_drift_detected_on_shifted_data(self, tmp_path, reference_df, drifted_df):
        save_reference_distributions(reference_df, str(tmp_path))
        save_current_distributions(drifted_df, str(tmp_path))
        report = compute_drift_report(str(tmp_path))

        assert report["drift_detected"]
        assert len(report["drifted_features"]) >= 1
        assert report["max_kl"] > DRIFT_THRESHOLD

    def test_report_json_written(self, tmp_path, reference_df, drifted_df):
        save_reference_distributions(reference_df, str(tmp_path))
        save_current_distributions(drifted_df, str(tmp_path))
        compute_drift_report(str(tmp_path))

        report_path = os.path.join(tmp_path, REPORT_FILE)
        assert os.path.exists(report_path)
        data = json.load(open(report_path))
        assert "drift_detected" in data
        assert "kl_divergences" in data

    def test_custom_threshold(self, tmp_path, reference_df):
        """A low threshold flags moderate differences that the default threshold ignores."""
        rng = np.random.default_rng(7)
        # Build a slightly shifted continuous-only DataFrame — no integer columns
        slightly_shifted = pd.DataFrame({
            "amount": rng.exponential(5500, 500),          # ~10 % scale change
            "type_encoded": rng.integers(0, 5, 500).astype(float),
            "orig_balance_diff": rng.normal(200, 1000, 500),  # shifted mean
            "amount_ratio_orig": rng.uniform(0.05, 1.05, 500),
        })
        save_reference_distributions(reference_df, str(tmp_path))
        save_current_distributions(slightly_shifted, str(tmp_path))
        # Very low threshold must flag the moderate shift
        report = compute_drift_report(str(tmp_path), threshold=1e-6)
        assert report["drift_detected"]

    def test_missing_reference_raises(self, tmp_path, reference_df, similar_df):
        save_current_distributions(similar_df, str(tmp_path))
        with pytest.raises(FileNotFoundError, match="Reference distributions not found"):
            compute_drift_report(str(tmp_path))

    def test_missing_current_raises(self, tmp_path, reference_df):
        save_reference_distributions(reference_df, str(tmp_path))
        with pytest.raises(FileNotFoundError, match="Current distributions not found"):
            compute_drift_report(str(tmp_path))


# ---------------------------------------------------------------------------
# 5. run_drift_detection — end-to-end entry point
# ---------------------------------------------------------------------------


class TestRunDriftDetection:
    def test_first_run_bootstraps_reference(self, tmp_path, reference_df):
        report = run_drift_detection(str(tmp_path), current_df=reference_df)

        assert report["note"] == "first_run_reference_saved"
        assert not report["drift_detected"]
        assert os.path.exists(os.path.join(tmp_path, REFERENCE_FILE))

    def test_second_run_returns_report(self, tmp_path, reference_df, similar_df):
        run_drift_detection(str(tmp_path), current_df=reference_df)  # bootstrap
        report = run_drift_detection(str(tmp_path), current_df=similar_df)

        assert "drift_detected" in report
        assert "kl_divergences" in report
        assert report["n_features_checked"] == len(reference_df.columns)

    def test_drift_triggers_true_flag(self, tmp_path, reference_df, drifted_df):
        run_drift_detection(str(tmp_path), current_df=reference_df)  # bootstrap
        report = run_drift_detection(str(tmp_path), current_df=drifted_df)

        assert report["drift_detected"] is True
        assert len(report["drifted_features"]) >= 1
