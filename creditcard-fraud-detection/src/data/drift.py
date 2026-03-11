"""Data drift detection using Kullback-Leibler (KL) divergence.

How it works
------------
1. **First pipeline run** — reference distributions are computed from the training
   feature set (X_train) and persisted as JSON.  They are NEVER overwritten by
   subsequent runs, so they represent the stable baseline.

2. **Subsequent pipeline runs** — the newly preprocessed X_train distributions are
   computed with the same bin edges as the reference and saved as "current".
   KL(reference || current) is then computed per feature.

3. **Drift flag** — any feature whose KL divergence exceeds `DRIFT_KL_THRESHOLD`
   (default 0.1, overridable via env) is considered drifted.  The overall flag
   `drift_detected` is True when at least one feature drifts.  The full report
   is written to `drift_report.json` alongside the other artefacts.

4. **Airflow integration** — `data_pipeline_dag` calls these helpers after
   preprocessing.  `retrain_dag` reads `drift_report.json` and uses the
   `drift_detected` flag as an additional trigger for retraining.

KL divergence formula
---------------------
    KL(P || Q) = Σ P(x) · log(P(x) / Q(x))

where P = reference distribution, Q = current distribution.
Both histograms are smoothed by epsilon before normalization to guarantee
a finite result even when a bin has zero counts.

JSON artefacts (all stored in the processed data directory)
-----------------------------------------------------------
    reference_distributions.json  — baseline (written once, never overwritten)
    current_distributions.json    — most recent batch distributions
    drift_report.json             — results of the last KL comparison
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (all overridable via environment variables)
# ---------------------------------------------------------------------------

N_BINS: int = 50
DRIFT_THRESHOLD: float = float(os.getenv("DRIFT_KL_THRESHOLD", "0.1"))
EPSILON: float = 1e-10  # smoothing — avoids log(0) when a bin is empty

REFERENCE_FILE = "reference_distributions.json"
CURRENT_FILE = "current_distributions.json"
REPORT_FILE = "drift_report.json"


# ---------------------------------------------------------------------------
# Core maths
# ---------------------------------------------------------------------------

def compute_kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    epsilon: float = EPSILON,
) -> float:
    """Compute KL(P ‖ Q) — P is reference, Q is current.

    Both arrays are smoothed by *epsilon* and renormalized so the result is
    always a finite non-negative float.

    Args:
        p: Reference histogram (raw counts or probabilities).
        q: Current histogram (raw counts or probabilities).
        epsilon: Small constant added before normalization to avoid log(0).

    Returns:
        KL divergence as a float ≥ 0.  A value of 0 means identical
        distributions; the higher the value, the more drift.
    """
    p = np.asarray(p, dtype=float) + epsilon
    q = np.asarray(q, dtype=float) + epsilon
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


# ---------------------------------------------------------------------------
# Histogram helpers
# ---------------------------------------------------------------------------

def _series_to_histogram(
    series: pd.Series,
    bin_edges: np.ndarray | None = None,
    n_bins: int = N_BINS,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a numeric series to a raw-count histogram.

    Args:
        series: Input data (NaN values are dropped).
        bin_edges: Pre-computed edges from the reference.  When provided the
                   same edges are reused so both histograms are comparable.
                   When None, edges are computed from *series* itself.
        n_bins: Number of bins to use when computing edges from scratch.

    Returns:
        (counts, bin_edges)
    """
    data = series.dropna().values.astype(float)
    if bin_edges is None:
        _, bin_edges = np.histogram(data, bins=n_bins)
    counts, _ = np.histogram(data, bins=bin_edges)
    return counts.astype(float), bin_edges


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def save_reference_distributions(
    df: pd.DataFrame,
    output_dir: str,
    n_bins: int = N_BINS,
    overwrite: bool = False,
) -> str:
    """Compute and persist per-feature histograms as the drift reference.

    The reference is written once and never overwritten (unless *overwrite=True*),
    so it always represents the original training-data distribution that the
    model was built on.

    Args:
        df: Feature DataFrame (typically scaled X_train).
        output_dir: Directory where the JSON file will be written.
        n_bins: Number of histogram bins.
        overwrite: Force overwrite even if the reference already exists.

    Returns:
        Absolute path to the written JSON file.
    """
    path = os.path.join(output_dir, REFERENCE_FILE)

    if os.path.exists(path) and not overwrite:
        logger.info(
            "Reference distributions already exist at %s — skipping (use overwrite=True to replace).",
            path,
        )
        return path

    distributions: dict = {}
    for col in df.columns:
        counts, edges = _series_to_histogram(df[col], n_bins=n_bins)
        distributions[col] = {
            "bin_edges": edges.tolist(),
            "hist": counts.tolist(),
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
        }

    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(distributions, fh)

    logger.info(
        "Saved reference distributions for %d features → %s", len(distributions), path
    )
    return path


def save_current_distributions(
    df: pd.DataFrame,
    output_dir: str,
    n_bins: int = N_BINS,
) -> str:
    """Compute per-feature histograms for the current data.

    If a reference file exists the same bin edges are reused so that the
    resulting histograms are directly comparable for KL computation.
    If no reference exists yet, edges are derived from *df* itself.

    Args:
        df: Feature DataFrame (typically the freshly preprocessed X_train).
        output_dir: Directory containing (optionally) the reference JSON and
                    where the current JSON will be written.

    Returns:
        Absolute path to the written JSON file.
    """
    ref_path = os.path.join(output_dir, REFERENCE_FILE)
    reference: dict = {}
    if os.path.exists(ref_path):
        with open(ref_path) as fh:
            reference = json.load(fh)

    distributions: dict = {}
    for col in df.columns:
        ref_edges = (
            np.array(reference[col]["bin_edges"]) if col in reference else None
        )
        counts, edges = _series_to_histogram(df[col], bin_edges=ref_edges, n_bins=n_bins)
        distributions[col] = {
            "bin_edges": edges.tolist(),
            "hist": counts.tolist(),
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
        }

    path = os.path.join(output_dir, CURRENT_FILE)
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(distributions, fh)

    return path


# ---------------------------------------------------------------------------
# Drift report
# ---------------------------------------------------------------------------

def compute_drift_report(
    output_dir: str,
    threshold: float | None = None,
) -> dict:
    """Compare the saved current distributions against the reference.

    Reads ``reference_distributions.json`` and ``current_distributions.json``
    from *output_dir*, computes KL(reference ‖ current) for every common
    feature, and writes a ``drift_report.json`` with the full results.

    Args:
        output_dir: Directory containing both JSON distribution files.
        threshold: KL threshold — features above it are flagged as drifted.
                   Defaults to the ``DRIFT_KL_THRESHOLD`` env var (or 0.1).

    Returns:
        Report dict::

            {
                "drift_detected": bool,
                "drifted_features": ["amount", ...],
                "kl_divergences": {"amount": 0.23, "type_encoded": 0.01, ...},
                "threshold": 0.1,
                "n_features_checked": 15,
            }

    Raises:
        FileNotFoundError: If either JSON file is missing.
    """
    if threshold is None:
        threshold = DRIFT_THRESHOLD

    ref_path = os.path.join(output_dir, REFERENCE_FILE)
    cur_path = os.path.join(output_dir, CURRENT_FILE)

    if not os.path.exists(ref_path):
        raise FileNotFoundError(
            f"Reference distributions not found at {ref_path}. "
            "Run the data_pipeline DAG at least once to create them."
        )
    if not os.path.exists(cur_path):
        raise FileNotFoundError(
            f"Current distributions not found at {cur_path}. "
            "Run save_current_distributions() before calling this function."
        )

    with open(ref_path) as fh:
        reference = json.load(fh)
    with open(cur_path) as fh:
        current = json.load(fh)

    kl_scores: dict[str, float] = {}
    for feat in reference:
        if feat not in current:
            logger.warning("Feature '%s' missing from current distributions — skipped.", feat)
            continue
        p = np.array(reference[feat]["hist"])
        q = np.array(current[feat]["hist"])
        kl_scores[feat] = compute_kl_divergence(p, q)

    drifted = sorted(f for f, kl in kl_scores.items() if kl > threshold)
    drift_detected = len(drifted) > 0
    max_kl = max(kl_scores.values(), default=0.0)

    if drift_detected:
        logger.warning(
            "DATA DRIFT DETECTED — %d/%d features drifted (max KL=%.4f, threshold=%.4f): %s",
            len(drifted),
            len(kl_scores),
            max_kl,
            threshold,
            drifted,
        )
    else:
        logger.info(
            "No drift detected — max KL=%.4f (threshold=%.4f).",
            max_kl,
            threshold,
        )

    report = {
        "drift_detected": drift_detected,
        "drifted_features": drifted,
        "kl_divergences": kl_scores,
        "threshold": threshold,
        "n_features_checked": len(kl_scores),
        "max_kl": max_kl,
    }

    report_path = os.path.join(output_dir, REPORT_FILE)
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)

    logger.info("Drift report written to %s", report_path)
    return report


# ---------------------------------------------------------------------------
# Convenience entry-point (used by Airflow tasks)
# ---------------------------------------------------------------------------

def run_drift_detection(
    processed_dir: str,
    current_df: pd.DataFrame | None = None,
    threshold: float | None = None,
) -> dict:
    """Full drift detection pipeline in a single call.

    1. If *current_df* is provided, saves its distributions (using reference
       bin edges when available).
    2. If no reference file exists yet, the current distributions ARE saved as
       the new reference (bootstrap mode — first run).
    3. Otherwise, computes and returns the KL drift report.

    Args:
        processed_dir: Directory with artefacts (parquet, joblibs, JSON).
        current_df: Freshly preprocessed feature DataFrame.  When None,
                    existing ``current_distributions.json`` is used directly.
        threshold: KL threshold (defaults to env var DRIFT_KL_THRESHOLD).

    Returns:
        Drift report dict (see ``compute_drift_report``).
    """
    ref_path = os.path.join(processed_dir, REFERENCE_FILE)
    first_run = not os.path.exists(ref_path)

    if current_df is not None:
        if first_run:
            # Bootstrap: persist as reference
            save_reference_distributions(current_df, processed_dir)
            logger.info(
                "First pipeline run — reference distributions saved. "
                "No drift comparison performed yet."
            )
            return {
                "drift_detected": False,
                "drifted_features": [],
                "kl_divergences": {},
                "threshold": threshold or DRIFT_THRESHOLD,
                "n_features_checked": 0,
                "note": "first_run_reference_saved",
            }
        else:
            # Save current distributions reusing reference bin edges
            save_current_distributions(current_df, processed_dir)

    if first_run:
        logger.info("No reference distributions — cannot compute drift yet.")
        return {
            "drift_detected": False,
            "drifted_features": [],
            "kl_divergences": {},
            "threshold": threshold or DRIFT_THRESHOLD,
            "n_features_checked": 0,
            "note": "first_run_no_data",
        }

    return compute_drift_report(processed_dir, threshold=threshold)
