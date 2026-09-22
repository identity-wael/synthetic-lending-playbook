"""
End-to-End Ensemble Pipeline for Kaggle Playground Series S6E9:
Predicting Electric Vehicle Purchases.

Combines:
1. Multi-scale engineered feature transformations
2. High-precision candidate models (OOF & Test rank-space fusion)
3. Deterministic boundary physics corrections
4. Strict zero-tie percentile rank generation
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ev_features import (
    extract_domain_features,
    compute_deotte_dgp_score,
    apply_deterministic_boundaries,
    zero_tie_lexicographic_ranking,
)


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, test, and sample_submission datasets."""
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    sample_path = data_dir / "sample_submission.csv"

    if not train_path.exists() or not test_path.exists() or not sample_path.exists():
        raise FileNotFoundError(f"Missing required competition files in {data_dir}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)
    return train, test, sample


def build_ensemble_submission(
    test_df: pd.DataFrame,
    candidate_predictions: list[np.ndarray],
    weights: list[float] | None = None,
) -> pd.DataFrame:
    """Combine candidate predictions in rank space, apply boundary physics, and zero-tie sort."""
    n = len(test_df)
    if weights is None:
        weights = [1.0 / len(candidate_predictions)] * len(candidate_predictions)
    weights = np.array(weights) / np.sum(weights)

    # Rank-space normalization of each candidate
    normalized_ranks = np.zeros(n, dtype=np.float64)
    for pred, w in zip(candidate_predictions, weights):
        r = (rankdata(pred, method="ordinal") - 0.5) / n
        normalized_ranks += w * r

    # Compute analytical DGP score to guide tie-breaking in mid-quantiles
    dgp_logits = compute_deotte_dgp_score(test_df)
    composite_base = normalized_ranks + 1e-6 * dgp_logits

    # Apply deterministic empirical boundary shifts (+10 / -10 / -5)
    calibrated_scores = apply_deterministic_boundaries(test_df, composite_base)

    # Final zero-tie lexicographical ordering
    final_ranks = zero_tie_lexicographic_ranking(test_df["id"].values, calibrated_scores)

    sub = pd.DataFrame({
        "id": test_df["id"].values,
        "Will_Buy_EV": final_ranks,
    })
    return sub


def validate_submission_format(sub: pd.DataFrame, expected_rows: int = 286571) -> None:
    """Strict assertion suite verifying competition requirements."""
    assert len(sub) == expected_rows, f"Expected {expected_rows} rows, got {len(sub)}"
    assert list(sub.columns) == ["id", "Will_Buy_EV"], f"Invalid columns: {sub.columns}"
    assert not sub["id"].isnull().any(), "Found null values in id column"
    assert not sub["Will_Buy_EV"].isnull().any(), "Found null values in Will_Buy_EV column"
    assert sub["Will_Buy_EV"].min() >= 0.0, f"Negative probability found: {sub['Will_Buy_EV'].min()}"
    assert sub["Will_Buy_EV"].max() <= 1.0, f"Probability > 1.0 found: {sub['Will_Buy_EV'].max()}"
    assert sub["Will_Buy_EV"].nunique() == expected_rows, "Ties found in prediction ranks"
