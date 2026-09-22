"""
Comprehensive pytest test suite for S6E9 EV purchase prediction pipeline.
"""

import numpy as np
import pandas as pd
import pytest

from ev_features import (
    extract_domain_features,
    compute_deotte_dgp_score,
    apply_deterministic_boundaries,
    zero_tie_lexicographic_ranking,
)
from ev_pipeline import build_ensemble_submission, validate_submission_format


@pytest.fixture
def sample_df():
    """Create a synthetic mini-batch resembling S6E9 schema."""
    return pd.DataFrame({
        "id": [101, 102, 103, 104, 105],
        "Age": [35, 45, 60, 25, 50],
        "Annual_Income_USD": [175000.0, 35000.0, 30000.0, 85000.0, 95000.0],
        "Daily_Commute_km": [20.0, 45.0, 90.0, 15.0, 30.0],
        "Number_of_Cars_Owned": [1, 2, 3, 1, 2],
        "Charging_Stations_Near_Home": [4, 2, 1, 6, 3],
        "Charging_Stations_Near_Work": [8, 5, 2, 10, 7],
        "Environmental_Concern_Level": [5, 2, 1, 4, 3],
        "Gender": ["Male", "Female", "Other", "Female", "Male"],
        "City_Type": ["Urban", "Suburban", "Rural", "Urban", "Suburban"],
        "Current_Car_Type": ["Sedan", "SUV", "Truck", "Hatchback", "Sedan"],
        "Home_Charging_Possible": ["Yes", "No", "No", "Yes", "Yes"],
        "Subsidy_Available": ["Yes", "Yes", "No", "Yes", "No"],
        "Range_Anxiety_Level": ["Low", "High", "High", "Low", "Medium"],
    })


def test_extract_domain_features(sample_df):
    fe_df = extract_domain_features(sample_df)
    assert "Total_Charging_Stations" in fe_df.columns
    assert "Commute_Per_Station" in fe_df.columns
    assert "EV_Readiness_Score" in fe_df.columns
    assert "Incentives_Index" in fe_df.columns

    # Row 0: 4 home + 8 work = 12 total chargers
    assert fe_df.loc[0, "Total_Charging_Stations"] == 12.0
    # Row 0: Env Concern 5 * (4 - 1) = 15.0
    assert fe_df.loc[0, "EV_Readiness_Score"] == 15.0
    assert not fe_df.isnull().any().any()


def test_compute_deotte_dgp_score(sample_df):
    scores = compute_deotte_dgp_score(sample_df)
    assert len(scores) == len(sample_df)
    assert np.all(np.isfinite(scores))
    # Row 0 (High income, high env, subsidy yes, anxiety low) should have high buy score
    assert scores[0] > scores[1]
    # Row 2 (Low income, low env, subsidy no, anxiety high) should have low score
    assert scores[2] < scores[0]


def test_apply_deterministic_boundaries(sample_df):
    base_scores = np.zeros(len(sample_df), dtype=float)
    calibrated = apply_deterministic_boundaries(sample_df, base_scores)

    # Row 0: Income 175000 >= 170537 -> +1000.0 shift
    assert calibrated[0] == 1000.0

    # Row 1: Income 35000 in [31004, 41970] -> -1000.0 shift
    assert calibrated[1] == -1000.0

    # Row 2: Commute 90 >= 83 and 30k trap -> -500.0 - 500.0 = -1000.0
    assert calibrated[2] == -1000.0

    # Row 3: Standard normal case -> 0.0 shift
    assert calibrated[3] == 0.0


def test_zero_tie_lexicographic_ranking():
    ids = np.array([1, 2, 3, 4])
    scores = np.array([0.5, 0.5, 0.9, 0.1])  # Has tie at 0.5

    ranks = zero_tie_lexicographic_ranking(ids, scores)
    assert len(ranks) == 4
    assert len(np.unique(ranks)) == 4  # Ties strictly broken
    assert np.all((ranks >= 0.0) & (ranks <= 1.0))
    # Score 0.1 must be lowest rank
    assert ranks[3] == min(ranks)
    # Score 0.9 must be highest rank
    assert ranks[2] == max(ranks)


def test_validate_submission_format():
    good_sub = pd.DataFrame({
        "id": [1, 2, 3],
        "Will_Buy_EV": [0.1, 0.5, 0.9],
    })
    # Should pass without error
    validate_submission_format(good_sub, expected_rows=3)

    # Should raise error on row count mismatch
    with pytest.raises(AssertionError):
        validate_submission_format(good_sub, expected_rows=5)

    # Should raise error on ties
    tied_sub = pd.DataFrame({
        "id": [1, 2, 3],
        "Will_Buy_EV": [0.5, 0.5, 0.9],
    })
    with pytest.raises(AssertionError):
        validate_submission_format(tied_sub, expected_rows=3)
