"""Unit tests for the synthetic geometric reasoning benchmark task."""

import math
import pytest
from pydantic import ValidationError

from benchmarks.geometric_reasoning_task import (
    GeometricAuditResult,
    synthetic_geometric_decision_boundary_reasoning,
)


class MockLLM:
    """Mock LLM to test benchmark task assertion logic offline."""

    def __init__(self, is_interior: bool = True, classification: str = "positive", prob: float = 0.5):
        self.is_interior = is_interior
        self.classification = classification
        self.prob = prob

    def prompt(self, prompt: str, schema=None):
        return schema(
            is_unit_circle_interior=self.is_interior,
            hyperplane_classification=self.classification,
            neutral_log_odds_probability=self.prob,
        )


def test_geometric_audit_schema_valid():
    """Verify GeometricAuditResult parses valid typed attributes."""
    res = GeometricAuditResult(
        is_unit_circle_interior=True,
        hyperplane_classification="positive",
        neutral_log_odds_probability=0.5,
    )
    assert res.is_unit_circle_interior is True
    assert res.hyperplane_classification == "positive"
    assert res.neutral_log_odds_probability == 0.5


def test_geometric_audit_schema_invalid():
    """Verify GeometricAuditResult rejects non-convertible types."""
    with pytest.raises(ValidationError):
        GeometricAuditResult(
            is_unit_circle_interior="not-a-bool",
            hyperplane_classification=12345,
            neutral_log_odds_probability="invalid-float",
        )


def test_ground_truth_geometry_math():
    """Verify the exact mathematical ground truths behind the benchmark expectations."""
    # Radial boundary: (0.3, 0.4) distance squared to origin
    x1, x2 = 0.3, 0.4
    r_sq = x1**2 + x2**2
    assert math.isclose(r_sq, 0.25)
    assert r_sq < 1.0  # strictly interior to unit circle

    # Linear hyperplane: (0.5, 0.5) against hyperplane x1 + x2 > 0
    val = 0.5 + 0.5
    assert val > 0.0  # strictly positive

    # Logistic calibration: sigmoid(0.0)
    sigmoid_0 = 1.0 / (1.0 + math.exp(-0.0))
    assert math.isclose(sigmoid_0, 0.5)


def test_benchmark_task_with_mock_llm():
    """Verify that synthetic_geometric_decision_boundary_reasoning succeeds with conforming mock outputs."""
    mock = MockLLM(is_interior=True, classification="positive", prob=0.5)
    # Executing the function directly with mock LLM should pass without throwing AssertionError
    synthetic_geometric_decision_boundary_reasoning.func(mock)
