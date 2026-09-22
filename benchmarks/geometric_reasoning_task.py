"""Kaggle Benchmark Task: Synthetic Geometric Decision Boundary Reasoning.

Evaluates whether Large Language Models can perform exact mathematical and
geometric reasoning over 2D decision boundaries (linear hyperplanes, radial
unit circles) and logistic calibration functions.
"""

import kaggle_benchmarks as kbench
from pydantic import BaseModel, Field


class GeometricAuditResult(BaseModel):
    """Structured response schema for geometric decision evaluation."""

    is_unit_circle_interior: bool = Field(
        description="Whether point (0.3, 0.4) lies strictly inside the unit circle r < 1.0 (True/False)."
    )
    hyperplane_classification: str = Field(
        description="Classification for point (0.5, 0.5) under decision rule x1 + x2 > 0 ('positive' or 'negative')."
    )
    neutral_log_odds_probability: float = Field(
        description="Exact calibrated sigmoid probability P(y=1) = 1 / (1 + exp(-z)) when log-odds z = 0.0."
    )


@kbench.task(
    name="synthetic-geometric-decision-boundary-reasoning",
    description=(
        "Evaluates whether an LLM can evaluate 2D points against linear and radial "
        "decision boundaries and compute standard logistic probability calibration."
    ),
    version=1,
)
def synthetic_geometric_decision_boundary_reasoning(llm) -> None:
    """Prompt the LLM with geometric and calibration scenarios and assert correctness."""
    prompt = (
        "Perform exact mathematical reasoning for three synthetic classification boundaries:\n"
        "1. For a radial boundary defined by the unit circle x1^2 + x2^2 = 1.0, does "
        "the point (0.3, 0.4) lie strictly inside the circle (r < 1)?\n"
        "2. For a linear boundary defined by x1 + x2 = 0 where x1 + x2 > 0 is 'positive', "
        "what is the classification of point (0.5, 0.5)?\n"
        "3. For logistic calibration P(y=1) = 1 / (1 + exp(-z)), what is the exact "
        "probability when log-odds z = 0.0?\n"
    )

    result = llm.prompt(prompt, schema=GeometricAuditResult)

    kbench.assertions.assert_true(
        result.is_unit_circle_interior,
        expectation="Point (0.3, 0.4) is strictly interior to the unit circle: 0.3^2 + 0.4^2 = 0.25 < 1.0.",
    )
    kbench.assertions.assert_equal(
        "positive",
        result.hyperplane_classification.lower().strip(),
        expectation="Point (0.5, 0.5) satisfies 0.5 + 0.5 = 1.0 > 0 and must classify as 'positive'.",
    )
    kbench.assertions.assert_equal(
        0.5,
        result.neutral_log_odds_probability,
        expectation="Calibrated sigmoid probability for neutral log-odds z = 0.0 must equal exactly 0.5.",
    )


if __name__ == "__main__":
    synthetic_geometric_decision_boundary_reasoning.run(kbench.llm)
