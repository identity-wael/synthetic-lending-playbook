# Kaggle Benchmarks: Synthetic Geometric Decision Boundary Reasoning

This module provides a Kaggle Benchmark Task that evaluates Large Language Models on exact mathematical and geometric decision boundary reasoning, as well as logistic probability calibration.

## Overview

Large language models frequently struggle with exact coordinate geometry and probability calibration math. This benchmark challenges models on three deterministic synthetic scenarios:

1. **Radial Decision Boundary (Concentric Circles)**: Given a decision circle $x_1^2 + x_2^2 = 1.0$, determine whether point $(0.3, 0.4)$ lies strictly in the interior.
   - Mathematical ground truth: $r^2 = 0.3^2 + 0.4^2 = 0.09 + 0.16 = 0.25 < 1.0 \implies \text{True}$.
2. **Linear Hyperplane Boundary**: Given a linear boundary $x_1 + x_2 = 0$ where $x_1 + x_2 > 0$ denotes the positive class, determine the classification of point $(0.5, 0.5)$.
   - Mathematical ground truth: $0.5 + 0.5 = 1.0 > 0 \implies \text{"positive"}$.
3. **Logistic Probability Calibration**: Given the standard logistic calibration equation $P(y=1) = \frac{1}{1 + e^{-z}}$, determine the exact probability when log-odds $z = 0.0$.
   - Mathematical ground truth: $\sigma(0.0) = \frac{1}{1 + e^0} = \frac{1}{2} = 0.5$.

## Files

- [`geometric_reasoning_task.py`](file:///Users/wael/kaggle/benchmarks/geometric_reasoning_task.py): The benchmark task definition using `@kbench.task` and structured Pydantic output.
- [`test_geometric_reasoning_task.py`](file:///Users/wael/kaggle/benchmarks/test_geometric_reasoning_task.py): Unit tests verifying schema parsing, mathematical calculations, and offline execution with a mock LLM.

## Local Execution & Testing

```bash
# Run unit tests
python -m pytest benchmarks/test_geometric_reasoning_task.py -v

# Run locally with Kaggle Benchmarks (requires MODEL_PROXY_API_KEY in .env)
python benchmarks/geometric_reasoning_task.py
```

## Kaggle CLI Deployment

To push and evaluate this benchmark task on Kaggle:

```bash
kaggle benchmarks tasks push synthetic-geometric-decision-boundary-reasoning -f benchmarks/geometric_reasoning_task.py
```
