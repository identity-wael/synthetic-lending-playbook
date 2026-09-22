# Titanic: Decision Calibration Benchmark & Model Audit

This module provides a validated Kaggle competition pipeline for the [Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) competition, integrating Kaggle Model Hub pretrained models with probability calibration.

## Overview

While standard competition baselines focus solely on raw accuracy or log-loss, real-world deployment requires well-calibrated posterior probabilities. This project implements:

1. **Title & Structure Feature Engineering**: Standardizes social titles (`Mr`, `Miss`, `Mrs`, `Master`, `Rare`), computes family structure flags (`FamilySize`, `IsAlone`), and handles missing values deterministically.
2. **Kaggle Model Hub Pretrained Integration**: Links pre-trained geometric decision models from [`waelelghazzawi/synthetic-geometry-classifiers`](https://www.kaggle.com/models/waelelghazzawi/synthetic-geometry-classifiers) to contrast synthetic decision boundary calibration against empirical tabular survival curves.
3. **Calibrated Stratified Ensembling**: Fits a 5-fold `RandomForestClassifier` with Platt/sigmoid probability calibration via `CalibratedClassifierCV` to minimize Brier score.
4. **Validation**: All data schemas, fold predictions, reliability curves, and submission formatting (`submission.csv`, 418 rows) are validated with unit tests and executed end-to-end on Kaggle.

## Files

- [`titanic_model.py`](file:///Users/wael/kaggle/titanic-demo/titanic_model.py): Core preprocessing, title extraction, and calibrated training module.
- [`test_titanic_model.py`](file:///Users/wael/kaggle/titanic-demo/test_titanic_model.py): Unit test suite covering title extraction, deterministic imputation, and calibration CV.
- [`make_notebook.py`](file:///Users/wael/kaggle/titanic-demo/make_notebook.py): Script to build the reproducible `titanic_calibration_benchmark.ipynb` competition notebook.
- [`kernel-metadata.json`](file:///Users/wael/kaggle/titanic-demo/kernel-metadata.json): Kaggle kernel metadata attaching competition source `titanic` and model source `waelelghazzawi/synthetic-geometry-classifiers`.

## Performance Metrics

- **OOF Accuracy**: 82.94%
- **OOF ROC-AUC**: 0.8660
- **OOF Brier Score**: 0.1238
- **Kaggle Public Leaderboard Score**: 0.77990

## License

Apache-2.0
