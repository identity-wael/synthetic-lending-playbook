# Synthetic Geometry Model Export Demo

Separate educational example demonstrating feature transforms and JSON model exports. All files in this directory are Apache-2.0 licensed. Training and evaluation use generated coordinates, with no lending or personal data.

Run train.py with NumPy and scikit-learn. See model_exports for metrics, explicit feature orders and test vectors. The Kaggle training notebook passed all export checks.

Inference and calibration audits:
- `inference.py` / `inference.ipynb`: Verifies test vectors and reproduces held-out accuracy, log-loss, and ROC-AUC.
- `calibration_audit.py` / `calibration_audit.ipynb`: Computes Expected Calibration Error (ECE), Brier score, radial margin confidence profiles, and out-of-distribution (OOD) domain shift behavior.
- Run `pytest test_calibration_audit.py` to verify unit and regression assertions.
