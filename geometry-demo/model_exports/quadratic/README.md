# Synthetic Geometry: quadratic

Educational binary logistic classifier. Apache-2.0. Created by Wael El Ghazzawi with OpenAI Codex assistance.

Inputs x1,x2 are uniform coordinates in [-1,1]. Label 1 means x1^2+x2^2 < 0.5, followed by independent 5% label flips. No validated real-world use.

Train:6000 points seed1201; validation:2000 seed1202; test:2000 seed1203. Select C among 0.1,1,10 by validation log loss, never test data. Metrics apply only to this noisy generator, without confidence intervals or external validation.

JSON contains explicit feature order, coefficients and intercept. Transform inputs, take the coefficient dot product plus intercept, apply sigmoid, then threshold at 0.5. Test vectors verify other implementations. No pickle or executable weights. No fine-tuning API; retrain using source. Version1.
