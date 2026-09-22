# Kaggle S6E9: Predicting Electric Vehicle Purchases
## Grandmaster Rank-Space Meta-Ensemble & Empirical Boundary Physics

[![ROC-AUC](https://img.shields.io/badge/Public%20LB-0.94657-success?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/competitions/playground-series-s6e9)
[![Rank](https://img.shields.io/badge/Rank-Top%201.5%25%20(41%2F2%2C725)-gold?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/competitions/playground-series-s6e9/leaderboard)
[![Zero Ties](https://img.shields.io/badge/Predictions-100%25%20Zero%20Ties-blue?style=for-the-badge)](https://www.kaggle.com/code/waelelghazzawi/s6e9-ev-rank-space-boundary-ensemble)

High-performance, mathematically calibrated machine learning pipeline for [Kaggle Playground Series Season 6 Episode 9: Predicting Electric Vehicle Purchases](https://www.kaggle.com/competitions/playground-series-s6e9).

---

## 🏆 Leaderboard Performance
- **Public ROC-AUC:** `0.94657`
- **Global Standing:** **Rank 41 out of 2,725 teams** (Top 1.5% worldwide) on submission #1.
- **Ties:** 0 (286,571 strictly unique predictions across 286,571 test cases).

---

## 🔬 Mathematical Boundary Physics Discovery
Across all 668,665 training rows, rigorous empirical audit identified four zero-error invariant regions in the synthetic data generator:

1. **High-Income Certainty Boundary (`Annual_Income_USD >= $170,537`):**
   - **100.000% True Positive Rate** across 393 training observations (0 false negatives).
   - Injected calibration offset: `+1000.0` rank score.
2. **Adoption Dead Zone (`$31,004 <= Annual_Income_USD <= $41,970`):**
   - **0.00000% True Positive Rate** across 1,257 training observations (0 false positives).
   - Injected calibration offset: `-1000.0` rank score.
3. **Extreme Commute Cutoff (`Daily_Commute_km >= 83.0`):**
   - **0.00000% True Positive Rate** across 186 training observations (0 false positives).
   - Injected calibration offset: `-500.0` rank score.
4. **Zero-Incentive Low-Income Trap:**
   - `Annual_Income_USD == $30,000` & `Subsidy_Available == 'No'` & (`Environmental_Concern_Level == 1` | `Range_Anxiety in ['Medium', 'High']`)
   - Purchase rate collapses to near zero; offset: `-500.0` rank score.

---

## 🧠 System Architecture & Methodology

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Raw S6E9 Test Set (N=286,571)                   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
┌───────────────────────────────┐             ┌───────────────────────────────┐
│ Domain Feature Extraction     │             │ High-Precision Candidate Pool │
│ • Total Charging Stations     │             │ • r8_m50 (50/50 Mega-Champ)   │
│ • Commute / Station Ratios    │             │ • r8_m75 (75/25 Mega-Champ)   │
│ • EV Readiness Index          │             │ • r8_m00 (Megayak Verbatim)   │
│ • Incentives Composite Score  │             │ • r6_rebuilt (94651 Honest)   │
└───────────────┬───────────────┘             └───────────────┬───────────────┘
                │                                             │
                ▼                                             ▼
┌───────────────────────────────┐             ┌───────────────────────────────┐
│ Analytical Deotte DGP Score   │             │ Quantile Rank Space Fusion    │
│ (Synthetic Generator Formula) │             │ (Normalized Uniform [0, 1])   │
└───────────────┬───────────────┘             └───────────────┬───────────────┘
                │                                             │
                └──────────────────────┬──────────────────────┘
                                       ▼
             ┌──────────────────────────────────────────────────┐
             │ Deterministic Boundary Physics Calibration Engine│
             │ (+1000 Cliff, -1000 Dead Zone, -500 Cutoffs)     │
             └─────────────────────────┬────────────────────────┘
                                       ▼
             ┌──────────────────────────────────────────────────┐
             │ Zero-Tie Lexicographical Ordinal Ranking         │
             │ order = np.lexsort((ids, calibrated_scores))     │
             │ ranks = (np.arange(N) + 0.5) / N                 │
             └─────────────────────────┬────────────────────────┘
                                       ▼
             ┌──────────────────────────────────────────────────┐
             │ Validated Final Submission (286,571 rows)        │
             │ Status: COMPLETE | Public ROC-AUC: 0.94657       │
             └──────────────────────────────────────────────────┘
```

---

## 📂 Repository Structure

- [`ev_features.py`](file:///Users/wael/kaggle/ev-purchases-demo/ev_features.py): Feature transformations, Deotte DGP formulation, and boundary physics rules.
- [`ev_pipeline.py`](file:///Users/wael/kaggle/ev-purchases-demo/ev_pipeline.py): End-to-end rank fusion, validation checks, and submission builder.
- [`generate_submission.py`](file:///Users/wael/kaggle/ev-purchases-demo/generate_submission.py): CLI tool to assemble and write verified `submission.csv`.
- [`test_ev_pipeline.py`](file:///Users/wael/kaggle/ev-purchases-demo/test_ev_pipeline.py): Unit test suite covering feature integrity, boundary shifts, and zero-tie ranking.
- [`notebook/s6e9_ev_rank_space_boundary_ensemble.ipynb`](file:///Users/wael/kaggle/ev-purchases-demo/notebook/s6e9_ev_rank_space_boundary_ensemble.ipynb): Published reproducible Kaggle notebook.
- [`submission.csv`](file:///Users/wael/kaggle/ev-purchases-demo/submission.csv): Final competition submission file scoring `0.94657`.

---

## 🚀 Reproduction & Execution

### 1. Run Unit Test Suite
```bash
.venv/bin/pytest ev-purchases-demo/test_ev_pipeline.py -v
```

### 2. Generate Submission File
```bash
.venv/bin/python ev-purchases-demo/generate_submission.py
```

### 3. Submit Directly via Kaggle API
```bash
.venv/bin/kaggle competitions submit -c playground-series-s6e9 \
    -f ev-purchases-demo/submission.csv \
    -m "Grandmaster Rank-Space Meta-Ensemble with Boundary Physics v1"
```
