"""
Feature engineering and boundary physics engine for Kaggle Playground Series S6E9:
Predicting Electric Vehicle Purchases.

Contains:
1. Domain interaction and readiness ratios
2. Synthetic Data Generating Process (DGP) analytical scoring
3. Physics-based deterministic empirical boundary shifts
4. Zero-tie lexicographical ordinal ranking
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata


def extract_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract domain-specific and physical interaction features."""
    out = df.copy()

    # Numeric conversions
    income = pd.to_numeric(out["Annual_Income_USD"], errors="coerce").fillna(85000.0)
    commute = pd.to_numeric(out["Daily_Commute_km"], errors="coerce").fillna(35.0)
    age = pd.to_numeric(out["Age"], errors="coerce").fillna(45.0)
    home_chargers = pd.to_numeric(out["Charging_Stations_Near_Home"], errors="coerce").fillna(5.0)
    work_chargers = pd.to_numeric(out["Charging_Stations_Near_Work"], errors="coerce").fillna(7.0)
    cars = pd.to_numeric(out["Number_of_Cars_Owned"], errors="coerce").fillna(2.0)
    env_concern = pd.to_numeric(out["Environmental_Concern_Level"], errors="coerce").fillna(3.0)

    # Ratios and interactions
    total_chargers = home_chargers + work_chargers
    out["Total_Charging_Stations"] = total_chargers
    out["Charging_Delta"] = work_chargers - home_chargers
    out["Commute_Per_Station"] = commute / (total_chargers + 1.0)
    out["Commute_Per_Home_Station"] = commute / (home_chargers + 1.0)
    out["Income_Per_Car"] = income / (cars + 1.0)
    out["Income_Per_Age"] = income / np.maximum(age, 18.0)

    # Categorical and psychological readiness
    anxiety_map = {"Low": 1, "Medium": 2, "High": 3}
    anxiety_num = out["Range_Anxiety_Level"].map(anxiety_map).fillna(1).astype(float)
    out["Anxiety_Num"] = anxiety_num
    out["EV_Readiness_Score"] = env_concern * (4.0 - anxiety_num)

    # Binary flags
    subsidy_yes = (out["Subsidy_Available"].astype(str) == "Yes").astype(float)
    home_charge_yes = (out["Home_Charging_Possible"].astype(str) == "Yes").astype(float)
    out["Incentives_Index"] = home_charge_yes * 2.0 + subsidy_yes * 3.0

    return out


def compute_deotte_dgp_score(df: pd.DataFrame) -> np.ndarray:
    """Compute the analytical buying score derived from the synthetic data generating process.

    Recipe: 1.2 * (Income / 100k) + 0.6 * Env - 1.0 * Anxiety_Med - 3.0 * Anxiety_High + 2.0 * Subsidy
    """
    inc = pd.to_numeric(df["Annual_Income_USD"], errors="coerce").fillna(85000.0).values
    env = pd.to_numeric(df["Environmental_Concern_Level"], errors="coerce").fillna(3.0).values
    sub = (df["Subsidy_Available"].astype(str) == "Yes").astype(float).values
    anx_med = (df["Range_Anxiety_Level"].astype(str) == "Medium").astype(float).values
    anx_high = (df["Range_Anxiety_Level"].astype(str) == "High").astype(float).values

    buy_score = 1.2 * (inc / 1e5) + 0.6 * env + 2.0 * sub - 1.0 * anx_med - 3.0 * anx_high
    p_norm = np.clip(norm.cdf(buy_score - 5.5), 1e-6, 1.0 - 1e-6)
    recipe_logit = np.log(p_norm / (1.0 - p_norm))
    return recipe_logit


def apply_deterministic_boundaries(df: pd.DataFrame, base_scores: np.ndarray) -> np.ndarray:
    """Apply empirical physics-based boundary adjustments for regions with 100% or 0% certainty.

    1. Annual_Income_USD >= $170,537 -> 100.0% Buyers (shift +10.0)
    2. $31,004 <= Annual_Income_USD <= $41,970 -> 0.0% Buyers (shift -10.0)
    3. Daily_Commute_km >= 83.0 km -> 0.0% Buyers (shift -5.0)
    4. Annual_Income_USD == 30,000.0 & Subsidy == 'No' & (Env == 1 | Anxiety >= Medium) -> ~0.0% Buyers (shift -5.0)
    """
    n = len(df)
    shift_val = np.zeros(n, dtype=np.float64)

    income = pd.to_numeric(df["Annual_Income_USD"], errors="coerce").values
    commute = pd.to_numeric(df["Daily_Commute_km"], errors="coerce").values
    subsidy_no = (df["Subsidy_Available"].astype(str) == "No").values
    env_1 = (pd.to_numeric(df["Environmental_Concern_Level"], errors="coerce").values == 1.0)
    anx_med_high = df["Range_Anxiety_Level"].isin(["Medium", "High"]).values

    # Rule 1: The Millionaire Cliff (100% true buyers)
    shift_val[income >= 170537.0] += 10.0

    # Rule 2: The Adoption Dead Zone (0% buyers)
    shift_val[(income >= 31004.0) & (income <= 41970.0)] -= 10.0

    # Rule 3: Extreme Commute Cutoff (0% buyers)
    shift_val[commute >= 83.0] -= 5.0

    # Rule 4: Zero-Incentive Low-Income Trap (0.1% buyers)
    shift_val[(income == 30000.0) & subsidy_no & (env_1 | anx_med_high)] -= 5.0

    calibrated = shift_val * 100.0 + base_scores
    return calibrated


def zero_tie_lexicographic_ranking(ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Break all ties analytically and return normalized percentiles [0, 1] with zero ties."""
    order = np.lexsort((ids, scores))
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = (np.arange(len(order)) + 0.5) / len(order)
    return ranks
