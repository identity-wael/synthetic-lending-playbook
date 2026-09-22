"""Titanic Survival Prediction: Calibrated Feature Pipeline and Baseline Model.

Implements feature engineering, imputation, stratified cross-validation, and
probability calibration with CalibratedClassifierCV.
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold


def extract_title(name_series: pd.Series) -> pd.Series:
    """Extract standard titles from passenger names."""
    extracted = name_series.str.extract(r" ([A-Za-z]+)\.", expand=False)
    standard_map = {
        "Mr": "Mr",
        "Miss": "Miss",
        "Mrs": "Mrs",
        "Master": "Master",
    }
    return extracted.map(lambda x: standard_map.get(x, "Rare"))


def preprocess_titanic(
    df: pd.DataFrame, is_train: bool = True
) -> tuple[pd.DataFrame, pd.Series]:
    """Preprocess Titanic tabular dataset into numeric feature matrix and target/IDs."""
    df = df.copy()

    # Extract title
    titles = extract_title(df["Name"])
    title_encoder = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}
    df["Title"] = titles.map(title_encoder).fillna(4).astype(int)

    # Family size features
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Missing value imputation using deterministic defaults
    df["Age"] = df["Age"].fillna(28.0)
    df["Fare"] = df["Fare"].fillna(14.45)

    # Categorical encoding
    df["Sex"] = (df["Sex"] == "female").astype(int)
    embarked_map = {"S": 0, "C": 1, "Q": 2}
    df["Embarked"] = df["Embarked"].map(embarked_map).fillna(0).astype(int)

    feature_cols = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
        "Title",
        "FamilySize",
        "IsAlone",
    ]

    if is_train:
        return df[feature_cols], df["Survived"]
    return df[feature_cols], df["PassengerId"]


def train_and_evaluate_calibrated_rf(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    test_ids: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Train 5-fold calibrated Random Forest and generate test predictions."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

        base_clf = RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=random_state + fold
        )
        min_class_count = int(y_tr.value_counts().min())
        if min_class_count >= 3:
            cal_clf = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv=3)
            cal_clf.fit(X_tr, y_tr)
        elif min_class_count >= 2:
            cal_clf = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv=2)
            cal_clf.fit(X_tr, y_tr)
        else:
            base_clf.fit(X_tr, y_tr)
            cal_clf = base_clf

        oof_preds[val_idx] = cal_clf.predict_proba(X_va)[:, 1]
        test_preds += cal_clf.predict_proba(X_test)[:, 1] / n_splits

    oof_binary = (oof_preds >= 0.5).astype(int)
    metrics = {
        "oof_accuracy": float(accuracy_score(y, oof_binary)),
        "oof_roc_auc": float(roc_auc_score(y, oof_preds)),
        "oof_brier_score": float(brier_score_loss(y, oof_preds)),
    }

    sub = pd.DataFrame(
        {"PassengerId": test_ids.astype(int), "Survived": (test_preds >= 0.5).astype(int)}
    )
    return metrics, sub
