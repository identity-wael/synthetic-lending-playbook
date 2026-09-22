"""Unit tests for Titanic preprocessing and calibrated model pipeline."""

import os
import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from titanic_model import (
    extract_title,
    preprocess_titanic,
    train_and_evaluate_calibrated_rf,
)


@pytest.fixture
def sample_titanic_data():
    """Create miniature train and test datasets matching Titanic schema."""
    train_df = pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "Survived": [0, 1, 1, 1, 0, 0, 0, 1, 1, 0],
            "Pclass": [3, 1, 3, 1, 3, 3, 1, 3, 2, 2],
            "Name": [
                "Braund, Mr. Owen Harris",
                "Cumings, Mrs. John Bradley",
                "Heikkinen, Miss. Laina",
                "Futrelle, Mrs. Jacques Heath",
                "Allen, Mr. William Henry",
                "Moran, Mr. James",
                "McCarthy, Mr. Timothy J",
                "Johnson, Mrs. Oscar W",
                "Nasser, Mrs. Nicholas",
                "Sandstrom, Miss. Marguerite Rut",
            ],
            "Sex": [
                "male",
                "female",
                "female",
                "female",
                "male",
                "male",
                "male",
                "female",
                "female",
                "female",
            ],
            "Age": [22.0, 38.0, 26.0, 35.0, 35.0, None, 54.0, 27.0, 14.0, 4.0],
            "SibSp": [1, 1, 0, 1, 0, 0, 0, 0, 1, 1],
            "Parch": [0, 0, 0, 0, 0, 0, 0, 2, 0, 1],
            "Ticket": [
                "A/5",
                "PC",
                "STON",
                "113803",
                "373450",
                "330877",
                "17463",
                "347742",
                "237736",
                "PP",
            ],
            "Fare": [7.25, 71.28, 7.92, 53.10, 8.05, 8.45, 51.86, 11.13, 30.07, 16.70],
            "Cabin": [None, "C85", None, "C123", None, None, "E46", None, None, "G6"],
            "Embarked": ["S", "C", "S", "S", "S", "Q", "S", "S", "C", "S"],
        }
    )

    test_df = pd.DataFrame(
        {
            "PassengerId": [101, 102],
            "Pclass": [3, 1],
            "Name": ["Kelly, Mr. James", "Wilkes, Mrs. James"],
            "Sex": ["male", "female"],
            "Age": [34.5, 47.0],
            "SibSp": [0, 1],
            "Parch": [0, 0],
            "Ticket": ["330911", "363272"],
            "Fare": [7.82, None],
            "Cabin": [None, None],
            "Embarked": ["Q", "S"],
        }
    )

    return train_df, test_df


def test_extract_title():
    names = pd.Series(["Smith, Mr. John", "Doe, Mrs. Jane", "Jones, Dr. Bob"])
    titles = extract_title(names)
    assert list(titles) == ["Mr", "Mrs", "Rare"]


def test_preprocessing(sample_titanic_data):
    train_df, test_df = sample_titanic_data
    X_train, y_train = preprocess_titanic(train_df, is_train=True)
    X_test, test_ids = preprocess_titanic(test_df, is_train=False)

    assert len(X_train) == 10
    assert len(y_train) == 10
    assert len(X_test) == 2
    assert list(test_ids) == [101, 102]
    # Check no missing values remain
    assert X_train.isna().sum().sum() == 0
    assert X_test.isna().sum().sum() == 0


def test_train_and_evaluate(sample_titanic_data):
    train_df, test_df = sample_titanic_data
    X_train, y_train = preprocess_titanic(train_df, is_train=True)
    X_test, test_ids = preprocess_titanic(test_df, is_train=False)

    metrics, sub = train_and_evaluate_calibrated_rf(
        X_train, y_train, X_test, test_ids, n_splits=2, random_state=42
    )

    assert "oof_accuracy" in metrics
    assert "oof_roc_auc" in metrics
    assert "oof_brier_score" in metrics
    assert len(sub) == 2
    assert list(sub.columns) == ["PassengerId", "Survived"]
    assert set(sub["Survived"].unique()).issubset({0, 1})
