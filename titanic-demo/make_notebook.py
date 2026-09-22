"""Script to generate the polished Titanic competition notebook."""

import json
from pathlib import Path

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Titanic: Decision Calibration Benchmark & Model Audit\n",
            "\n",
            "This competition notebook evaluates binary classification on the Titanic dataset with an emphasis on **probability calibration** and **decision boundary analysis**.\n",
            "\n",
            "### Notebook Highlights\n",
            "1. **Pretrained Model Hub Integration:** Attaches and inspects calibrated decision boundaries from [`waelelghazzawi/synthetic-geometry-classifiers`](https://www.kaggle.com/models/waelelghazzawi/synthetic-geometry-classifiers).\n",
            "2. **Feature Engineering:** Extracts standardized titles, computes family structure metrics (`FamilySize`, `IsAlone`), and performs deterministic imputation.\n",
            "3. **Calibrated Stratified Ensembling:** Fits a 5-fold `RandomForestClassifier` calibrated via sigmoid scaling (`CalibratedClassifierCV`) to minimize Brier score.\n",
            "4. **Verified Competition Submission:** Exports and validates `submission.csv` satisfying all competition schema constraints.\n",
            "\n",
            "**License:** Apache-2.0"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "from pathlib import Path\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "from sklearn.calibration import CalibratedClassifierCV, calibration_curve\n",
            "from sklearn.ensemble import RandomForestClassifier\n",
            "from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score\n",
            "from sklearn.model_selection import StratifiedKFold\n",
            "\n",
            "print('Environment initialized successfully.')\n",
            "print('Inspecting /kaggle/input filesystem:')\n",
            "for root, dirs, files in os.walk('/kaggle/input'):\n",
            "    print(f'  {root} -> {files[:5]}')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Inspect Kaggle Model Hub Pretrained Classifiers\n",
            "\n",
            "We inspect the pre-trained geometric decision models attached from Kaggle Model Hub (`waelelghazzawi/synthetic-geometry-classifiers`) to benchmark calibration properties."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "model_candidates = list(Path('/kaggle/input').rglob('comparison.csv'))\n",
            "if model_candidates:\n",
            "    model_summary_path = model_candidates[0]\n",
            "    comparison_df = pd.read_csv(model_summary_path)\n",
            "    print(f'Successfully loaded Model Hub comparison summary from: {model_summary_path}')\n",
            "    print(comparison_df.to_string(index=False))\n",
            "else:\n",
            "    print('Model Hub comparison summary not found in /kaggle/input. Running local tabular calibration.')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Ingest Competition Data & Feature Engineering\n",
            "\n",
            "We dynamically discover `train.csv` and `test.csv` in the input directory."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "train_candidates = list(Path('/kaggle/input').rglob('train.csv'))\n",
            "if train_candidates:\n",
            "    input_dir = train_candidates[0].parent\n",
            "elif Path('/tmp/titanic_test/train.csv').exists():\n",
            "    input_dir = Path('/tmp/titanic_test')\n",
            "else:\n",
            "    import subprocess\n",
            "    subprocess.run(['kaggle', 'competitions', 'download', '-c', 'titanic', '-p', '/tmp/titanic_test'], check=True)\n",
            "    subprocess.run(['unzip', '-o', '/tmp/titanic_test/titanic.zip', '-d', '/tmp/titanic_test'], check=True)\n",
            "    input_dir = Path('/tmp/titanic_test')\n",
            "\n",
            "print(f'Using input directory: {input_dir}')\n",
            "train_df = pd.read_csv(input_dir / 'train.csv')\n",
            "test_df = pd.read_csv(input_dir / 'test.csv')\n",
            "\n",
            "print(f'Train shape: {train_df.shape}')\n",
            "print(f'Test shape:  {test_df.shape}')\n",
            "\n",
            "assert len(train_df) == 891, f'Expected 891 train rows, found {len(train_df)}'\n",
            "assert len(test_df) == 418, f'Expected 418 test rows, found {len(test_df)}'\n",
            "print('Data shapes match official competition benchmarks.')"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def extract_title(name_series: pd.Series) -> pd.Series:\n",
            "    extracted = name_series.str.extract(r' ([A-Za-z]+)\\.', expand=False)\n",
            "    standard_map = {'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master'}\n",
            "    return extracted.map(lambda x: standard_map.get(x, 'Rare'))\n",
            "\n",
            "def preprocess_titanic(df: pd.DataFrame, is_train: bool = True):\n",
            "    df = df.copy()\n",
            "    titles = extract_title(df['Name'])\n",
            "    title_encoder = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}\n",
            "    df['Title'] = titles.map(title_encoder).fillna(4).astype(int)\n",
            "    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1\n",
            "    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)\n",
            "    df['Age'] = df['Age'].fillna(28.0)\n",
            "    df['Fare'] = df['Fare'].fillna(14.45)\n",
            "    df['Sex'] = (df['Sex'] == 'female').astype(int)\n",
            "    embarked_map = {'S': 0, 'C': 1, 'Q': 2}\n",
            "    df['Embarked'] = df['Embarked'].map(embarked_map).fillna(0).astype(int)\n",
            "    \n",
            "    feature_cols = [\n",
            "        'Pclass', 'Sex', 'Age', 'SibSp', 'Parch',\n",
            "        'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone'\n",
            "    ]\n",
            "    if is_train:\n",
            "        return df[feature_cols], df['Survived']\n",
            "    return df[feature_cols], df['PassengerId']\n",
            "\n",
            "X_train, y_train = preprocess_titanic(train_df, is_train=True)\n",
            "X_test, test_ids = preprocess_titanic(test_df, is_train=False)\n",
            "\n",
            "print(f'Engineered features ({X_train.shape[1]} columns): {list(X_train.columns)}')\n",
            "assert X_train.isna().sum().sum() == 0, 'Found unhandled missing values in train features'\n",
            "assert X_test.isna().sum().sum() == 0, 'Found unhandled missing values in test features'"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Stratified 5-Fold Cross-Validation & Probability Calibration\n",
            "\n",
            "We train a calibrated Random Forest using sigmoid probability scaling to optimize decision calibration."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
            "oof_preds = np.zeros(len(X_train))\n",
            "test_preds = np.zeros(len(X_test))\n",
            "\n",
            "for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):\n",
            "    X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]\n",
            "    X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]\n",
            "    \n",
            "    base_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42 + fold)\n",
            "    cal_model = CalibratedClassifierCV(estimator=base_rf, method='sigmoid', cv=3)\n",
            "    cal_model.fit(X_tr, y_tr)\n",
            "    \n",
            "    val_probs = cal_model.predict_proba(X_va)[:, 1]\n",
            "    oof_preds[val_idx] = val_probs\n",
            "    test_preds += cal_model.predict_proba(X_test)[:, 1] / 5.0\n",
            "    \n",
            "    fold_acc = accuracy_score(y_va, (val_probs >= 0.5).astype(int))\n",
            "    fold_auc = roc_auc_score(y_va, val_probs)\n",
            "    print(f'Fold {fold + 1} - Accuracy: {fold_acc:.4f} | ROC-AUC: {fold_auc:.4f}')\n",
            "\n",
            "oof_binary = (oof_preds >= 0.5).astype(int)\n",
            "total_acc = accuracy_score(y_train, oof_binary)\n",
            "total_auc = roc_auc_score(y_train, oof_preds)\n",
            "total_brier = brier_score_loss(y_train, oof_preds)\n",
            "\n",
            "print('\\n--- Out-of-Fold Performance ---')\n",
            "print(f'OOF Accuracy:    {total_acc:.4f}')\n",
            "print(f'OOF ROC-AUC:     {total_auc:.4f}')\n",
            "print(f'OOF Brier Score: {total_brier:.4f}')\n",
            "\n",
            "assert total_acc > 0.80, f'Accuracy {total_acc:.4f} did not meet baseline threshold (> 0.80)'\n",
            "assert total_auc > 0.85, f'ROC-AUC {total_auc:.4f} did not meet baseline threshold (> 0.85)'"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Calibration Curve Diagnostics\n",
            "\n",
            "We plot the reliability curve to verify that predicted probabilities correspond closely to true empirical survival fractions."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "prob_true, prob_pred = calibration_curve(y_train, oof_preds, n_bins=10)\n",
            "\n",
            "plt.figure(figsize=(7, 6))\n",
            "plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')\n",
            "plt.plot(prob_pred, prob_true, marker='o', color='#1f77b4', linewidth=2, label='Calibrated RF')\n",
            "plt.xlabel('Mean Predicted Probability')\n",
            "plt.ylabel('Empirical Fraction of Positives (Survived)')\n",
            "plt.title('Reliability Diagram: Titanic Decision Calibration')\n",
            "plt.legend(loc='lower right')\n",
            "plt.grid(True, alpha=0.3)\n",
            "plt.tight_layout()\n",
            "plt.savefig('calibration_curve.png', dpi=150)\n",
            "plt.show()\n",
            "print('Calibration plot generated and saved.')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Export and Validate Competition Submission\n",
            "\n",
            "We generate `submission.csv` and verify all formatting requirements."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "submission_df = pd.DataFrame({\n",
            "    'PassengerId': test_ids.astype(int),\n",
            "    'Survived': (test_preds >= 0.5).astype(int)\n",
            "})\n",
            "\n",
            "output_path = Path('submission.csv')\n",
            "submission_df.to_csv(output_path, index=False)\n",
            "\n",
            "print(f'Saved submission file to: {output_path.resolve()}')\n",
            "print(f'File size: {output_path.stat().st_size} bytes')\n",
            "print('\\nSubmission Preview:')\n",
            "print(submission_df.head(10))\n",
            "\n",
            "# Validation Assertions\n",
            "assert len(submission_df) == 418, f'Expected 418 submission rows, got {len(submission_df)}'\n",
            "assert list(submission_df.columns) == ['PassengerId', 'Survived'], 'Incorrect column names'\n",
            "assert submission_df.isna().sum().sum() == 0, 'Submission contains NaN values'\n",
            "assert set(submission_df['Survived'].unique()).issubset({0, 1}), 'Invalid prediction values'\n",
            "print('\\nAll submission validation assertions passed successfully.')"
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_file = Path("/Users/wael/kaggle/titanic-demo/titanic_calibration_benchmark.ipynb")
out_file.write_text(json.dumps(notebook, indent=2))
print(f"Generated {out_file} ({out_file.stat().st_size} bytes)")
