# Synthetic Validation Reference Results

Reference output for the 11 isolated cases in [Synthetic Lending Data Quality Test Fixtures](https://www.kaggle.com/datasets/waelelghazzawi/synthetic-lending-data-quality-test-fixtures). All data are fictional. This is a tiny software-testing artifact, not a machine-learning benchmark or a measure of production accuracy.

- case_results.csv: each case, expected and detected finding counts, and whether they match. Includes three clean cases.
- rule_results.csv: the nine expected row-rule findings and detection flags. Both rows of the duplicate case count separately.
- manifest.json: exact source revision, SHA-256 input fingerprints, and summary counts.

The reference validator matched all nine expected findings, with no missing or unexpected findings. This means it handles these designed cases; it does not demonstrate exhaustive validation coverage. The parent fixture dataset includes source code, field definitions, and scope limitations. Reproduce with its validate.py. Source revision: 7e40b24b475cb6dc6dc618821386547f3b225d95 of https://github.com/identity-wael/synthetic-lending-playbook.

Created by Wael El Ghazzawi with OpenAI Codex assistance. CC0-1.0. Static release; no automatic refresh.