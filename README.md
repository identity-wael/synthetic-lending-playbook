# Synthetic Lending Portfolio

A reproducible educational dataset and Python playbook by Wael El Ghazzawi.

## Contents

- `synthetic_lending_playbook.ipynb`: full generator, executable checks, analysis, charts and exercises.
- `synthetic_lending/loans.csv`: 1,500 fictional loan contracts.
- `synthetic_lending/monthly_performance.csv`: 22,743 monthly observations through August 31, 2026.
- `synthetic_lending/data_dictionary.json`: every field, unit and caveat.
- `synthetic_lending/manifest.json`: generator settings, runtime versions and checksums.
- Three generated charts showing the portfolio snapshot, DPD transitions and age-matched vintage comparison.

## Run

Use Python 3.10+ with NumPy, pandas, Matplotlib and a Jupyter environment. Run every notebook cell in order. No internet connection, credentials, GPU, or external data source is needed. The notebook regenerates its data rather than relying on a download.

Local dependencies can be installed with `pip install numpy pandas matplotlib jupyter`.

Outputs are written to `/kaggle/working/synthetic_lending` on Kaggle, or `./synthetic_lending` relative to the notebook's working directory elsewhere.

## What the data represent

All data are synthetic, generated from scratch with seed 20260922. There is no real borrower information or employer data. Dollar values are fictional USD. All installments fall on month-end dates. Payments consist of whole installments allocated oldest first. The model has no partial payments, fees, late interest, charge-offs, recoveries, closures or prepayments.

Scheduled principal is the contractual balance assuming on-time payments; it is NOT an accounting balance. Scheduled-weighted delinquency is a teaching proxy and must not be presented as actual principal at risk. A DPD90+ state is not a legal default label.

Payment probabilities and segment relationships are manually selected instructional assumptions. This dataset is not representative of a lender, market or customer population and is unsuitable for estimating real-world default risk, pricing, fairness, or model performance. Public uploads should carry the Synthetic tag.

## Validation

A clean local kernel and Kaggle Save & Run All (versions 1 and 2; version 2 imported directly from GitHub) completed the full notebook successfully. Assertions check amortization (including zero and near-zero rates), invalid inputs, FIFO payment allocation across unequal-length calendar months, unique keys, referential integrity, payment conservation, observation dates, scheduled balances, same-seed reproducibility, transition denominators, vintage eligibility and exported schemas/row counts.

Development used OpenAI Codex assistance. Executable tests establish internal consistency, not empirical realism or independent human review.

## License

Original generator notebook, utility, documentation and generated data: CC0 1.0 Universal, https://creativecommons.org/publicdomain/zero/1.0/ . The companion `published_dataset_walkthrough.ipynb` is Apache-2.0. Dependency licenses remain their own.

## GitHub to Kaggle

Source: https://github.com/identity-wael/synthetic-lending-playbook

Kaggle: https://www.kaggle.com/code/waelelghazzawi/synthetic-lending-payments-to-delinquency

To publish an update: execute all cells in a clean local kernel, commit the tested notebook, import its GitHub URL using File > Import Notebook in Kaggle, then use Save & Run All. Check the saved run before publishing. Importing copies a snapshot; GitHub pushes do not automatically update Kaggle.

## Publication status (September 22, 2026)

The GitHub repository, Kaggle generator notebook, reusable utility, and companion dataset are public. Both generator versions passed Kaggle Save & Run All; version 2 was imported directly from GitHub. The utility passed five tests locally and on Kaggle.

- Dataset: https://www.kaggle.com/datasets/waelelghazzawi/synthetic-lending-portfolio-and-monthly-payments
- Utility: https://www.kaggle.com/code/waelelghazzawi/lending-utilities-payments-and-calendar-dpd
- Published-data walkthrough: https://www.kaggle.com/code/waelelghazzawi/synthetic-lending-published-data-walkthrough (passed locally and on Kaggle; saved run 22.6 seconds)
- Colab copy: https://colab.research.google.com/drive/1pC9oquO3CtI1CdC5cJl6uap9bZhpgPWJ (owner access; full run passed)

The dataset was created from notebook outputs, documented, tagged Synthetic, and licensed CC0. No automatic GitHub synchronization is enabled. Kaggle distributes public notebooks under its Apache-2.0 publication license; the original generator, utility, and data are additionally offered here under CC0. The new `published_dataset_walkthrough.ipynb` is Apache-2.0 and reads the published dataset rather than regenerating it.

## R companion

`synthetic_lending_r.ipynb` provides a base-R portfolio analysis, scenario chart, monthly delinquency and payment charts, and two summary CSV exports. It passed Kaggle Save & Run All using R 4.4.0 in 15.1 seconds, including integrity, calendar DPD and denominator assertions. This companion is Apache-2.0 licensed.

Public notebook: https://www.kaggle.com/code/waelelghazzawi/synthetic-lending-portfolio-analysis-in-r

## R Markdown vintage report

`synthetic_lending_vintages.Rmd` compares origination cohorts at MOB 6, reports observed counts and coverage, and contrasts this with a common-calendar snapshot. It passed a Kaggle saved run in 6.3 seconds, with all three validation groups passing and two summary CSV exports. The report is Apache-2.0 licensed.

Public report: https://www.kaggle.com/code/waelelghazzawi/synthetic-lending-vintage-analysis-in-r-markdown
