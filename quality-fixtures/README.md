# Synthetic Lending Data Quality Test Fixtures

A small, intentionally flawed dataset for testing validation logic, not training models. All records and amounts are fictional. Created by Wael El Ghazzawi with OpenAI Codex assistance. CC0-1.0.

## Use
Run `python validate.py` from any directory (Python 3.9+; standard library only). Compare your own validator with expected_violations.csv. Never combine these fixtures with the clean lending portfolio as if they were genuine observations.

There are 11 isolated cases with 12 observations: 3 clean boundary cases and 8 flawed cases producing 9 row-level findings. Duplicate detection is scoped to case_id, loan_id and observation_date; both rows of the duplicate case are flagged. record_id identifies fixture rows and is not the business key.

Clean cases cover a current account, a 31-day interval from leap day, and an unpaid installment due today with DPD zero. Flawed cases cover duplicate keys, missing and impossible observation dates, an unknown loan, a calendar-DPD mismatch, an unpaid-queue/date mismatch, an inconsistent payment total, and negative payments.

## Files and fields
- observations.csv: case_id and record_id identify the test; loan_id references loans.csv. Dates are ISO YYYY-MM-DD or empty. dpd is integer calendar days past due. unpaid_installments and installments_paid are integer counts. monthly_payment_usd and payment_received_usd are fictional decimal USD amounts.
- loans.csv: the allowed loan_id reference set.
- expected_violations.csv: explicit case_id, record_id, rule triplets; no rows means a case passes.
- cases.csv: case_id and expected_violation_count, including clean cases.
- validate.py: executable reference validator.

The payment rule is exact decimal equality: receipts equal whole installments paid times the monthly payment. It is a fixture convention, not a universal lending rule. Negative receipts are invalid here even though real ledgers may use them for reversals. A due-today unpaid installment legitimately has DPD zero. Invalid or missing observation dates suppress dependent calendar-DPD checks to avoid cascading errors.

The validator is scoped to these fixtures. It is not a general parser, production accounting system, financial forecast, fairness benchmark or exhaustive quality suite. Non-numeric fields and malformed due dates are outside this release's test cases. Use these known answers to test detection, not to claim broad production coverage.

Source: https://github.com/identity-wael/synthetic-lending-playbook
