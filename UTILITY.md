# Lending utilities

Reusable educational Python functions by Wael El Ghazzawi. CC0-1.0.

```python
from lending_utils import monthly_payment, scheduled_balance, allocate_installments, dpd_bucket
from datetime import date
payment = monthly_payment(1200, 0, 12)  # 100
queue, paid, dpd = allocate_installments([], date(2026, 1, 31), 0)
queue, paid, dpd = allocate_installments(queue, date(2026, 2, 28), 0)
assert dpd == 28
```

Run `python lending_utils.py` for five tests covering independent amortization recurrence, calendar FIFO allocation and leap years, bucket boundaries, and rejected inputs. Importing runs no tests, writes no files, and prints nothing.

Uses Python standard library only. Whole installments only; excess payments are capped. Due dates must be strictly increasing calendar dates. Scheduled balance assumes on-time repayment and is not an accounting balance. No fees, partial payments, charge-offs, currency rounding, or legal default determination. Developed with OpenAI Codex assistance.

On Kaggle, add the saved utility script through the notebook input picker, then use the import name provided by Kaggle. The downloadable module can be imported directly as shown above.
