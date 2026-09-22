"""Educational repayment and calendar-delinquency utilities (CC0-1.0).

By Wael El Ghazzawi, developed with OpenAI Codex assistance.
No real customer data. Floating-point teaching calculations, not a servicing
ledger: no fees, partial installments, charge-offs, or currency rounding.
Import this module without side effects; run it directly for executable tests.
"""
from collections import deque
from datetime import date
from math import expm1, isfinite, log1p
from numbers import Integral, Real

__all__ = ['monthly_payment', 'scheduled_balance', 'allocate_installments', 'dpd_bucket']


def _number(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f'{name} must be a finite real number')
    if value < 0 or (positive and value == 0):
        raise ValueError(f'{name} must be {"positive" if positive else "nonnegative"}')


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def monthly_payment(principal, annual_rate, term_months):
    """Level installment; annual_rate is nominal annual rate, not fee-inclusive APR."""
    _number(principal, 'principal', positive=True)
    _number(annual_rate, 'annual_rate')
    _integer(term_months, 'term_months', 1)
    r = annual_rate / 12
    return principal / term_months if r == 0 else principal * r / (-expm1(-term_months * log1p(r)))


def scheduled_balance(principal, annual_rate, term_months, installments_elapsed):
    """Contractual principal assuming on-time payments, NOT actual ledger balance."""
    payment = monthly_payment(principal, annual_rate, term_months)
    _integer(installments_elapsed, 'installments_elapsed')
    if installments_elapsed > term_months:
        raise ValueError('installments_elapsed exceeds term')
    r = annual_rate / 12
    remaining = term_months - installments_elapsed
    return payment * remaining if r == 0 else payment * (-expm1(-remaining * log1p(r))) / r


def allocate_installments(unpaid_due_dates, new_due_date, paid_installments):
    """Add today's installment and allocate a whole-installment payment FIFO.

    Returns (remaining_dates, allocated_count, calendar_dpd). Inputs are not
    mutated. Dates must be strictly increasing datetime.date objects (not
    datetimes). Overpayments are capped; a due-today unpaid installment has
    DPD zero. Call once for each successive installment date.
    """
    _integer(paid_installments, 'paid_installments')
    dates = tuple(unpaid_due_dates)
    if type(new_due_date) is not date or any(type(d) is not date for d in dates):
        raise ValueError('due dates must be datetime.date objects')
    all_dates = dates + (new_due_date,)
    if any(a >= b for a, b in zip(all_dates, all_dates[1:])):
        raise ValueError('due dates must be strictly increasing')
    queue = deque(all_dates)
    allocated = min(paid_installments, len(queue))
    for _ in range(allocated):
        queue.popleft()
    dpd = (new_due_date - queue[0]).days if queue else 0
    return tuple(queue), allocated, dpd


def dpd_bucket(dpd):
    """Calendar DPD label; 90+ is not a legal default determination."""
    _integer(dpd, 'dpd')
    if dpd == 0:
        return 'Current / due today'
    if dpd < 30:
        return '1–29'
    if dpd < 60:
        return '30–59'
    if dpd < 90:
        return '60–89'
    return '90+'


def self_test():
    """Independent recurrence, calendar boundary, immutability, and invalid-input tests."""
    import unittest

    class LendingTests(unittest.TestCase):
        def test_amortization_recurrence(self):
            for rate in (0, 1e-10, .06, .24):
                payment = monthly_payment(10000, rate, 36)
                balance = 10000.
                for month in range(37):
                    self.assertAlmostEqual(balance, scheduled_balance(10000, rate, 36, month), places=6)
                    balance = balance * (1 + rate / 12) - payment
            self.assertEqual(monthly_payment(1200, 0, 12), 100)
            self.assertAlmostEqual(monthly_payment(1000, .12, 1), 1010)

        def test_calendar_fifo(self):
            q, paid, dpd = allocate_installments([], date(2026, 1, 31), 0)
            self.assertEqual((paid, dpd), (0, 0))
            original = q
            q, paid, dpd = allocate_installments(q, date(2026, 2, 28), 0)
            self.assertEqual(original, (date(2026, 1, 31),))
            self.assertEqual((paid, dpd), (0, 28))
            q, paid, dpd = allocate_installments(q, date(2026, 3, 31), 1)
            self.assertEqual((paid, dpd), (1, 31))
            q, paid, dpd = allocate_installments(q, date(2026, 4, 30), 99)
            self.assertEqual((q, paid, dpd), ((), 3, 0))
            self.assertEqual(allocate_installments([date(2024, 1, 31)], date(2024, 2, 29), 0)[2], 29)

        def test_buckets(self):
            self.assertEqual([dpd_bucket(x) for x in (0, 1, 29, 30, 59, 60, 89, 90)],
                             ['Current / due today', '1–29', '1–29', '30–59', '30–59', '60–89', '60–89', '90+'])

        def test_invalid_contract(self):
            for args in ((0, .1, 12), (100, -.1, 12), (100, .1, 0), (100, .1, True), (True, 0, 12), (100, float('nan'), 12)):
                with self.assertRaises(ValueError):
                    monthly_payment(*args)
            for elapsed in (-1, 37, 1.5, True):
                with self.assertRaises(ValueError):
                    scheduled_balance(10000, .1, 36, elapsed)

        def test_invalid_allocation(self):
            for paid in (-1, .5, True):
                with self.assertRaises(ValueError):
                    allocate_installments([], date(2026, 1, 31), paid)
            for dates in ([date(2026, 1, 31)], [date(2026, 2, 28)], ['2025-12-31']):
                with self.assertRaises(ValueError):
                    allocate_installments(dates, date(2026, 1, 31), 1)
            for dpd in (-1, .5, True):
                with self.assertRaises(ValueError):
                    dpd_bucket(dpd)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(LendingTests))
    if not result.wasSuccessful():
        raise RuntimeError('Utility validation failed')
    print('PASS: lending utilities validated; import is side-effect free.')


if __name__ == '__main__':
    self_test()
