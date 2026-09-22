"""Reference validator for isolated synthetic fixtures. Python standard library only."""
from pathlib import Path
from collections import Counter
from datetime import date
from decimal import Decimal
import csv
ROOT = Path(__file__).resolve().parent

def validate(rows, known_loans):
    findings = set()
    keys = Counter((r['case_id'], r['loan_id'], r['observation_date']) for r in rows)
    for r in rows:
        def flag(rule): findings.add((r['case_id'], r['record_id'], rule))
        if keys[r['case_id'], r['loan_id'], r['observation_date']] > 1: flag('duplicate_observation')
        if r['loan_id'] not in known_loans: flag('unknown_loan')
        observation = None
        if not r['observation_date']: flag('missing_observation_date')
        else:
            try: observation = date.fromisoformat(r['observation_date'])
            except ValueError: flag('invalid_observation_date')
        unpaid = int(r['unpaid_installments'])
        due = date.fromisoformat(r['oldest_unpaid_due_date']) if r['oldest_unpaid_due_date'] else None
        if (due is None) != (unpaid == 0): flag('unpaid_date_mismatch')
        if observation and (due or unpaid == 0):
            expected_dpd = (observation - due).days if due else 0
            if int(r['dpd']) != expected_dpd: flag('calendar_dpd_mismatch')
        paid = int(r['installments_paid'])
        receipt = Decimal(r['payment_received_usd'])
        if paid < 0 or receipt < 0: flag('negative_payment')
        if receipt != paid * Decimal(r['monthly_payment_usd']): flag('payment_total_mismatch')
    return findings

if __name__ == '__main__':
    def read(name):
        with (ROOT/name).open(newline='') as f: return list(csv.DictReader(f))
    actual = validate(read('observations.csv'), {r['loan_id'] for r in read('loans.csv')})
    expected = {(r['case_id'],r['record_id'],r['rule']) for r in read('expected_violations.csv')}
    assert actual == expected, f'Missing: {expected-actual}; unexpected: {actual-expected}'
    counts = Counter(c for c,_,_ in actual)
    for case in read('cases.csv'):
        assert counts[case['case_id']] == int(case['expected_violation_count'])
    print(f'PASS: {len(read("cases.csv"))} cases; {len(actual)} expected violations; no unexpected findings')
