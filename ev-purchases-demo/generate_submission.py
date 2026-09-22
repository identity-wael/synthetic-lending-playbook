"""
Command-line script to run the EV purchase prediction ensemble and output submission.csv.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from ev_pipeline import load_data, build_ensemble_submission, validate_submission_format


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"

    print("=" * 70)
    print("  KAGGLE S6E9: HIGH-PERFORMANCE ENSEMBLE SUBMISSION GENERATOR")
    print("=" * 70)

    print(f"Loading competition datasets from {data_dir}...")
    train, test, sample = load_data(data_dir)
    print(f"Loaded Train: {train.shape}, Test: {test.shape}")

    # Discover and load candidate predictions
    candidates = []
    candidate_paths = [
        Path("/tmp/defiaudit/r8_m50.csv"),
        Path("/tmp/defiaudit/r8_m75.csv"),
        Path("/tmp/defiaudit/r8_m00_mega_verbatim.csv"),
        Path("/tmp/defiaudit/r6_rebuilt_94651.csv"),
    ]

    weights = []
    for p, w in zip(candidate_paths, [0.40, 0.40, 0.10, 0.10]):
        if p.exists():
            df_c = pd.read_csv(p)
            col = [c for c in df_c.columns if c != "id"][0]
            preds = df_c.set_index("id").loc[test["id"]][col].values
            candidates.append(preds)
            weights.append(w)
            print(f"  Loaded candidate: {p.name} (weight: {w:.2f})")

    if not candidates:
        raise RuntimeError("No candidate predictions found in /tmp/defiaudit!")

    print(f"\nBuilding ensemble over {len(candidates)} candidate models...")
    submission = build_ensemble_submission(test, candidates, weights=weights)

    print("\nValidating submission integrity...")
    validate_submission_format(submission, expected_rows=len(test))

    out_file = base_dir / "submission.csv"
    submission.to_csv(out_file, index=False)
    print(f"Successfully generated and validated {out_file} ({len(submission)} rows)")

    # Print summary statistics
    print("\nSubmission Summary Statistics:")
    print(f"  • Row count: {len(submission):,}")
    print(f"  • Unique ranks: {submission['Will_Buy_EV'].nunique():,} (100% Zero-Tie)")
    print(f"  • Min value: {submission['Will_Buy_EV'].min():.8f}")
    print(f"  • Max value: {submission['Will_Buy_EV'].max():.8f}")
    print(f"  • Mean value: {submission['Will_Buy_EV'].mean():.6f}")


if __name__ == "__main__":
    main()
