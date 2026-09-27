"""Ensure full reproduction agrees with the published aggregate results."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_aggregate_reproduction():
    for reference in sorted((ROOT / 'reference_results').rglob('*.csv')):
        relative = reference.relative_to(ROOT / 'reference_results')
        actual = ROOT / 'results_sd10' / relative
        expected_frame = pd.read_csv(reference)
        actual_frame = pd.read_csv(actual)
        # CSV serialization and linear algebra can introduce tiny differences.
        pd.testing.assert_frame_equal(actual_frame, expected_frame,
                                      check_exact=False, rtol=1e-8, atol=1e-8)
