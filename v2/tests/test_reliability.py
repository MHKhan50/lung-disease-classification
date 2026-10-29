from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reliability_analysis as ra  # noqa: E402


def synthetic(n: int = 4000, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = (rng.random((n, 14)) < 0.15).astype(int)
    score = y * 1.0 + rng.normal(0, 0.8, (n, 14))
    p = 1 / (1 + np.exp(-(score - 1.5)))
    return y, p


class ReliabilityTests(unittest.TestCase):
    def test_ece_is_zero_for_matching_frequencies(self):
        y = np.array([0, 1] * 50)
        p = np.full(100, 0.5)
        self.assertAlmostEqual(ra.ece(y, p), 0.0, places=9)

    def test_ece_detects_overconfidence(self):
        y = np.zeros(200, dtype=int)
        y[:20] = 1
        self.assertGreater(ra.ece(y, np.full(200, 0.6)), 0.4)

    def test_paired_bootstrap_of_identical_models_is_zero(self):
        y, p = synthetic(800)
        out = ra.paired_bootstrap(y, p, p, n=50)
        self.assertEqual(out["observed_diff"], 0.0)
        self.assertEqual(out["ci95"], [0.0, 0.0])

    def test_platt_scaling_reduces_ece_for_inflated_probabilities(self):
        y, p = synthetic()
        inflated = np.clip(p * 3, 0, 0.999)
        before = np.mean([ra.ece(y[:, k], inflated[:, k]) for k in range(14)])
        fixed = ra.platt_cross_fit(y, inflated)
        after = np.mean([ra.ece(y[:, k], fixed[:, k]) for k in range(14)])
        self.assertLess(after, before / 2)

    def test_selective_keeps_requested_share(self):
        y, p = synthetic(1000)
        rows = ra.selective(y, p, (1.0, 0.5))
        self.assertEqual(rows[0]["n_images"], 1000)
        self.assertEqual(rows[1]["n_images"], 500)
        self.assertEqual(rows[1]["referred_images"], 500)


if __name__ == "__main__":
    unittest.main()
