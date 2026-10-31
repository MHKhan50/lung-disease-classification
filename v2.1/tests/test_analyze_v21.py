from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze_v21 as av  # noqa: E402


def synthetic(n: int = 4000, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = (rng.random((n, 14)) < 0.15).astype(int)
    score = y * 1.0 + rng.normal(0, 0.8, (n, 14))
    p = 1 / (1 + np.exp(-(score - 1.5)))
    return y, p


class AnalysisTests(unittest.TestCase):
    def test_confidence_order_puts_confident_images_first(self):
        p = np.full((3, 14), 0.5)      # most uncertain
        p[1] = 0.01                    # most confident
        p[2] = 0.2
        self.assertEqual(av.confidence_order(p).tolist(), [1, 2, 0])

    def test_referral_keeps_requested_share(self):
        y, p = synthetic(1000)
        rows = av.referral(y, p, (1.0, 0.5))
        self.assertEqual(rows[0]["kept_images"], 1000)
        self.assertEqual(rows[1]["kept_images"], 500)
        self.assertEqual(rows[1]["referred_images"], 500)

    def test_referral_helps_when_confidence_is_informative(self):
        y, p = synthetic(6000)
        rows = av.referral(y, p, (1.0, 0.5))
        self.assertGreater(rows[1]["mean_auroc_kept"], rows[0]["mean_auroc_kept"])

    def test_platt_fitted_on_one_split_fixes_inflated_probabilities_on_another(self):
        y_fit, p_fit = synthetic(4000, seed=1)
        y_new, p_new = synthetic(4000, seed=2)
        inflate = lambda p: np.clip(p * 3, 0, 0.999)  # noqa: E731
        params = av.fit_platt(y_fit, inflate(p_fit))
        before = np.mean([av.ece(y_new[:, k], inflate(p_new)[:, k]) for k in range(14)])
        after_p = av.apply_platt(params, inflate(p_new))
        after = np.mean([av.ece(y_new[:, k], after_p[:, k]) for k in range(14)])
        self.assertLess(after, before / 3)

    def test_platt_keeps_ranking_so_auroc_is_unchanged(self):
        y, p = synthetic(3000)
        params = av.fit_platt(y, p)
        self.assertAlmostEqual(av.mean_auroc(y, p), av.mean_auroc(y, av.apply_platt(params, p)), places=9)

    def test_subgroup_mean_skips_labels_with_too_few_positives(self):
        y, p = synthetic(2000)
        y[:, 13] = 0
        y[0, 13] = 1                    # one positive only, ranked last
        p[:, 13] = 0.9
        p[0, 13] = 0.0
        rows = av.subgroup_rows(y, p, {"all": np.ones(len(y), dtype=bool)}, min_pos=10)
        self.assertEqual(rows[0]["labels_excluded"], "Hernia")
        self.assertGreater(rows[0]["mean_auroc_labels_with_enough_positives"], rows[0]["mean_auroc_all_labels"])

    def test_ece_is_zero_for_matching_frequencies(self):
        self.assertAlmostEqual(av.ece(np.array([0, 1] * 50), np.full(100, 0.5)), 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
