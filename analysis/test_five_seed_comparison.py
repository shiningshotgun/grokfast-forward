#!/usr/bin/env python3
"""test_five_seed_comparison.py - Unit tests for marginal median and paired comparison logic under censoring."""

import unittest
import sys
import os

# Add repo root to path to import analysis functions
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "analysis"))

from five_seed_comparison import (
    compute_marginal_median,
    format_marginal_median,
    classify_pair_and_diff,
    get_crossing_bar_props,
)


class TestMarginalMedianUnderCensoring(unittest.TestCase):
    """Synthetic regression tests for marginal median under right-censoring beyond the 50k horizon."""

    def test_zero_observed_crossings(self):
        # 0 of 5 crossed (5 censored)
        self.assertIsNone(compute_marginal_median([], total_seeds=5))
        formatted = format_marginal_median([], total_seeds=5)
        self.assertEqual(formatted, "not estimable within the 50,000-update horizon (0/5 crossed)")

    def test_two_observed_crossings(self):
        # 2 of 5 crossed (3 censored) -> 3rd ordered value is censored, median not estimable
        self.assertIsNone(compute_marginal_median([10000, 20000], total_seeds=5))
        formatted = format_marginal_median([10000, 20000], total_seeds=5)
        self.assertEqual(formatted, "not estimable within the 50,000-update horizon (2/5 crossed)")

    def test_three_observed_crossings(self):
        # 3 of 5 crossed (2 censored) -> sorted: [15000, 25000, 35000, >50k, >50k]
        # 3rd ordered value is 35000 (observed crossing)
        self.assertEqual(compute_marginal_median([35000, 15000, 25000], total_seeds=5), 35000)
        formatted = format_marginal_median([35000, 15000, 25000], total_seeds=5)
        self.assertEqual(formatted, "35,000")

    def test_four_observed_crossings(self):
        # 4 of 5 crossed (1 censored beyond horizon) -> sorted: [c1, c2, c3, c4, >50k]
        # 3rd ordered value is c3 (third observed crossing)
        # Case matching wd=0 EMA: [46800, 49900, 49700, 49400] -> sorted: 46800, 49400, 49700, 49900 -> 49700
        self.assertEqual(compute_marginal_median([46800, 49900, 49700, 49400], total_seeds=5), 49700)
        self.assertEqual(format_marginal_median([46800, 49900, 49700, 49400], total_seeds=5), "49,700")

        # Case matching wd=0.005 baseline: [41300, 49000, 46900, 21000] -> sorted: 21000, 41300, 46900, 49000 -> 46900
        self.assertEqual(compute_marginal_median([41300, 49000, 46900, 21000], total_seeds=5), 46900)
        self.assertEqual(format_marginal_median([41300, 49000, 46900, 21000], total_seeds=5), "46,900")

    def test_five_observed_crossings(self):
        # 5 of 5 crossed (0 censored)
        # Case matching wd=0.005 EMA: [40200, 48700, 17600, 48700, 44700] -> sorted: 17600, 40200, 44700, 48700, 48700 -> 44700
        self.assertEqual(compute_marginal_median([40200, 48700, 17600, 48700, 44700], total_seeds=5), 44700)
        self.assertEqual(format_marginal_median([40200, 48700, 17600, 48700, 44700], total_seeds=5), "44,700")

    def test_censored_third_value_is_not_estimable(self):
        # Explicit test: When fewer than 3 crossings occur (e.g. 1 or 2 crossings), the 3rd ordered value
        # falls in the censored region (>50,000), so the median must be strictly not estimable.
        for n_obs in [0, 1, 2]:
            crossings = [10000 * (i + 1) for i in range(n_obs)]
            self.assertIsNone(compute_marginal_median(crossings, total_seeds=5))
            self.assertIn("not estimable", format_marginal_median(crossings, total_seeds=5))


class TestPairedComparisonUnderCensoring(unittest.TestCase):
    """Verifies that pairs containing censored arms never produce a numeric crossing difference."""

    def test_both_cross_produces_numeric_diff(self):
        # none crossed at 40000, ema crossed at 30000
        m_none = {"crossed": True, "cross_step": 40000}
        m_ema = {"crossed": True, "cross_step": 30000}
        quadrant, diff = classify_pair_and_diff(m_none, m_ema)
        self.assertEqual(quadrant, "both-cross")
        self.assertEqual(diff, -10000)
        self.assertIsInstance(diff, int)

    def test_ema_only_cross_never_produces_numeric_diff(self):
        # none censored, ema crossed at 45000
        m_none = {"crossed": False, "cross_step": None}
        m_ema = {"crossed": True, "cross_step": 45000}
        quadrant, diff = classify_pair_and_diff(m_none, m_ema)
        self.assertEqual(quadrant, "EMA-only-cross")
        self.assertIsNone(diff)

    def test_none_only_cross_never_produces_numeric_diff(self):
        # none crossed at 45000, ema censored
        m_none = {"crossed": True, "cross_step": 45000}
        m_ema = {"crossed": False, "cross_step": None}
        quadrant, diff = classify_pair_and_diff(m_none, m_ema)
        self.assertEqual(quadrant, "none-only-cross")
        self.assertIsNone(diff)

    def test_neither_cross_never_produces_numeric_diff(self):
        # both censored
        m_none = {"crossed": False, "cross_step": None}
        m_ema = {"crossed": False, "cross_step": None}
        quadrant, diff = classify_pair_and_diff(m_none, m_ema)
        self.assertEqual(quadrant, "neither")
        self.assertIsNone(diff)


class TestPlottingCensoredProps(unittest.TestCase):
    """Regression test ensuring censored runs are distinctly styled from observed crossings."""

    def test_observed_crossing_bar_props(self):
        # Observed crossing at step 41,300
        m_arm = {"crossed": True, "cross_step": 41300}
        props = get_crossing_bar_props(m_arm, "#d95f02", default_cap=50000)
        self.assertFalse(props["is_censored"])
        self.assertEqual(props["height"], 41300)
        self.assertEqual(props["color"], "#d95f02")
        self.assertEqual(props["edgecolor"], "none")
        self.assertIsNone(props["hatch"])
        self.assertEqual(props["linewidth"], 0.0)

    def test_censored_bar_props(self):
        # Censored run (never crossed)
        m_arm = {"crossed": False, "cross_step": None}
        props = get_crossing_bar_props(m_arm, "#d95f02", default_cap=50000)
        self.assertTrue(props["is_censored"])
        self.assertEqual(props["height"], 50000)
        self.assertEqual(props["color"], "none")
        self.assertEqual(props["edgecolor"], "#d95f02")
        self.assertEqual(props["hatch"], "///")
        self.assertGreater(props["linewidth"], 0.0)


if __name__ == "__main__":
    unittest.main()
