"""Current bin-level replay regression checks; no training or inference."""
from pathlib import Path
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.current_figures.score_distributions import (
    BINS, SPECS, distribution_panel, pchip, validate_bins,
)

ROOT = Path(__file__).resolve().parents[1]


class ScoreDistributionTests(unittest.TestCase):
    def test_current_bins_match_every_frozen_probability(self):
        total = 0
        for panel, species in [("Figure3c", "human"), ("Figure3d", "mouse")]:
            table = pd.read_csv(ROOT / f"data/panels/current/{panel}.csv",
                                float_precision="round_trip")
            validate_bins(table, panel)
            raw = pd.read_csv(ROOT / f"data/frozen_model/{species}_all_predictions.csv",
                              float_precision="round_trip")
            for group in SPECS[panel]["order"]:
                block = table.loc[table.dataset.eq(group)].sort_values("bin_left")
                counts, _ = np.histogram(raw.loc[raw.dataset.eq(group), "probability"], BINS)
                np.testing.assert_array_equal(block.protein_count, counts)
            total += len(table)
        self.assertEqual(total, 140)

    def test_pchip_matches_knots_and_does_not_extrapolate_or_overshoot(self):
        x = (BINS[:-1] + BINS[1:]) / 2
        y = np.array([0, 5, 2, 0, 1, 1, 1, 3, 10, 0, 1, 2, 8, 1, 0, 0, 0, 0, 0, 65.])
        xx, yy = pchip(x, y)
        self.assertEqual(len(xx), 381)
        np.testing.assert_array_equal(xx[::20], x)
        np.testing.assert_array_equal(yy[::20], y)
        self.assertEqual((xx[0], xx[-1]), (.025, .975))
        for i in range(19):
            block = yy[20*i:20*(i+1)+1]
            self.assertTrue((block >= min(y[i], y[i+1])-1e-12).all())
            self.assertTrue((block <= max(y[i], y[i+1])+1e-12).all())

    def test_group_mean_schema_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "distribution-bin schema"):
            validate_bins(pd.DataFrame({"dataset": ["negative"],
                "mean_probability": [.2], "sample_size": [16642]}), "Figure3d")

    def test_panels_have_seven_curves_with_expected_axes_and_no_bars(self):
        total = 0
        for panel, groups in [("Figure3c", 4), ("Figure3d", 3)]:
            fig = distribution_panel(ROOT / "data/panels/current", panel)
            try:
                self.assertEqual(len(fig.axes), groups)
                for ax in fig.axes:
                    self.assertEqual(len(ax.lines), 1)
                    self.assertEqual(len(ax.patches), 0)
                    self.assertEqual(ax.get_xlim(), (0, 1))
                    self.assertEqual(ax.get_ylim(), (0, 100))
                    self.assertEqual(len(ax.lines[0].get_xdata()), 381)
                    total += 1
            finally:
                plt.close(fig)
        self.assertEqual(total, 7)


if __name__ == "__main__":
    unittest.main()
