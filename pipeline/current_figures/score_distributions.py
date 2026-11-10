"""Current Figure 3c–d: PCHIP through frozen, within-group bin frequencies.

Interpolation is graphical only: no threshold filtering, density estimation,
statistical smoothing, extrapolation, or model execution occurs here.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SPECS = {
    "Figure3c": {
        "title": "Human prediction scores",
        "order": ["sig_pos_ids", "potential_positive", "untrained_proteome", "neg_ids"],
        "labels": ["Significant\npositive", "Potential\npositive", "Untrained\nproteome", "Negative"],
        "colors": ["#D62728", "#2C7FB8", "#6BAED6", "#9AA5B1"],
        "n": [376, 1844, 5640, 9109],
        "positions": [30, 148, 266, 384],
        "width": 84,
    },
    "Figure3d": {
        "title": "Zero-shot mouse prediction",
        "order": ["significant_positive", "potential_positive", "negative"],
        "labels": ["Significant\npositive", "Potential\npositive", "Negative"],
        "colors": ["#FF8000", "#F8B62D", "#D4D4D4"],
        "n": [76, 284, 16642],
        "positions": [30, 186, 342],
        "width": 126,
    },
}
BINS = np.arange(21, dtype=float) / 20


def pchip(x, y, samples_per_interval=20):
    """Shape-preserving cubic Hermite interpolation, without extrapolation."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or len(x) != len(y) or samples_per_interval < 1:
        raise ValueError("PCHIP requires matching knots and positive sampling density")
    h = np.diff(x)
    if not (h > 0).all() or not np.isfinite(y).all():
        raise ValueError("PCHIP knots must be ordered and finite")
    delta = np.diff(y) / h
    slopes = np.zeros_like(y)
    for k in range(1, len(y) - 1):
        if delta[k - 1] * delta[k] > 0:
            w1 = 2 * h[k] + h[k - 1]
            w2 = h[k] + 2 * h[k - 1]
            slopes[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])

    def endpoint(ha, hb, da, db):
        value = ((2 * ha + hb) * da - ha * db) / (ha + hb)
        if np.sign(value) != np.sign(da):
            return 0.0
        if np.sign(da) != np.sign(db) and abs(value) > 3 * abs(da):
            return 3 * da
        return value

    slopes[0] = endpoint(h[0], h[1], delta[0], delta[1])
    slopes[-1] = endpoint(h[-1], h[-2], delta[-1], delta[-2])
    xx, yy = [], []
    for k in range(len(h)):
        t = np.arange(samples_per_interval, dtype=float) / samples_per_interval
        values = ((2*t**3 - 3*t**2 + 1)*y[k]
                  + (t**3 - 2*t**2 + t)*h[k]*slopes[k]
                  + (-2*t**3 + 3*t**2)*y[k + 1]
                  + (t**3 - t**2)*h[k]*slopes[k + 1])
        if not ((values >= min(y[k], y[k + 1]) - 1e-12).all()
                and (values <= max(y[k], y[k + 1]) + 1e-12).all()):
            raise ValueError("Interpolation overshot adjacent frequency knots")
        xx.extend(x[k] + t*h[k])
        yy.extend(values)
    return np.array(xx + [x[-1]]), np.array(yy + [y[-1]])


def validate_bins(table, panel):
    """Validate the complete 20-bin source table; never accept group means."""
    spec = SPECS[panel]
    columns = ["dataset", "n", "bin_left", "bin_right", "bin_center",
               "protein_count", "within_group_percentage", "right_endpoint_inclusive"]
    if list(table.columns) != columns or len(table) != 20 * len(spec["order"]):
        raise ValueError("Expected the current complete distribution-bin schema")
    if set(table.dataset) != set(spec["order"]):
        raise ValueError("Distribution groups differ from the frozen manuscript groups")
    for group, n in zip(spec["order"], spec["n"]):
        block = table.loc[table.dataset.eq(group)].sort_values("bin_left")
        if (len(block) != 20 or not block.n.eq(n).all()
                or not np.allclose(block.bin_left, BINS[:-1], rtol=0, atol=1e-15)
                or not np.allclose(block.bin_right, BINS[1:], rtol=0, atol=1e-15)
                or not np.allclose(block.bin_center, (BINS[:-1]+BINS[1:])/2, rtol=0, atol=1e-15)
                or block.protein_count.sum() != n
                or not np.equal(block.protein_count, np.floor(block.protein_count)).all()
                or (block.protein_count < 0).any()
                or not np.allclose(block.within_group_percentage,
                                   block.protein_count*100/n, rtol=1e-12, atol=1e-12)
                or block.right_endpoint_inclusive.astype(str).str.lower().tolist()
                != ["false"]*19 + ["true"]):
            raise ValueError(f"Invalid complete distribution bins for {group}")
    return table


def distribution_panel(data_dir, panel):
    """Replay one row with unchanged 20-bin knots and seven total group curves."""
    table = validate_bins(pd.read_csv(data_dir / f"{panel}.csv",
                                     float_precision="round_trip"), panel)
    spec = SPECS[panel]
    with plt.rc_context({"font.size": 7, "axes.labelsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "axes.linewidth": .5, "path.simplify": False}):
        fig = plt.figure(figsize=(480/72, 180/72))
        fig.text(30/480, 1-8/180, spec["title"], ha="left", va="center",
                 fontsize=8, fontweight="bold")
        fig.text(6/480, 1-99/180, "Proteins (%)", rotation=90,
                 ha="center", va="center", fontsize=7)
        fig.text(249/480, 1-170/180, "Predicted probability", ha="center",
                 va="center", fontsize=7)
        for group, label, color, n, left in zip(spec["order"], spec["labels"],
                spec["colors"], spec["n"], spec["positions"]):
            block = table.loc[table.dataset.eq(group)].sort_values("bin_left")
            xx, yy = pchip(block.bin_center, block.within_group_percentage)
            ax = fig.add_axes([left/480, 34/180, spec["width"]/480, 94/180])
            ax.plot(xx, yy, color=color, linewidth=1.1, alpha=.92,
                    solid_capstyle="round", solid_joinstyle="round")
            ax.set(xlim=(0, 1), ylim=(0, 100))
            ax.set_xticks([0, .2, .4, .6, .8, 1], ["0", "0.2", "0.4", "0.6", "0.8", "1"])
            ax.set_yticks([0, 50, 100], ["0", "50", "100"])
            ax.tick_params(pad=2, width=.5, length=2.8)
            center = (left + spec["width"]/2)/480
            fig.text(center, 1-27/180, label, ha="center", va="center",
                     fontsize=7, linespacing=1.1)
            fig.text(center, 1-42/180, f"n = {n:,}", ha="center",
                     va="center", fontsize=6.2)
    return fig
