#!/usr/bin/env python3
"""Plot FPR-calibrated threshold selection for novel RLBP candidate screening."""

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[2]
DATA_DIR = BASE / "results" / "inference_four_datasets"
OUT_DIR = BASE / "results" / "untrained_proteome_analysis"

TARGET_FPRS = [5.0, 1.0, 0.5, 0.1]
MAIN_THRESHOLD = 0.85


def load_probabilities(filename):
    df = pd.read_csv(DATA_DIR / filename)
    return df["probability"].to_numpy(dtype=float)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    neg = load_probabilities("neg_ids_predictions.csv")
    untrained = load_probabilities("untrained_proteome_predictions.csv")
    sig_pos = load_probabilities("sig_pos_ids_predictions.csv")
    potential = load_probabilities("potential_positive_predictions.csv")

    thresholds = np.linspace(0.5, 0.999, 500)
    neg_fpr = np.array([(neg >= t).mean() * 100 for t in thresholds])
    untrained_count = np.array([(untrained >= t).sum() for t in thresholds])

    fpr_rows = []
    for target in TARGET_FPRS:
        threshold = float(np.quantile(neg, 1 - target / 100))
        fpr_rows.append(
            {
                "target_neg_fpr_percent": target,
                "threshold": threshold,
                "untrained_count": int((untrained >= threshold).sum()),
                "untrained_rate_percent": (untrained >= threshold).mean() * 100,
                "sig_pos_rate_percent": (sig_pos >= threshold).mean() * 100,
                "potential_rate_percent": (potential >= threshold).mean() * 100,
            }
        )

    main_row = {
        "threshold": MAIN_THRESHOLD,
        "neg_fpr_percent": (neg >= MAIN_THRESHOLD).mean() * 100,
        "untrained_count": int((untrained >= MAIN_THRESHOLD).sum()),
        "untrained_rate_percent": (untrained >= MAIN_THRESHOLD).mean() * 100,
        "sig_pos_rate_percent": (sig_pos >= MAIN_THRESHOLD).mean() * 100,
        "potential_rate_percent": (potential >= MAIN_THRESHOLD).mean() * 100,
    }

    summary_csv = OUT_DIR / "fpr_threshold_selection_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["selection", "threshold", "neg_fpr_percent", "untrained_count", "sig_pos_rate_percent", "potential_rate_percent"])
        for row in fpr_rows:
            writer.writerow([
                f"Target FPR {row['target_neg_fpr_percent']:.1f}%",
                f"{row['threshold']:.6f}",
                f"{row['target_neg_fpr_percent']:.2f}",
                row["untrained_count"],
                f"{row['sig_pos_rate_percent']:.2f}",
                f"{row['potential_rate_percent']:.2f}",
            ])
        writer.writerow([
            "Adopted threshold",
            f"{main_row['threshold']:.2f}",
            f"{main_row['neg_fpr_percent']:.2f}",
            main_row["untrained_count"],
            f"{main_row['sig_pos_rate_percent']:.2f}",
            f"{main_row['potential_rate_percent']:.2f}",
        ])

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.8,
    })

    blue = "#3F7FB5"
    dark_blue = "#08306B"
    orange = "#D55E00"
    grey = "#6B7280"
    light_grey = "#E5E7EB"

    fig, ax1 = plt.subplots(figsize=(4.8, 3.15), constrained_layout=False)
    ax2 = ax1.twinx()
    one_percent = next(r for r in fpr_rows if abs(r["target_neg_fpr_percent"] - 1.0) < 1e-6)

    fpr_line, = ax1.plot(thresholds, neg_fpr, color=dark_blue, lw=2.0,
                         label="Negative-set FPR")
    count_line, = ax2.plot(thresholds, untrained_count, color=blue, lw=2.0,
                           label="Candidate count")

    ax1.axhline(1.0, color=grey, lw=0.9, ls=":", alpha=0.75)
    ax1.axvline(one_percent["threshold"], color=grey, lw=0.9, ls=":", alpha=0.75)
    ax1.axvline(MAIN_THRESHOLD, color=orange, lw=1.4, ls="--")

    ax1.scatter([MAIN_THRESHOLD], [main_row["neg_fpr_percent"]], s=36, color=orange, zorder=5)
    ax2.scatter([MAIN_THRESHOLD], [main_row["untrained_count"]], s=42, marker="s",
                color=orange, edgecolor="white", linewidth=0.5, zorder=5)

    for row in fpr_rows:
        ax1.scatter([row["threshold"]], [row["target_neg_fpr_percent"]], s=24, color=grey, zorder=4)
    ax2.scatter([one_percent["threshold"]], [one_percent["untrained_count"]], s=32, marker="s",
                color=grey, edgecolor="white", linewidth=0.5, zorder=4)

    ax1.set_xlim(0.5, 1.0)
    ax1.set_ylim(0, max(10.5, neg_fpr.max() * 1.08))
    ax2.set_ylim(0, untrained_count.max() * 1.12)
    ax1.set_xlabel("Probability threshold")
    ax1.set_ylabel("Negative-set FPR (%)")
    ax2.set_ylabel("Candidate count")
    ax1.set_title("FPR-calibrated candidate screening")

    ax1.tick_params(axis="y", colors=dark_blue)
    ax2.tick_params(axis="y", colors=blue)
    ax1.yaxis.label.set_color(dark_blue)
    ax2.yaxis.label.set_color(blue)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.spines["left"].set_color(dark_blue)
    ax2.spines["right"].set_color(blue)
    ax1.grid(axis="y", color=light_grey, lw=0.7)

    from matplotlib.lines import Line2D

    handles = [
        fpr_line,
        count_line,
        Line2D([0], [0], color=orange, lw=1.4, ls="--", marker="o", markersize=4.5,
               label=f"Adopted threshold 0.85: FPR {main_row['neg_fpr_percent']:.2f}%, {main_row['untrained_count']} candidates"),
        Line2D([0], [0], color=grey, lw=0.9, ls=":", marker="o", markersize=4.0,
               label=f"1% FPR-calibrated threshold {one_percent['threshold']:.4f}: {one_percent['untrained_count']} candidates"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.03),
               ncol=2, frameon=False, columnspacing=1.2, handlelength=2.3,
               borderaxespad=0.0)
    fig.subplots_adjust(left=0.13, right=0.86, bottom=0.34, top=0.86)

    out_prefix = OUT_DIR / "fpr_threshold_selection_process"
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out_prefix.with_suffix(f".{ext}"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_prefix}.png/.pdf/.svg")
    print(f"Saved: {summary_csv}")


if __name__ == "__main__":
    main()
