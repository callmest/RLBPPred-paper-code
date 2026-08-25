#!/usr/bin/env python3
"""Create the publication combined sampling-stability figure with panel labels."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = PROJECT_ROOT / "results/sampling_stability"


COLORS = {
    "AUROC": "#0072B2",
    "AUPRC": "#D55E00",
    "F1": "#009E73",
    "MCC": "#CC79A7",
}
MARKERS = {
    "AUROC": "o",
    "AUPRC": "s",
    "F1": "^",
    "MCC": "D",
}


def apply_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def plot_metric_group(ax, df, x_col, x_labels, metrics, ylabel, xlabel, ylim, panel_label):
    x = range(len(df))
    for metric in metrics:
        ax.plot(
            x,
            df[metric.lower()],
            color=COLORS[metric],
            marker=MARKERS[metric],
            linewidth=1.8,
            markersize=5.5,
            markerfacecolor="white",
            markeredgewidth=1.2,
            label=metric,
        )

    ax.set_xticks(list(x), labels=x_labels)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3, width=0.8)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=len(metrics),
        frameon=False,
        handlelength=1.8,
        columnspacing=1.4,
        borderaxespad=0,
    )
    ax.text(
        -0.16,
        1.10,
        panel_label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def main():
    apply_style()

    ratio_df = pd.read_csv(RESULT_DIR / "sampling_ratio_metrics.csv").sort_values("neg_ratio")
    round_df = pd.read_csv(RESULT_DIR / "ensemble_round_metrics.csv").sort_values("n_rounds")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))

    plot_metric_group(
        axes[0, 0],
        ratio_df,
        "neg_ratio",
        [f"1:{int(v)}" for v in ratio_df["neg_ratio"]],
        ["AUROC", "AUPRC"],
        "Ranking metric",
        "Positive:negative sampling ratio",
        (0.915, 1.000),
        "a",
    )
    plot_metric_group(
        axes[0, 1],
        ratio_df,
        "neg_ratio",
        [f"1:{int(v)}" for v in ratio_df["neg_ratio"]],
        ["F1", "MCC"],
        "Threshold-dependent metric",
        "Positive:negative sampling ratio",
        (0.55, 0.90),
        "b",
    )
    plot_metric_group(
        axes[1, 0],
        round_df,
        "n_rounds",
        [str(int(v)) for v in round_df["n_rounds"]],
        ["AUROC", "AUPRC"],
        "Ranking metric",
        "Number of ensemble rounds",
        (0.920, 1.000),
        "c",
    )
    plot_metric_group(
        axes[1, 1],
        round_df,
        "n_rounds",
        [str(int(v)) for v in round_df["n_rounds"]],
        ["F1", "MCC"],
        "Threshold-dependent metric",
        "Number of ensemble rounds",
        (0.56, 0.66),
        "d",
    )

    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.09, top=0.94, wspace=0.24, hspace=0.55)

    prefix = RESULT_DIR / "sampling_stability_combined_publication"
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(f"{prefix}.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)
    print(f"Saved combined figure to {prefix}.png/.pdf/.svg")


if __name__ == "__main__":
    main()
