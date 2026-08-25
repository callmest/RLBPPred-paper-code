#!/usr/bin/env python3
"""Replot mouse inference probability bar charts as separate publication figures."""

from pathlib import Path
import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator


COLORS = {
    "significant_positive": "#D62728",
    "potential_positive": "#2C7FB8",
    "negative": "#9AA5B1",
}

LABELS = {
    "significant_positive": "Sig. Positive\n(Train Pos)",
    "potential_positive": "Potential\nPositive",
    "negative": "Negative\n(Train Neg)",
}


def load_data(csv_path: Path):
    order = ["significant_positive", "potential_positive", "negative"]
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "dataset": row["dataset"],
                    "mean_probability": float(row["mean_probability"]),
                    "percentage_positive": float(row["percentage_positive"]),
                    "sample_size": int(row["sample_size"]),
                }
            )
    by_name = {r["dataset"]: r for r in rows}
    return [by_name[name] for name in order if name in by_name]


def apply_style():
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.labelsize": 9.5,
            "axes.labelcolor": "#000000",
            "axes.edgecolor": "#000000",
            "axes.linewidth": 1.1,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.color": "#000000",
            "ytick.color": "#000000",
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "xtick.major.size": 3.8,
            "ytick.major.size": 3.8,
            "xtick.minor.size": 2.4,
            "ytick.minor.size": 2.4,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def style_axis(ax):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig, out_prefix: Path):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_prefix.with_suffix(f".{ext}"))
    plt.close(fig)


def make_mean_probability_plot(rows: list[dict], out_prefix: Path):
    apply_style()
    fig, ax = plt.subplots(figsize=(3.25, 3.05), dpi=300)

    x = range(len(rows))
    names = [r["dataset"] for r in rows]
    colors = [COLORS[n] for n in names]
    xlabels = [LABELS[n] for n in names]

    bars = ax.bar(
        x,
        [r["mean_probability"] for r in rows],
        color=colors,
        edgecolor="#000000",
        linewidth=0.8,
        width=0.70,
        alpha=0.92,
    )
    ax.set_ylabel("Mean predicted probability")
    ax.set_xlabel("Dataset")
    ax.set_xticks(list(x), xlabels)
    ax.set_ylim(0, 1.10)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    style_axis(ax)

    for bar, mean, n in zip(
        bars, [r["mean_probability"] for r in rows], [r["sample_size"] for r in rows]
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(bar.get_height() + 0.018, 1.055),
            f"{mean:.3f}\n(n={n:,})",
            ha="center",
            va="bottom",
            fontsize=7.8,
            color="#000000",
        )

    fig.subplots_adjust(left=0.18, right=0.985, top=0.965, bottom=0.24)
    save_figure(fig, out_prefix)


def make_positive_percentage_plot(rows: list[dict], out_prefix: Path):
    apply_style()
    fig, ax = plt.subplots(figsize=(3.25, 3.05), dpi=300)

    x = range(len(rows))
    names = [r["dataset"] for r in rows]
    colors = [COLORS[n] for n in names]
    xlabels = [LABELS[n] for n in names]

    bars = ax.bar(
        x,
        [r["percentage_positive"] for r in rows],
        color=colors,
        edgecolor="#000000",
        linewidth=0.8,
        width=0.70,
        alpha=0.92,
    )
    ax.set_ylabel("Predicted as RLBP (%)")
    ax.set_xlabel("Dataset")
    ax.set_xticks(list(x), xlabels)
    ax.set_ylim(0, 108)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    style_axis(ax)

    for bar, pct in zip(bars, [r["percentage_positive"] for r in rows]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(bar.get_height() + 1.6, 104),
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7.8,
            color="#000000",
        )

    fig.subplots_adjust(left=0.19, right=0.985, top=0.965, bottom=0.24)
    save_figure(fig, out_prefix)


def main():
    parser = argparse.ArgumentParser(description="Replot mouse probability bar charts")
    parser.add_argument(
        "--input-csv",
        default=str(Path(__file__).resolve().parents[2] / "results" / "inference_mouse_datasets" / "prob_barplot_data.csv"),
        help="Input summary CSV",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[2] / "results" / "inference_mouse_datasets"),
        help="Output directory",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_data(input_csv)
    mean_prefix = output_dir / "prob_barplot_mean_probability_scientific"
    positive_prefix = output_dir / "prob_barplot_positive_percentage_scientific"
    make_mean_probability_plot(rows, mean_prefix)
    make_positive_percentage_plot(rows, positive_prefix)
    print(f"Saved: {mean_prefix}.png/.pdf/.svg")
    print(f"Saved: {positive_prefix}.png/.pdf/.svg")


if __name__ == "__main__":
    main()
