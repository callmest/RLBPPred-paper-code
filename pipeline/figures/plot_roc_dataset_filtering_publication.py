#!/usr/bin/env python3
"""Render ROC comparison figure for dataset filtering analysis."""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "sa_mlp_comparison"

COLORS = [
    "#1F77B4",  # blue
    "#D62728",  # red
]


def load_curves(csv_path: Path) -> OrderedDict[str, dict[str, list[float] | float]]:
    curves: OrderedDict[str, dict[str, list[float] | float]] = OrderedDict()
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"label", "auroc", "fpr", "tpr"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"Missing required ROC columns: {missing_str}")

        for row in reader:
            label = row["label"].strip()
            if label not in curves:
                curves[label] = {"auroc": float(row["auroc"]), "fpr": [], "tpr": []}
            curves[label]["fpr"].append(float(row["fpr"]))
            curves[label]["tpr"].append(float(row["tpr"]))

    return OrderedDict(
        sorted(curves.items(), key=lambda item: item[1]["auroc"], reverse=True)
    )


def prepare_points(fpr: list[float], tpr: list[float]) -> np.ndarray:
    points = np.column_stack([np.asarray(fpr, dtype=float), np.asarray(tpr, dtype=float)])
    points = np.clip(points, 0.0, 1.0)

    order = np.argsort(points[:, 0], kind="mergesort")
    points = points[order]
    points[:, 0] = np.maximum.accumulate(points[:, 0])
    points[:, 1] = np.maximum.accumulate(points[:, 1])

    if len(points) == 0 or not np.allclose(points[0], [0.0, 0.0], atol=1e-12):
        points = np.vstack([[0.0, 0.0], points])
    else:
        points[0] = [0.0, 0.0]

    if not np.allclose(points[-1], [1.0, 1.0], atol=1e-12):
        points = np.vstack([points, [1.0, 1.0]])
    else:
        points[-1] = [1.0, 1.0]

    return points


def display_label(label: str) -> str:
    normalized = label.replace("SaProt-AF2", "SaProt-650M-AF2")
    if normalized.startswith("Original Dataset"):
        return normalized.replace("Original Dataset", "Filtered Dataset", 1)
    if normalized.startswith("Filtered Dataset"):
        return normalized.replace("Filtered Dataset", "Unfiltered Dataset", 1)
    return normalized


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 10.5,
            "text.color": "#000000",
            "axes.labelsize": 12.5,
            "axes.labelcolor": "#000000",
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.titlecolor": "#000000",
            "axes.edgecolor": "#000000",
            "axes.linewidth": 1.0,
            "xtick.color": "#000000",
            "ytick.color": "#000000",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.antialiased": True,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "svg.fonttype": "none",
        }
    )


def plot_roc(csv_path: Path, out_path: Path, png_out: Path, title: str) -> None:
    curves = load_curves(csv_path)
    configure_style()

    fig, ax = plt.subplots(figsize=(7.0, 7.0), dpi=200)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, pad=10)

    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    ax.grid(False)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color="#A0A8B5",
        linewidth=1.0,
        linestyle=(0, (5, 4)),
        label="Random (0.5000)",
        zorder=1,
    )

    for idx, (label, data) in enumerate(curves.items()):
        points = prepare_points(
            data["fpr"],  # type: ignore[arg-type]
            data["tpr"],  # type: ignore[arg-type]
        )
        color = COLORS[idx % len(COLORS)]
        ax.step(
            points[:, 0],
            points[:, 1],
            where="post",
            color=color,
            linewidth=1.8,
            alpha=0.96,
            label=f"{display_label(label)} ({data['auroc']:.4f})",
            zorder=3 + idx,
        )

    legend = ax.legend(
        loc="lower right",
        bbox_to_anchor=(0.985, 0.02),
        ncol=1,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#D0D7DE",
        borderpad=0.7,
        labelspacing=0.45,
        handlelength=2.2,
        handletextpad=0.55,
        columnspacing=0.9,
        fontsize=10.2,
        title="Classifier (AUROC)",
        title_fontsize=10.6,
    )
    legend.get_frame().set_linewidth(0.9)
    legend.get_title().set_color("#000000")
    for text in legend.get_texts():
        text.set_color("#000000")

    fig.subplots_adjust(left=0.11, right=0.98, top=0.90, bottom=0.11)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    png_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    fig.savefig(png_out, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved SVG: {out_path}")
    print(f"Saved PNG: {png_out}")


def parse_args() -> argparse.Namespace:
    default_csv = RESULTS_DIR / "roc_comparison.csv"
    default_out = RESULTS_DIR / "roc_comparison.svg"
    parser = argparse.ArgumentParser(description="Create ROC comparison figure.")
    parser.add_argument("--csv", type=Path, default=default_csv, help="Input ROC CSV file.")
    parser.add_argument("--out", type=Path, default=default_out, help="Output SVG file.")
    parser.add_argument(
        "--png-out",
        type=Path,
        default=None,
        help="Optional PNG output path. Defaults to the SVG path with a .png suffix.",
    )
    parser.add_argument(
        "--title",
        default="ROC Curves - Dataset Filtering Comparison",
        help="Figure title.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    png_out = args.png_out or args.out.with_suffix(".png")
    plot_roc(
        csv_path=args.csv,
        out_path=args.out,
        png_out=png_out,
        title=args.title,
    )


if __name__ == "__main__":
    main()
