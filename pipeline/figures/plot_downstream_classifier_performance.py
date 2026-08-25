#!/usr/bin/env python3
"""Plot downstream classifier performance across upstream protein representations."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
OUTPUT_DIR = RESULTS_DIR / "downstream_classifier_performance"
OUT_PREFIX = OUTPUT_DIR / "downstream_classifier_performance_heatmap"


CLASSIFIERS = [
    ("LogisticRegression", "Logistic regression"),
    ("SVM", "SVM"),
    ("RandomForest", "Random forest"),
    ("XGBoost", "XGBoost"),
    ("LightGBM", "LightGBM"),
    ("KNN", "KNN"),
    ("GaussianNB", "Gaussian NB"),
    ("MLP", "MLP"),
    ("RNN", "RNN"),
    ("LSTM", "LSTM"),
]

UPSTREAM_ORDER = [
    "esmc-300m",
    "esmc_600m",
    "pst_t33",
    "pst_t33_so",
    "saprot-650m-af2",
    "saprot-650m-pdb",
]

UPSTREAM_LABELS = {
    "esmc-300m": "ESM-C\n300M",
    "esmc_600m": "ESM-C\n600M",
    "pst_t33": "PST\nT33",
    "pst_t33_so": "PST\nT33-SO",
    "saprot-650m-af2": "SaProt\nAF2",
    "saprot-650m-pdb": "SaProt\nPDB",
}

METRICS = ["AUROC", "AUPRC", "Precision", "Recall", "F1", "MCC"]

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "paper_blue",
    ["#F7FBFF", "#D6E6F4", "#8FBAD9", "#3F7FB5", "#08306B"],
)


def load_long_table():
    frames = []
    for file_stem, label in CLASSIFIERS:
        path = RESULTS_DIR / f"{file_stem}.csv"
        df = pd.read_csv(path)
        df["Classifier"] = label
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data["Upstream"] = pd.Categorical(data["Upstream"], categories=UPSTREAM_ORDER, ordered=True)
    data["Classifier"] = pd.Categorical(
        data["Classifier"],
        categories=[label for _, label in CLASSIFIERS],
        ordered=True,
    )
    return data.sort_values(["Classifier", "Upstream"])


def draw_heatmap(ax, data, metric, show_ylabels=True, show_xlabels=True):
    matrix = (
        data.pivot(index="Classifier", columns="Upstream", values=metric)
        .loc[[label for _, label in CLASSIFIERS], UPSTREAM_ORDER]
    )
    arr = matrix.to_numpy(dtype=float)
    im = ax.imshow(arr, cmap=HEATMAP_CMAP, vmin=0, vmax=1, aspect="auto")

    ax.set_title(metric, fontsize=11, fontweight="bold", pad=8)
    ax.set_xticks(np.arange(len(UPSTREAM_ORDER)))
    if show_xlabels:
        ax.set_xticklabels([UPSTREAM_LABELS[x] for x in UPSTREAM_ORDER], fontsize=8)
    else:
        ax.set_xticklabels([])
    ax.set_yticks(np.arange(len(CLASSIFIERS)))
    if show_ylabels:
        ax.set_yticklabels([label for _, label in CLASSIFIERS], fontsize=8)
    else:
        ax.set_yticklabels([])

    ax.tick_params(axis="both", length=0)
    ax.set_xticks(np.arange(-0.5, len(UPSTREAM_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CLASSIFIERS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)

    threshold = 0.62 if metric in {"AUROC", "AUPRC", "Recall", "MCC"} else 0.42
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            value = arr[i, j]
            color = "white" if value >= threshold else "#222222"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color=color)

    best_pos = np.unravel_index(np.nanargmax(arr), arr.shape)
    ax.add_patch(
        plt.Rectangle(
            (best_pos[1] - 0.5, best_pos[0] - 0.5),
            1,
            1,
            fill=False,
            edgecolor="#D55E00",
            linewidth=2.2,
        )
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#333333")
    return im


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_long_table()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )

    data.to_csv(OUT_PREFIX.with_suffix(".merged.csv"), index=False)

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(6.7, 4.9), constrained_layout=False)
        image = draw_heatmap(ax, data, metric, show_ylabels=True, show_xlabels=True)
        cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
        cbar.set_label("Metric value", fontsize=8)
        cbar.ax.tick_params(labelsize=8, length=3)
        fig.subplots_adjust(left=0.24, right=0.92, bottom=0.14, top=0.88)

        metric_slug = metric.lower()
        out_png = OUTPUT_DIR / f"downstream_classifier_performance_{metric_slug}_heatmap.png"
        out_pdf = OUTPUT_DIR / f"downstream_classifier_performance_{metric_slug}_heatmap.pdf"
        out_svg = OUTPUT_DIR / f"downstream_classifier_performance_{metric_slug}_heatmap.svg"
        fig.savefig(out_png, dpi=600, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")
        fig.savefig(out_svg, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_png}")
        print(f"Saved: {out_pdf}")
        print(f"Saved: {out_svg}")

    print(f"Saved: {OUT_PREFIX.with_suffix('.merged.csv')}")


if __name__ == "__main__":
    main()
