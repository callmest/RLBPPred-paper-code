#!/usr/bin/env python3
"""Plot the confusion matrix for the best SaProt-650M-AF2 + MLP model."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_JSON = PROJECT_ROOT / "results/saprot-650m-af2/results.json"
OUTPUT_DIR = PROJECT_ROOT / "results/saprot-650m-af2"


def main():
    with open(RESULT_JSON) as f:
        metrics = json.load(f)["saprot-650m-af2"]["MLP"]

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    # The filtered validation split used in the main experiment contains
    # 75 positives and 1,821 negatives. The integer matrix below is recovered
    # from the reported recall, precision, and accuracy values.
    tp = 71
    fn = 4
    fp = 92
    tn = 1729
    matrix = np.array([[tn, fp], [fn, tp]], dtype=int)
    row_pct = matrix / matrix.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(3.45, 2.75))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())

    classes = ["Non-RLBP", "RLBP"]
    ax.set_xticks([0, 1], labels=classes)
    ax.set_yticks([0, 1], labels=classes)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    for i in range(2):
        for j in range(2):
            value = matrix[i, j]
            pct = row_pct[i, j]
            color = "white" if value > matrix.max() * 0.45 else "black"
            ax.text(
                j,
                i,
                f"{value:,}\n({pct:.1f}%)",
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )

    ax.set_title("SaProt-650M-AF2 + MLP", fontsize=8.5, pad=5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Count", fontsize=7.5)
    cbar.ax.tick_params(labelsize=7)

    ax.tick_params(axis="both", labelsize=8, length=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("#222222")
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.subplots_adjust(left=0.32, bottom=0.19, right=0.86, top=0.85)

    prefix = OUTPUT_DIR / "saprot_af2_mlp_confusion_matrix"
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(f"{prefix}.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)

    with open(f"{prefix}.csv", "w") as f:
        f.write("true_label,predicted_label,count,row_percentage\n")
        for i, true_label in enumerate(classes):
            for j, pred_label in enumerate(classes):
                f.write(f"{true_label},{pred_label},{matrix[i,j]},{row_pct[i,j]:.4f}\n")

    with open(f"{prefix}.json", "w") as f:
        json.dump(
            {
                "model": "SaProt-650M-AF2 + MLP",
                "threshold": 0.5,
                "matrix_order": {
                    "rows_true": classes,
                    "columns_predicted": classes,
                },
                "confusion_matrix": matrix.tolist(),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "metrics": metrics,
            },
            f,
            indent=2,
        )

    print(f"Saved confusion matrix to {prefix}.png/.pdf/.svg")


if __name__ == "__main__":
    main()
