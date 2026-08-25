#!/usr/bin/env python3
"""
Plot original vs filtered dataset ROC curves for SaProt-650M-AF2 + MLP.
Saves SVG + raw CSV source data.
"""
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results"
OUTPUT_DIR = RESULTS_ROOT / "sa_mlp_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load original ROC data from CSV ──────────────────────────────────────────
orig_fpr, orig_tpr = [], []
orig_auroc = None
with open(RESULTS_ROOT / "saprot-650m-af2" / "roc_curves.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if "MLP" in row["label"]:
            orig_fpr.append(float(row["fpr"]))
            orig_tpr.append(float(row["tpr"]))
            orig_auroc = float(row["auroc"])

# ── Load filtered ROC data from val_predictions.csv ──────────────────────────
labels, probs = [], []
with open(RESULTS_ROOT / "sa_mlp_filter" / "val_predictions.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        labels.append(int(row["label"]))
        probs.append(float(row["probability"]))

fpr_arr, tpr_arr, _ = roc_curve(labels, probs)
filter_fpr = fpr_arr.tolist()
filter_tpr = tpr_arr.tolist()
filter_auroc = round(roc_auc_score(labels, probs), 4)

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(orig_fpr, orig_tpr, lw=2, color="#2171B5",
        label=f"Original Dataset (AUC={orig_auroc:.4f})")
ax.plot(filter_fpr, filter_tpr, lw=2, color="#CB181D",
        label=f"Filtered Dataset (AUC={filter_auroc:.4f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")

ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.02)
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — SaProt-650M-AF2 + MLP: Original vs Filtered Dataset", fontsize=13)
ax.legend(loc="lower right", fontsize=10, framealpha=0.7)
ax.grid(True, alpha=0.3)

fig.tight_layout()
svg_path = OUTPUT_DIR / "roc_comparison.svg"
fig.savefig(svg_path, format="svg")
plt.close(fig)
print(f"ROC plot saved to {svg_path}")

# ── Save source CSV ──────────────────────────────────────────────────────────
csv_path = OUTPUT_DIR / "roc_comparison.csv"
with open(csv_path, "w") as f:
    f.write("label,auroc,fpr,tpr\n")
    for fp, tp in zip(orig_fpr, orig_tpr):
        f.write(f"Original Dataset (SaProt-AF2 + MLP),{orig_auroc:.6f},{fp:.6f},{tp:.6f}\n")
    for fp, tp in zip(filter_fpr, filter_tpr):
        f.write(f"Filtered Dataset (SaProt-AF2 + MLP),{filter_auroc:.6f},{fp:.6f},{tp:.6f}\n")

print(f"Source data saved to {csv_path}")
