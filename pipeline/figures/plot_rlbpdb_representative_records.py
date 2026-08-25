#!/usr/bin/env python3
"""Create publication previews for representative RLBPPredDB records."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle


COLORS = {
    "ink": "#17242B",
    "muted": "#5E717A",
    "line": "#AAB8BE",
    "light": "#F4F7F8",
    "surface": "#E7EEF1",
    "orange": "#C76549",
    "teal": "#3B8C83",
    "blue": "#627FA8",
    "gray": "#9CA8B2",
    "green": "#4D846D",
}

ROLE_LABELS = {
    "SRSF1": "Co-transcriptional RNA processing",
    "TOP1": "Transcription-induced supercoiling",
    "XRN2": "RNA degradation and termination",
    "DHX9": "RNA:DNA hybrid remodeling",
    "DDX5": "R-loop resolution / RNA helicase",
    "AQR": "Spliceosome-associated regulation",
    "RNASEH1": "RNA degradation in RNA:DNA hybrids",
}
GENES = list(ROLE_LABELS)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def configure_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 200,
        "savefig.dpi": 600,
    })


def rounded(ax, xy, width, height, facecolor, edgecolor=COLORS["line"],
            linewidth=0.8, radius=0.015, zorder=1):
    patch = FancyBboxPatch(
        xy, width, height,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        transform=ax.transAxes, facecolor=facecolor, edgecolor=edgecolor,
        linewidth=linewidth, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def text(ax, x, y, value, size=8, color=None, weight="normal", ha="left", va="center"):
    return ax.text(x, y, value, transform=ax.transAxes, fontsize=size,
                   color=color or COLORS["ink"], fontweight=weight,
                   ha=ha, va=va)


def draw_search(ax, top1, total_records):
    ax.set_axis_off()
    rounded(ax, (0.01, 0.02), 0.98, 0.96, "#FFFFFF", COLORS["line"], radius=0.012)
    ax.add_patch(Rectangle((0.01, 0.86), 0.98, 0.12, transform=ax.transAxes,
                           facecolor="#233A43", edgecolor="none", zorder=2))
    text(ax, 0.035, 0.92, "RLBP", 8, "#E58A68", "bold")
    text(ax, 0.105, 0.92, "RLBPPredDB", 11, "#FFFFFF", "bold")
    text(ax, 0.955, 0.92, "Home     Annotation     Data Access     Help", 6.5,
         "#DCE8EC", ha="right")

    text(ax, 0.04, 0.80, "RLBP Annotation Browser", 12, weight="bold")
    text(ax, 0.04, 0.755, "Search and inspect proteome-scale RLBP predictions", 7, COLORS["muted"])

    controls = [
        (0.04, 0.64, 0.30, "TOP1"),
        (0.355, 0.64, 0.14, "Human"),
        (0.51, 0.64, 0.16, "Reviewed only"),
        (0.685, 0.64, 0.12, "Min 0.85"),
        (0.82, 0.64, 0.13, "Confidence ↓"),
    ]
    for x, y, w, label in controls:
        rounded(ax, (x, y), w, 0.075, "#FFFFFF", COLORS["line"], radius=0.01)
        text(ax, x + 0.012, y + 0.038, label, 7, COLORS["ink"])

    text(ax, 0.04, 0.585,
         f"Loaded {total_records:,} records  |  Matched 1  |  Showing 1–1",
         7, COLORS["muted"])

    x0, y0, w, h = 0.04, 0.17, 0.92, 0.35
    rounded(ax, (x0, y0), w, h, "#FFFFFF", COLORS["line"], radius=0.008)
    widths = np.array([0.12, 0.10, 0.10, 0.235, 0.125, 0.10, 0.09, 0.13])
    widths = widths / widths.sum() * w
    headers = ["UniProt ID", "Species", "Gene", "Protein", "Confidence", "Reviewed", "Length", "Links"]
    values = [
        top1.uniprot_id, top1.species, top1.gene_name, "DNA topoisomerase 1",
        f"{top1.confidence:.6f}", "Yes" if top1.reviewed else "No",
        f"{int(top1.length)} aa", "UniProt  PDB  AF2",
    ]
    ax.add_patch(Rectangle((x0, y0 + 0.19), w, 0.16, transform=ax.transAxes,
                           facecolor=COLORS["surface"], edgecolor="none"))
    current = x0
    for i, (width, header, value) in enumerate(zip(widths, headers, values)):
        if i:
            ax.plot([current, current], [y0, y0 + h], transform=ax.transAxes,
                    color="#D3DDE1", linewidth=0.55)
        text(ax, current + 0.008, y0 + 0.27, header, 5.9, COLORS["ink"], "bold")
        value_color = COLORS["orange"] if header == "Confidence" else COLORS["ink"]
        text(ax, current + 0.008, y0 + 0.095, value, 5.9, value_color,
             "bold" if header in ("UniProt ID", "Confidence") else "normal")
        current += width
    ax.plot([x0, x0 + w], [y0 + 0.19, y0 + 0.19], transform=ax.transAxes,
            color=COLORS["line"], linewidth=0.7)


def draw_record(ax, row):
    ax.set_axis_off()
    rounded(ax, (0.015, 0.02), 0.97, 0.96, "#FFFFFF", COLORS["line"], radius=0.012)
    ax.add_patch(Rectangle((0.015, 0.79), 0.97, 0.19, transform=ax.transAxes,
                           facecolor="#233A43", edgecolor="none"))
    text(ax, 0.055, 0.905, row.gene_name, 18, "#FFFFFF", "bold")
    text(ax, 0.055, 0.835, "DNA topoisomerase 1", 8, "#DCE8EC")
    rounded(ax, (0.71, 0.865), 0.22, 0.055, "#DDEBE5", "#DDEBE5", radius=0.02)
    text(ax, 0.82, 0.893, "Established RLBP", 7, COLORS["green"], "bold", ha="center")

    fields = [
        ("UniProt accession", row.uniprot_id),
        ("Species", "Homo sapiens"),
        ("Entry", row.entry_name),
        ("Reviewed", "Yes" if row.reviewed else "No"),
        ("Protein length", f"{int(row.length)} aa"),
    ]
    for i, (label, value) in enumerate(fields):
        y = 0.70 - i * 0.095
        text(ax, 0.055, y, label, 6.5, COLORS["muted"], "bold")
        text(ax, 0.30, y, str(value), 7.5, COLORS["ink"])

    text(ax, 0.055, 0.205, "Functional annotation", 6.5, COLORS["muted"], "bold")
    annotation = "DNA topoisomerase I; relieves torsional stress generated during transcription."
    text(ax, 0.055, 0.145, "\n".join(textwrap.wrap(annotation, 70)), 7, COLORS["ink"], va="top")

    text(ax, 0.58, 0.705, "RLBPPred probability", 7, COLORS["muted"], "bold")
    text(ax, 0.58, 0.61, f"{row.confidence:.6f}", 20, COLORS["orange"], "bold")
    rounded(ax, (0.58, 0.525), 0.34, 0.035, "#D9E2E6", "#D9E2E6", radius=0.015)
    rounded(ax, (0.58, 0.525), 0.34 * float(row.confidence), 0.035,
            COLORS["orange"], COLORS["orange"], radius=0.015, zorder=3)
    text(ax, 0.58, 0.465, "0", 6, COLORS["muted"])
    text(ax, 0.92, 0.465, "1", 6, COLORS["muted"], ha="right")

    text(ax, 0.58, 0.36, "Cross-references", 7, COLORS["muted"], "bold")
    for i, (label, color) in enumerate([("UniProt", COLORS["teal"]),
                                        ("PDB", COLORS["blue"]),
                                        ("AlphaFold", COLORS["orange"])]):
        x = 0.58 + i * 0.115
        rounded(ax, (x, 0.275), 0.10, 0.055, "#FFFFFF", color, radius=0.01)
        text(ax, x + 0.05, 0.303, label, 6.2, color, "bold", ha="center")
    text(ax, 0.58, 0.19, "Current database fields", 6.2, COLORS["muted"])


def draw_probabilities(ax, records, compact_labels=False):
    genes = GENES[::-1]
    data = records.set_index("gene_name").loc[genes]
    y = np.arange(len(genes))
    values = data["confidence"].to_numpy()
    base = 0.80
    colors = [COLORS["orange"] if value < 0.85 else COLORS["blue"] for value in values]

    ax.hlines(y, base, values, color=colors, linewidth=2.0, alpha=0.9)
    ax.scatter(values, y, s=42, color=colors, edgecolor="#222222", linewidth=0.55, zorder=3)
    ax.axvline(0.85, color=COLORS["muted"], linestyle=(0, (3, 2)), linewidth=0.8)
    ax.text(0.852, len(genes) - 0.35, "0.85 cutoff", fontsize=6.5,
            color=COLORS["muted"], va="top")

    labels = genes if compact_labels else [f"{gene}  |  {ROLE_LABELS[gene]}" for gene in genes]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.6)
    ax.set_xlim(base, 1.005)
    ax.set_xticks([0.80, 0.85, 0.90, 0.95, 1.00])
    ax.set_xlabel("RLBPPred probability")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for yi, value in zip(y, values):
        if value > 0.97:
            label_x, ha = value - 0.004, "right"
        else:
            label_x, ha = value + 0.004, "left"
        ax.text(label_x, yi, f"{value:.3f}", fontsize=6.2,
                ha=ha, va="center", color=COLORS["ink"])


def save(fig, prefix):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{prefix}.{ext}", bbox_inches="tight",
                    dpi=600 if ext == "png" else None, facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    configure_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    db = pd.read_csv(args.db_dir / "rlbp_annotations.csv")
    db["reviewed"] = db["reviewed"].astype(str).str.lower().isin(["true", "1", "yes"])
    top1 = db.loc[db["gene_name"] == "TOP1"].iloc[0]
    representatives = db[db["gene_name"].isin(GENES)].copy()
    representatives["representative_role"] = representatives["gene_name"].map(ROLE_LABELS)
    representatives.to_csv(args.output_dir / "representative_established_rlbps.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    draw_search(ax, top1, len(db))
    save(fig, args.output_dir / "rlbpdb_search_example")

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    draw_record(ax, top1)
    save(fig, args.output_dir / "rlbpdb_top1_record")

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    draw_probabilities(ax, representatives)
    fig.subplots_adjust(left=0.48, right=0.96, bottom=0.16, top=0.96)
    save(fig, args.output_dir / "rlbpdb_established_rlbp_probabilities")

    fig = plt.figure(figsize=(11.0, 7.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.1], hspace=0.18, wspace=0.17)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    draw_search(ax_a, top1, len(db))
    draw_record(ax_b, top1)
    draw_probabilities(ax_c, representatives, compact_labels=True)
    for label, ax in zip("abc", [ax_a, ax_b, ax_c]):
        ax.text(-0.035, 1.02, label, transform=ax.transAxes, fontsize=11,
                fontweight="bold", ha="left", va="bottom")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.08, top=0.97)
    save(fig, args.output_dir / "rlbpdb_representative_records_combined")

    print("Saved previews to", args.output_dir)


if __name__ == "__main__":
    main()
