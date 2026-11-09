# Historical 342-candidate figure replay; NOT the current manuscript default.
#!/usr/bin/env python3
"""Reproduce RLBPPred computational figure panels from panel-level CSV files.

The outputs are analysis-grade panel reconstructions.  Illustrator schematics,
wet-lab images, electrostatic-surface renders, and browser screenshots are
documented separately because they are not generated from tabular source data.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = ["#3B6FB6", "#D95F5F", "#45A17A", "#8C6BB1", "#E39C37", "#6C7A89"]
GROUP_ORDER = ["Novel predicted RLBPs", "Known RLBPs", "Non-RLBPs"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true", help="Generate all computational panels.")
    selection.add_argument("--figure", help="Generate one panel, e.g. Fig3e or SupFig5d.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/panels"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load(data_dir: Path, name: str) -> pd.DataFrame:
    path = data_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing panel source data: {path}")
    return pd.read_csv(path)


def finish(fig: plt.Figure, output_dir: Path, panel: str, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{panel}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{panel}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def heatmap(ax: plt.Axes, values: np.ndarray, rows: list[str], cols: list[str],
            title: str, fmt: str = ".3f") -> None:
    image = ax.imshow(values, cmap="Blues", aspect="auto", vmin=np.nanmin(values), vmax=np.nanmax(values))
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.set_title(title, fontweight="bold")
    midpoint = (np.nanmin(values) + np.nanmax(values)) / 2
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            ax.text(j, i, format(value, fmt), ha="center", va="center",
                    fontsize=7, color="white" if value > midpoint else "black")
    plt.colorbar(image, ax=ax, fraction=0.035, pad=0.02)


def plot_upset(df: pd.DataFrame, id_column: str, set_columns: list[str], title: str) -> plt.Figure:
    matrix = df[set_columns].fillna(0).astype(int)
    counts = matrix.groupby(set_columns, sort=False).size().sort_values(ascending=False)
    counts = counts[counts.index.map(lambda bits: any(bits))]
    set_sizes = matrix.sum(axis=0)

    fig = plt.figure(figsize=(max(8, 0.38 * len(counts)), 6.3))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.2, 4.8], height_ratios=[3.1, 2.2],
                            wspace=0.08, hspace=0.05)
    ax_blank = fig.add_subplot(grid[0, 0]); ax_blank.axis("off")
    ax_top = fig.add_subplot(grid[0, 1])
    ax_left = fig.add_subplot(grid[1, 0])
    ax_matrix = fig.add_subplot(grid[1, 1])

    x = np.arange(len(counts))
    ax_top.bar(x, counts.values, color="#30343B", width=0.72)
    for i, value in enumerate(counts.values):
        ax_top.text(i, value, f"{value:,}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax_top.set_ylabel("Intersection size")
    ax_top.set_xticks([])
    ax_top.spines[["top", "right"]].set_visible(False)

    y = np.arange(len(set_columns))
    ax_left.barh(y, set_sizes.values, color="#6F879C")
    ax_left.invert_xaxis()
    ax_left.set_yticks(y, set_columns)
    ax_left.set_xlabel("Set size")
    ax_left.spines[["top", "left"]].set_visible(False)

    ax_matrix.set_xlim(-0.5, len(counts) - 0.5)
    ax_matrix.set_ylim(len(set_columns) - 0.5, -0.5)
    for col, bits in enumerate(counts.index):
        active = [row for row, bit in enumerate(bits) if bit]
        ax_matrix.scatter([col] * len(set_columns), y, s=22, color="#D8DDE2", zorder=1)
        if active:
            ax_matrix.plot([col, col], [min(active), max(active)], color="#30343B", lw=1.6, zorder=2)
            ax_matrix.scatter([col] * len(active), active, s=30, color="#30343B", zorder=3)
    ax_matrix.set_yticks(y, [""] * len(y)); ax_matrix.set_xticks([])
    ax_matrix.spines[:].set_visible(False)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    return fig


def fig1b(data_dir: Path) -> plt.Figure:
    df = load(data_dir, "Figure1b")
    return plot_upset(df, "Gene", list(df.columns[1:]), "Five-study RLBP overlap")


def fig2b(data_dir: Path) -> plt.Figure:
    df = load(data_dir, "Figure2b").set_index("Classifier")
    fig, ax = plt.subplots(figsize=(8.2, 6.3))
    heatmap(ax, df.to_numpy(float), list(df.index), list(df.columns), "Validation AUPRC", ".3f")
    return fig


def plot_roc_panel(data_dir: Path, sheet: str, title: str) -> plt.Figure:
    df = load(data_dir, sheet)
    fig, ax = plt.subplots(figsize=(5.4, 5.1))
    for color, (label, group) in zip(COLORS, df.groupby("label", sort=False)):
        auc = float(group["auroc"].iloc[0])
        ax.plot(group["fpr"], group["tpr"], lw=2, color=color, label=f"{label} (AUROC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="#888888", lw=1, label="Random (0.5000)")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate", ylabel="True positive rate", title=title)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def probability_bar(data_dir: Path, sheet: str, value: str, ylabel: str, title: str) -> plt.Figure:
    df = load(data_dir, sheet)
    labels = [str(v).replace("_", " ") for v in df["dataset"]]
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    bars = ax.bar(range(len(df)), df[value], color=COLORS[:len(df)])
    ax.set_xticks(range(len(df)), labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val, n in zip(bars, df[value], df["sample_size"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{val:.3g}\n(n={int(n):,})",
                ha="center", va="bottom", fontsize=8)
    return fig


def fig3e(data_dir: Path) -> plt.Figure:
    df = load(data_dir, "Figure3e").copy()
    df = df[df["rank"] <= 5].copy()
    df["score"] = -np.log10(np.clip(pd.to_numeric(df["fdr_bh"]), 1e-300, None))
    df["label"] = df["term_name"].astype(str).str.lstrip("; ")
    df = df.sort_values(["source", "rank"], ascending=[True, False])
    palette = dict(zip(df["source"].drop_duplicates(), COLORS))
    fig, ax = plt.subplots(figsize=(9, 7.2))
    y = np.arange(len(df))
    ax.barh(y, df["score"], color=[palette[v] for v in df["source"]])
    ax.set_yticks(y, df["label"], fontsize=8)
    ax.set_xlabel("−log10(BH-adjusted P value)")
    ax.set_title("Functional enrichment of high-confidence candidates", fontweight="bold")
    handles = [plt.Line2D([0], [0], color=color, lw=6, label=source) for source, color in palette.items()]
    ax.legend(handles=handles, frameon=False, ncol=3, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def grouped_bar(df: pd.DataFrame, category: str, group: str, value: str,
                title: str, ylabel: str) -> plt.Figure:
    categories = list(dict.fromkeys(df[category].astype(str)))
    groups = [g for g in GROUP_ORDER if g in set(df[group])] or list(dict.fromkeys(df[group].astype(str)))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(categories)); width = 0.8 / len(groups)
    for i, name in enumerate(groups):
        subset = df[df[group] == name].set_index(category).reindex(categories)
        ax.bar(x - 0.4 + width/2 + i*width, subset[value], width, label=name, color=COLORS[i])
    ax.set_xticks(x, categories, rotation=20, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title, fontweight="bold")
    ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    return fig


def fig5c(data_dir: Path) -> plt.Figure:
    df = load(data_dir, "Figure5c_raw")
    groups = [g for g in GROUP_ORDER if g in set(df["group_label"])]
    arrays = [pd.to_numeric(df.loc[df["group_label"] == g, "disorder_percent"], errors="coerce").dropna() for g in groups]
    fig, ax = plt.subplots(figsize=(6.4, 5))
    parts = ax.violinplot(arrays, showmedians=True, showextrema=False)
    for body, color in zip(parts["bodies"], COLORS):
        body.set_facecolor(color); body.set_alpha(0.75)
    ax.set_xticks(range(1, len(groups)+1), groups, rotation=15, ha="right")
    ax.set_ylabel("Disordered residues (%)"); ax.set_title("Intrinsic disorder", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def fig5d(data_dir: Path) -> plt.Figure:
    df = load(data_dir, "Figure5d").copy()
    df["E-value"] = pd.to_numeric(df["E-value"])
    df["score"] = -np.log10(df["E-value"])
    df = df.sort_values("Rank", ascending=False)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    colors = ["#3B6FB6" if x == "Yes" else "#B8C1C8" for x in df["Significant (E<0.05)"]]
    ax.barh(df["Consensus"], df["score"], color=colors)
    ax.axvline(-math.log10(0.05), ls="--", color="#555555", lw=1)
    ax.set_xlabel("−log10(E-value)"); ax.set_title("Candidate sequence motifs", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def supfig1a(data_dir: Path) -> plt.Figure:
    df = load(data_dir, "SuppFigure1a")
    columns = [c for c in df.columns if c not in {"Protein ID", "Human Proteome"}]
    return plot_upset(df, "Protein ID", columns, "Negative-set exclusion sources")


def supfig1b(data_dir: Path) -> plt.Figure:
    df = load(data_dir, "SuppFigure1b").copy()
    df["fdr_bh"] = pd.to_numeric(df["fdr_bh"])
    selected = (df.sort_values("fdr_bh").groupby(["direction", "source"], sort=False).head(5).copy())
    selected["score"] = -np.log10(np.clip(selected["fdr_bh"], 1e-300, None))
    selected.loc[selected["direction"] == "negative", "score"] *= -1
    selected = selected.sort_values("score")
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#3B6FB6" if d == "positive" else "#D95F5F" for d in selected["direction"]]
    ax.barh(range(len(selected)), selected["score"], color=colors)
    ax.set_yticks(range(len(selected)), selected["term_name"], fontsize=7)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Signed −log10(FDR): positive class ← / → negative class")
    ax.set_title("GO enrichment of benchmark classes", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def supfig1c(data_dir: Path) -> plt.Figure:
    df = load(data_dir, "SuppFigure1c")
    df = df[(df["direction"] == "positive") & (df["source"] == "INTERPRO")].copy()
    df["fdr_bh"] = pd.to_numeric(df["fdr_bh"])
    df = df.nsmallest(15, "fdr_bh").sort_values("fdr_bh", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(df["term_name"], -np.log10(np.clip(df["fdr_bh"], 1e-300, None)), color="#3B6FB6")
    ax.set_xlabel("−log10(FDR)"); ax.set_title("Positive-set InterPro enrichment", fontweight="bold")
    ax.tick_params(axis="y", labelsize=8); ax.spines[["top", "right"]].set_visible(False)
    return fig


def supfig2a(data_dir: Path) -> plt.Figure:
    df = load(data_dir, "SuppFigure2a")
    reps = list(dict.fromkeys(df["Representation"])); subsets = ["full", "sampled"]
    fig, axes = plt.subplots(len(reps), 2, figsize=(9, 3.2*len(reps)), squeeze=False)
    for row, rep in enumerate(reps):
        for col, subset in enumerate(subsets):
            ax = axes[row, col]; part = df[(df["Representation"] == rep) & (df["Subset"] == subset)]
            for label, color in [("negative", "#AAB2B8"), ("positive", "#D95F5F")]:
                points = part[part["Label"] == label]
                ax.scatter(points["UMAP_1"], points["UMAP_2"], s=3, alpha=0.45,
                           color=color, label=label, rasterized=True)
            ax.set_title(f"{rep} — {subset}", fontsize=9, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].legend(frameon=False, markerscale=2)
    fig.suptitle("Embedding-space separation", fontweight="bold")
    fig.tight_layout()
    return fig


def supfig2b(data_dir: Path) -> plt.Figure:
    df = load(data_dir, "SuppFigure2b")
    metrics = ["AUROC", "F1", "MCC", "Precision", "Recall"]
    reps = list(dict.fromkeys(df["Representation"])); classifiers = list(dict.fromkeys(df["Classifier"]))
    fig, axes = plt.subplots(1, len(metrics), figsize=(22, 6), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        pivot = df.pivot(index="Classifier", columns="Representation", values=metric).reindex(index=classifiers, columns=reps)
        heatmap(ax, pivot.to_numpy(float), classifiers, reps, metric, ".2f")
    fig.tight_layout()
    return fig


def supfig2c(data_dir: Path) -> plt.Figure:
    ratio = load(data_dir, "SuppFigure2c_ratio"); rounds = load(data_dir, "SuppFigure2c_rounds")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for metric, color in [("auroc", COLORS[0]), ("auprc", COLORS[1])]:
        axes[0].plot(ratio["neg_ratio"], ratio[metric], marker="o", label=metric.upper(), color=color)
        axes[1].plot(rounds["n_rounds"], rounds[metric], marker="o", label=metric.upper(), color=color)
    axes[0].set(xlabel="Negative:positive ratio", ylabel="Score", title="Sampling ratio")
    axes[1].set(xlabel="Ensemble rounds", ylabel="Score", title="Number of rounds")
    for ax in axes: ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Sampling robustness", fontweight="bold")
    return fig


def supfig2d(data_dir: Path) -> plt.Figure:
    df = load(data_dir, "SuppFigure2d")
    fig, ax = plt.subplots(figsize=(7, 4.5)); x = np.arange(len(df)); width = 0.35
    ax.bar(x-width/2, df["auroc"], width, label="AUROC", color=COLORS[0])
    ax.bar(x+width/2, df["auprc"], width, label="AUPRC", color=COLORS[1])
    ax.set_xticks(x, df["strategy"], rotation=18, ha="right"); ax.set_ylim(0.85, 1.0)
    ax.set_title("Negative-undersampling robustness", fontweight="bold"); ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def supfig3a(data_dir: Path) -> plt.Figure:
    df = load(data_dir, "SuppFigure3a")
    labels = ["Non-RLBP", "RLBP"]
    matrix = df.pivot(index="true_label", columns="predicted_label", values="count").reindex(index=labels, columns=labels).to_numpy()
    fig, ax = plt.subplots(figsize=(4.8, 4.4)); image = ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2): ax.text(j, i, f"{int(matrix[i,j]):,}", ha="center", va="center", fontsize=12)
    ax.set_xticks(range(2), labels); ax.set_yticks(range(2), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Threshold 0.5", fontweight="bold")
    plt.colorbar(image, ax=ax, fraction=0.045)
    return fig


def supfig3b(data_dir: Path) -> plt.Figure:
    df = load(data_dir, "SuppFigure3b")
    fig, ax = plt.subplots(figsize=(7.5, 4.6)); twin = ax.twinx()
    ax.plot(df["threshold"], df["neg_rate"], color=COLORS[1], label="Empirical FPR")
    twin.plot(df["threshold"], df["untrained_count"], color=COLORS[0], label="Candidates")
    for value, style in [(0.85, "--"), (0.8583, ":")]: ax.axvline(value, color="#333333", ls=style, lw=1)
    ax.set(xlabel="Probability threshold", ylabel="Negative-set pass rate (%)")
    twin.set_ylabel("Untrained-proteome candidates")
    ax.set_title("False-positive-rate-calibrated screening", fontweight="bold")
    lines = ax.lines[:1] + twin.lines[:1]; ax.legend(lines, [l.get_label() for l in lines], frameon=False)
    return fig


def supfig5b(data_dir: Path) -> plt.Figure:
    df = load(data_dir, "SuppFigure5b")
    categories = list(dict.fromkeys(df["Category"])); groups = list(dict.fromkeys(df["Group"]))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, group in enumerate(groups):
        part = df[df["Group"] == group].set_index("Category").reindex(categories)
        ax.scatter(np.arange(len(categories)), [i]*len(categories), s=part["Count"]*18,
                   c=part["-log10(P)"], cmap="viridis", vmin=0, vmax=df["-log10(P)"].max(), edgecolor="black")
        for x, count, odds in zip(range(len(categories)), part["Count"], part["Odds ratio"]):
            ax.text(x, i, f"{int(count)}\nOR {odds:.1f}", ha="center", va="center", fontsize=7, color="white")
    ax.set_xticks(range(len(categories)), categories, rotation=20, ha="right"); ax.set_yticks(range(len(groups)), groups)
    ax.set_title("Category-specific EpiFactors enrichment", fontweight="bold")
    return fig


def supfig5c(data_dir: Path) -> plt.Figure:
    df = load(data_dir, "SuppFigure5c").sort_values("Total motifs")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.barh(df["Category"], df["Total motifs"], color="#C9D3DB", label="Total motifs")
    ax.barh(df["Category"], df["Significant motifs (E<0.05)"], color=COLORS[0], label="E < 0.05")
    ax.set_xlabel("Motif count"); ax.set_title("Motif-function summary", fontweight="bold"); ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def supfig5d(data_dir: Path) -> plt.Figure:
    df = load(data_dir, "SuppFigure5d_summary")
    aa_order = list(dict.fromkeys(df["amino_acid"])); groups = [g for g in GROUP_ORDER if g in set(df["group_label"])]
    angles = np.linspace(0, 2*np.pi, len(aa_order), endpoint=False); closed = np.r_[angles, angles[0]]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    for color, group in zip(COLORS, groups):
        part = df[df["group_label"] == group].set_index("amino_acid").reindex(aa_order)
        values = part["pooled_percentage"].to_numpy(float); values = np.r_[values, values[0]]
        ax.plot(closed, values, lw=1.8, color=color, label=group); ax.fill(closed, values, color=color, alpha=0.08)
    ax.set_xticks(angles, aa_order); ax.set_title("Amino-acid composition", fontweight="bold", pad=24)
    ax.legend(frameon=False, bbox_to_anchor=(1.25, 1.05))
    return fig


PLOTS: dict[str, Callable[[Path], plt.Figure]] = {
    "Fig1b": fig1b,
    "Fig2b": fig2b,
    "Fig3a": lambda d: plot_roc_panel(d, "Figure3a", "Effect of benchmark filtering"),
    "Fig3b": lambda d: plot_roc_panel(d, "Figure3b", "RLBPPred versus handcrafted-feature baseline"),
    "Fig3c": lambda d: probability_bar(d, "Figure3c", "mean_probability", "Mean predicted probability", "Mouse zero-shot prioritization"),
    "Fig3d": lambda d: probability_bar(d, "Figure3d", "mean_probability", "Mean predicted probability", "Human evidence gradient"),
    "Fig3e": fig3e,
    "Fig5a": lambda d: grouped_bar(load(d, "Figure5a"), "Category", "Group", "Percentage (%)", "Chromatin-regulator annotations", "Proteins (%)"),
    "Fig5b": lambda d: grouped_bar(load(d, "Figure5b"), "Database", "Group", "Overlap (%)", "LLPS database overlap", "Overlap (%)"),
    "Fig5c": fig5c,
    "Fig5d": fig5d,
    "SupFig1a": supfig1a,
    "SupFig1b": supfig1b,
    "SupFig1c": supfig1c,
    "SupFig2a": supfig2a,
    "SupFig2b": supfig2b,
    "SupFig2c": supfig2c,
    "SupFig2d": supfig2d,
    "SupFig3a": supfig3a,
    "SupFig3b": supfig3b,
    "SupFig3c": lambda d: probability_bar(d, "SuppFigure3c", "percentage_positive", "Called as RLBP (%)", "Human group-level predictions"),
    "SupFig3d": lambda d: probability_bar(d, "SuppFigure3d", "percentage_positive", "Called as RLBP (%)", "Mouse group-level predictions"),
    "SupFig5a": lambda d: grouped_bar(load(d, "SuppFigure5a"), "Group", "Group", "Percentage (%)", "Overall EpiFactors enrichment", "Annotated proteins (%)"),
    "SupFig5b": supfig5b,
    "SupFig5c": supfig5c,
    "SupFig5d": supfig5d,
}


MANUAL_PANELS = {
    "Fig1a": "workflow schematic assembled in Illustrator",
    "Fig2a": "model-workflow schematic assembled in Illustrator",
    "Fig4a": "wet-lab image", "Fig4b": "APBS surface render",
    "Fig4c": "wet-lab image", "Fig4d": "APBS surface render",
    "Fig6a": "browser screenshot", "Fig6b": "browser screenshot",
}


def main() -> None:
    args = parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": 0.8})
    panels = list(PLOTS) if args.all else [args.figure]
    for panel in panels:
        if panel in MANUAL_PANELS:
            print(f"{panel}: not generated ({MANUAL_PANELS[panel]}). See docs/reproduce_figures.md.")
            continue
        if panel not in PLOTS:
            available = ", ".join(list(PLOTS) + list(MANUAL_PANELS))
            raise SystemExit(f"Unknown panel {panel!r}. Available panels: {available}")
        fig = PLOTS[panel](args.data_dir)
        finish(fig, args.output_dir, panel, args.dpi)
        print(f"Generated {panel}")


if __name__ == "__main__":
    main()

