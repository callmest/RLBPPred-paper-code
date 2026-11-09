#!/usr/bin/env python3
"""
Experiment 2.1 (publication version): Chromatin/epigenetic regulator classification.

Compares Novel Predicted RLBPs (n=342), Known RLBPs/significant positives
(n=376), and Non-RLBPs (n=9,109).
Outputs three separate figures:
  1. Overall EpiFactors enrichment
  2. Functional category composition
  3. Category-specific enrichment dot plot
"""


# Historical 342-candidate analysis, retained for provenance only.
# Current frozen publication replay: python scripts/reproduce_figures.py --all
import sys as _historical_sys
if __name__ == "__main__":
    if "--historical-342" not in _historical_sys.argv:
        raise SystemExit("Historical 342-candidate workflow. Use the current frozen replay, or explicitly add --historical-342.")
    _historical_sys.argv.remove("--historical-342")
import json
from pathlib import Path

import matplotlib
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

BASE = Path(__file__).resolve().parents[2]
OUT_DIR = BASE / "results" / "idr_analysis"
DATA_JSON = OUT_DIR / "exp2_1_chromatin_classification.json"

# Publication style
FONT_FAMILY = ["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"]
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": FONT_FAMILY,
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

CATEGORIES = ["Writer", "Reader", "Eraser", "Remodeler", "Other chromatin"]
GROUP_ORDER = ["Novel predicted RLBPs", "Known RLBPs", "Non-RLBPs"]
GROUP_LABELS = {
    "Novel predicted RLBPs": "Novel predicted\nRLBPs (n=342)",
    "Known RLBPs": "Known RLBPs\n(n=376)",
    "Non-RLBPs": "Non-RLBPs\n(n=9,109)",
}

CATEGORY_COLORS = {
    "Writer": "#B84A3A",
    "Reader": "#4F6F9F",
    "Eraser": "#5B8C85",
    "Remodeler": "#9E7A3A",
    "Other chromatin": "#8F6D99",
}
GROUP_COLORS = {
    "Novel predicted RLBPs": "#B84A3A",
    "Known RLBPs": "#4F6F9F",
    "Non-RLBPs": "#9AA5B1",
}
SIG_COLOR = "#8B1E24"


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig, prefix):
    for ext in ["svg", "pdf", "png"]:
        out = OUT_DIR / f"{prefix}.{ext}"
        fig.savefig(str(out), dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


def load_data():
    with open(DATA_JSON) as f:
        return json.load(f)


def fisher_vs_negative(data, group, category=None):
    neg = data["Non-RLBPs"]
    grp = data[group]
    if category is None:
        g_hit = grp["epi_count"]
        n_hit = neg["epi_count"]
    else:
        g_hit = grp["categories"].get(category, 0)
        n_hit = neg["categories"].get(category, 0)

    table = [
        [g_hit, grp["total"] - g_hit],
        [n_hit, neg["total"] - n_hit],
    ]
    odds, p_value = sp_stats.fisher_exact(table, alternative="greater")
    return {
        "count": g_hit,
        "pct": 100.0 * g_hit / grp["total"],
        "odds_ratio": odds,
        "p_value": p_value,
    }


def compute_enrichment(data):
    results = {}
    for group in ["Novel predicted RLBPs", "Known RLBPs"]:
        results[group] = {
            "overall": fisher_vs_negative(data, group),
            "categories": {
                cat: fisher_vs_negative(data, group, cat) for cat in CATEGORIES
            },
        }
    return results


def plot_overall_bar(data, enrichment, ax):
    pcts = [
        100.0 * data[group]["epi_count"] / data[group]["total"]
        for group in GROUP_ORDER
    ]
    colors = [GROUP_COLORS[group] for group in GROUP_ORDER]

    bars = ax.bar(
        range(len(GROUP_ORDER)),
        pcts,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        width=0.58,
        alpha=0.86,
    )

    for bar, pct, group in zip(bars, pcts, GROUP_ORDER):
        grp = data[group]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.35,
            f"{grp['epi_count']}/{grp['total']}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )

    y_max = max(pcts) * 1.55
    ax.set_ylim(0, y_max)
    ax.set_xticks(range(len(GROUP_ORDER)))
    ax.set_xticklabels([GROUP_LABELS[group] for group in GROUP_ORDER], fontsize=7.2)
    ax.set_ylabel("EpiFactors-annotated proteins (%)", fontsize=8)
    style_axis(ax)

    for x_pos, group in [(0, "Novel predicted RLBPs"), (1, "Known RLBPs")]:
        stat = enrichment[group]["overall"]
        or_text = f"OR={stat['odds_ratio']:.1f}" if stat["odds_ratio"] < 1000 else "OR=inf"
        ax.text(
            x_pos,
            y_max * (0.86 if group == "Known RLBPs" else 0.76),
            f"vs Non-RLBPs\np={stat['p_value']:.1e}\n{or_text}",
            ha="center",
            va="center",
            fontsize=6.2,
            color=SIG_COLOR,
        )


def plot_grouped_bars(data, ax):
    x = np.arange(len(CATEGORIES))
    width = 0.23

    for gi, group in enumerate(GROUP_ORDER):
        grp = data[group]
        total = grp["total"]
        pcts = [100.0 * grp["categories"].get(cat, 0) / total for cat in CATEGORIES]
        counts = [grp["categories"].get(cat, 0) for cat in CATEGORIES]

        offset = (gi - 1) * width
        bars = ax.bar(
            x + offset,
            pcts,
            width,
            color=GROUP_COLORS[group],
            edgecolor="black",
            linewidth=0.35,
            alpha=0.86,
            label=GROUP_LABELS[group].replace("\n", " "),
        )

        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.10,
                    str(count),
                    ha="center",
                    va="bottom",
                    fontsize=5.7,
                    rotation=90 if count >= 10 else 0,
                )

    ax.set_ylim(0, max(
        100.0 * data[group]["categories"].get(cat, 0) / data[group]["total"]
        for group in GROUP_ORDER for cat in CATEGORIES
    ) * 1.18)
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIES, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("Percentage of proteins (%)")
    ax.legend(frameon=False, fontsize=6.8, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, ncol=1)
    style_axis(ax)


def plot_enrichment_bubble(enrichment, ax, ax_text):
    group_offsets = {
        "Novel predicted RLBPs": -0.16,
        "Known RLBPs": 0.16,
    }
    for y, cat in enumerate(CATEGORIES):
        for group in ["Novel predicted RLBPs", "Known RLBPs"]:
            ct = enrichment[group]["categories"][cat]
            neg_log_p = -np.log10(max(ct["p_value"], 1e-300))
            size = max(ct["count"] * 8.5, 26)
            ax.scatter(
                neg_log_p,
                y + group_offsets[group],
                s=size,
                c=GROUP_COLORS[group],
                marker="o",
                edgecolor="black",
                linewidth=0.45,
                alpha=0.86,
                zorder=3,
            )

    ax.set_xlim(0, 31.5)
    ax.axvline(-np.log10(0.05), color=SIG_COLOR, linestyle="--", linewidth=0.6, alpha=0.55)
    ax.axvline(-np.log10(0.001), color=SIG_COLOR, linestyle=":", linewidth=0.55, alpha=0.45)
    ax.text(-np.log10(0.05) + 0.3, len(CATEGORIES) - 0.2, "p=0.05", fontsize=6.2, color=SIG_COLOR)

    ax.set_yticks(range(len(CATEGORIES)))
    ax.set_yticklabels(CATEGORIES, fontsize=7.5)
    ax.set_xlabel("-log$_{10}$(p-value)", fontsize=9)
    ax.invert_yaxis()
    style_axis(ax)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GROUP_COLORS["Novel predicted RLBPs"],
               markeredgecolor="black", markersize=6, label="Novel predicted RLBPs"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GROUP_COLORS["Known RLBPs"],
               markeredgecolor="black", markersize=6, label="Known RLBPs"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=6.8, loc="lower right")

    ax_text.set_xlim(0, 1)
    ax_text.set_ylim(ax.get_ylim())
    ax_text.axis("off")
    ax_text.text(0.02, -0.65, "Count, OR", ha="left", va="center",
                 fontsize=6.8, fontweight="bold")
    for y, cat in enumerate(CATEGORIES):
        lines = []
        for group, short in [("Novel predicted RLBPs", "Novel"), ("Known RLBPs", "Known")]:
            ct = enrichment[group]["categories"][cat]
            or_text = f"{ct['odds_ratio']:.1f}" if ct["odds_ratio"] < 1000 else "inf"
            lines.append(f"{short}: n={ct['count']}, OR={or_text}")
        ax_text.text(0.02, y, "\n".join(lines), ha="left", va="center",
                     fontsize=6.3, linespacing=1.2)


def write_legend(data, enrichment, out_dir):
    novel = data["Novel predicted RLBPs"]
    known = data["Known RLBPs"]
    neg = data["Non-RLBPs"]

    novel_overall = enrichment["Novel predicted RLBPs"]["overall"]
    known_overall = enrichment["Known RLBPs"]["overall"]
    neg_pct = 100.0 * neg["epi_count"] / neg["total"]

    md = f"""# Figure: Chromatin/Epigenetic Regulator Classification (Experiment 2.1)

## Method

Proteins were mapped to the EpiFactors v2.1 database (Medvedeva et al., NAR 2023;
DOI: 10.1093/nar/gkac989), containing 795 human proteins annotated as epigenetic
and chromatin-associated regulatory factors. Proteins were classified as Writer
(histone/DNA/RNA modification writers), Reader (modification readers), Eraser
(modification erasers), Remodeler (chromatin remodeling factors), or Other
chromatin factors. Enrichment was assessed by one-sided Fisher exact test comparing
Novel Predicted RLBPs and Known RLBPs/significant positive samples against Non-RLBPs.

## Panel descriptions

Overall enrichment figure: Overall percentage of proteins mapped to EpiFactors
in each group. Novel Predicted RLBPs show {100.0 * novel['epi_count'] / novel['total']:.1f}%
EpiFactors-annotated proteins vs {neg_pct:.1f}% in Non-RLBPs
(OR={novel_overall['odds_ratio']:.1f}, p={novel_overall['p_value']:.1e}).
Known RLBPs/significant positive samples show {100.0 * known['epi_count'] / known['total']:.1f}%
EpiFactors-annotated proteins (OR={known_overall['odds_ratio']:.1f},
p={known_overall['p_value']:.1e} vs Non-RLBPs).

Category composition figure: Grouped bar chart showing the percentage of proteins
in each EpiFactors category across Novel Predicted RLBPs, Known RLBPs and Non-RLBPs.
Numbers above bars indicate protein counts.

Category enrichment figure: Dot plot showing enrichment significance (-log10 p-value)
for each category. Dot size is proportional to protein count. Dashed red line = p=0.05,
dotted red line = p=0.001. The right-side text column lists count and odds ratio
for Novel Predicted RLBPs and Known RLBPs relative to Non-RLBPs.

## Results Summary

"""
    for group in ["Novel predicted RLBPs", "Known RLBPs", "Non-RLBPs"]:
        grp = data[group]
        md += f"**{group}** (n={grp['total']}): {grp['epi_count']} EpiFactors-annotated proteins "
        md += f"({100.0 * grp['epi_count'] / grp['total']:.1f}%).\n"
        for cat in CATEGORIES:
            md += f"- {cat}: {grp['categories'].get(cat, 0)}\n"
        md += "\n"

    for group in ["Novel predicted RLBPs", "Known RLBPs"]:
        overall = enrichment[group]["overall"]
        md += f"**{group} vs Non-RLBPs overall**: OR={overall['odds_ratio']:.2f}, "
        md += f"p={overall['p_value']:.2e}.\n"
        for cat in CATEGORIES:
            ct = enrichment[group]["categories"][cat]
            or_text = f"{ct['odds_ratio']:.1f}" if ct["odds_ratio"] < 1000 else "inf"
            md += f"- {cat}: n={ct['count']}, OR={or_text}, p={ct['p_value']:.2e}\n"
        md += "\n"

    md += """## Key Findings

Novel Predicted RLBPs and Known RLBPs/significant positive samples are both strongly
enriched for EpiFactors-annotated proteins compared with Non-RLBPs. The matched
enrichment pattern supports the interpretation that the newly prioritized proteins
share chromatin- and epigenetic-regulatory properties with experimentally supported
R-loop binding proteins, while the negative set is strongly depleted for such factors.
"""
    path = out_dir / "exp2_1_chromatin_classification_pub_legend.md"
    with open(path, "w") as f:
        f.write(md)
    print(f"Saved legend: {path}")


def main():
    print("Loading data...")
    data = load_data()
    enrichment = compute_enrichment(data)

    fig1, ax1 = plt.subplots(figsize=(3.5, 3.25))
    fig1.subplots_adjust(left=0.22, right=0.97, top=0.96, bottom=0.22)
    plot_overall_bar(data, enrichment, ax1)
    save_figure(fig1, "exp2_1_chromatin_overall_enrichment_pub")

    fig2, ax2 = plt.subplots(figsize=(6.1, 3.4))
    fig2.subplots_adjust(left=0.12, right=0.73, top=0.96, bottom=0.28)
    plot_grouped_bars(data, ax2)
    save_figure(fig2, "exp2_1_chromatin_category_composition_pub")

    fig3 = plt.figure(figsize=(5.9, 3.45))
    gs = fig3.add_gridspec(1, 2, width_ratios=[3.6, 1.65], wspace=0.08)
    ax3 = fig3.add_subplot(gs[0, 0])
    ax3_text = fig3.add_subplot(gs[0, 1], sharey=ax3)
    fig3.subplots_adjust(left=0.18, right=0.98, top=0.92, bottom=0.17)
    plot_enrichment_bubble(enrichment, ax3, ax3_text)
    save_figure(fig3, "exp2_1_chromatin_category_enrichment_pub")

    write_legend(data, enrichment, OUT_DIR)


if __name__ == "__main__":
    main()
