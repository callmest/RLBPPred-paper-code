#!/usr/bin/env python3
"""Plot Experiment 1.3b (publication version): LLPS database overlap."""

import json
from pathlib import Path

import matplotlib
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Publication style
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.2,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "lines.linewidth": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

BASE = Path(__file__).resolve().parents[2]
RESULTS = BASE / "results" / "idr_analysis"
DATA_JSON = RESULTS / "exp1_3b_llps_overlap.json"

DATABASES = ["PhaSepDB", "DrLLPS", "LLPSDB", "PhaSePro", "Union"]
GROUPS = ["Novel_RLBP", "Known_RLBP", "Non_RLBP"]
GROUP_LABELS = {
    "Novel_RLBP": "Novel predicted RLBPs (n=342)",
    "Known_RLBP": "Known RLBPs (n=376)",
    "Non_RLBP": "Non-RLBPs (n=9,109)",
}
GROUP_COLORS = {
    "Novel_RLBP": "#B84A3A",
    "Known_RLBP": "#4F6F9F",
    "Non_RLBP": "#9AA5B1",
}


def p_to_stars(p_value):
    if p_value < 1e-10:
        return "***"
    if p_value < 1e-5:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def load_data():
    with open(DATA_JSON) as f:
        return json.load(f)


def format_db_labels(db_sizes):
    return [
        f"PhaSepDB\n({db_sizes['PhaSepDB_human']})",
        f"DrLLPS\n({db_sizes['DrLLPS_human']})",
        f"LLPSDB\n({db_sizes['LLPSDB_human']})",
        f"PhaSePro\n({db_sizes['PhaSePro_human']})",
        f"Union\n({db_sizes['Union_human']})",
    ]


def draw_overlap_plot(data):
    enrichment = data["enrichment_results"]
    db_labels = format_db_labels(data["database_sizes"])

    fig, ax = plt.subplots(figsize=(6.2, 3.85))
    fig.subplots_adjust(left=0.11, right=0.76, top=0.96, bottom=0.23)

    x = np.arange(len(DATABASES))
    width = 0.22
    offsets = [-width, 0, width]

    for group, offset in zip(GROUPS, offsets):
        pcts = [enrichment[db][group]["overlap_pct"] for db in DATABASES]
        bars = ax.bar(
            x + offset,
            pcts,
            width,
            label=GROUP_LABELS[group],
            color=GROUP_COLORS[group],
            edgecolor="#000000",
            linewidth=0.45,
            alpha=0.86,
            zorder=3,
        )

        for db, bar in zip(DATABASES, bars):
            result = enrichment[db][group]
            count = result["overlap_count"]
            pct = result["overlap_pct"]
            y = bar.get_height()

            if count <= 0:
                continue

            if pct >= 20:
                label_y = y - 2.1
                label_color = "#FFFFFF"
                va = "top"
            else:
                label_y = y + (1.0 if pct >= 1 else 0.75)
                label_color = "#222222"
                va = "bottom"

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                label_y,
                f"{count}",
                ha="center",
                va=va,
                fontsize=5.8,
                color=label_color,
                fontweight="bold" if pct >= 20 else "normal",
                zorder=4,
            )

            if group != "Non_RLBP":
                stars = p_to_stars(result["p_value"])
                if stars:
                    star_y = min(y + (2.0 if pct >= 20 else 3.2), 98)
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        star_y,
                        stars,
                        ha="center",
                        va="bottom",
                        fontsize=7.0,
                        fontweight="bold",
                        color="#8B1E24",
                    )

    ax.set_xticks(x)
    ax.set_xticklabels(db_labels, fontsize=8)
    ax.set_ylabel("Overlap with LLPS databases (%)")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(plt.MultipleLocator(20))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(10))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        borderaxespad=0.0,
    )
    for handle in legend.legend_handles:
        handle.set_alpha(0.86)

    ax.text(
        1.02,
        0.54,
        "* p<0.05\n** p<1e-5\n*** p<1e-10\nFisher exact test\nvs Non-RLBPs",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.4,
        color="#333333",
        linespacing=1.35,
    )

    return fig


def write_legend(data):
    enrichment = data["enrichment_results"]
    group_sizes = data["group_sizes"]

    md = f"""# Figure: LLPS Database Overlap Analysis (Experiment 1.3b)

## Method

Novel predicted RLBPs, known RLBPs/significant positive samples and Non-RLBPs were
intersected with four curated human LLPS-related protein resources: PhaSepDB, DrLLPS,
PhaSePro and LLPSDB. The Union category represents proteins present in at least one
of these LLPS resources. Enrichment was assessed by one-sided Fisher exact test using
Non-RLBPs as the background/reference group.

## Group sizes

- Novel predicted RLBPs: n={group_sizes['Novel_RLBP']}
- Known RLBPs/significant positive samples: n={group_sizes['Known_RLBP']}
- Non-RLBPs: n={group_sizes['Non_RLBP']}

## Results Summary

"""
    for db in DATABASES:
        md += f"**{db}**\n"
        for group in GROUPS:
            result = enrichment[db][group]
            md += (
                f"- {GROUP_LABELS[group]}: {result['overlap_count']}/{result['group_total']} "
                f"({result['overlap_pct']:.2f}%), OR={result['odds_ratio']:.2f}, "
                f"p={result['p_value']:.2e}\n"
            )
        md += "\n"

    union_novel = enrichment["Union"]["Novel_RLBP"]
    union_known = enrichment["Union"]["Known_RLBP"]
    union_neg = enrichment["Union"]["Non_RLBP"]
    md += f"""## Key Findings

Novel predicted RLBPs show substantial overlap with curated LLPS-associated proteins:
{union_novel['overlap_count']}/{union_novel['group_total']} ({union_novel['overlap_pct']:.1f}%)
overlap with the union LLPS database set, compared with {union_neg['overlap_count']}/{union_neg['group_total']}
({union_neg['overlap_pct']:.1f}%) in Non-RLBPs (OR={union_novel['odds_ratio']:.2f},
p={union_novel['p_value']:.1e}). Known RLBPs/significant positive samples provide a positive
reference, showing {union_known['overlap_count']}/{union_known['group_total']}
({union_known['overlap_pct']:.1f}%) overlap with the union LLPS set (OR={union_known['odds_ratio']:.2f},
p={union_known['p_value']:.1e} vs Non-RLBPs). These results support that the newly
prioritized candidates share LLPS-associated annotations with experimentally supported RLBPs.
"""

    path = RESULTS / "exp1_3b_llps_overlap_legend.md"
    with open(path, "w") as f:
        f.write(md)
    print(f"Saved: {path}")


def main():
    data = load_data()
    fig = draw_overlap_plot(data)
    for ext in ["svg", "pdf", "png"]:
        out = RESULTS / f"exp1_3b_llps_overlap_pub.{ext}"
        fig.savefig(out, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)
    write_legend(data)


if __name__ == "__main__":
    main()
