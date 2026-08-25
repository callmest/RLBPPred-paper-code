#!/usr/bin/env python3
"""
Experiment 2.1: Chromatin/epigenetic regulator classification.

Maps 342 novel predicted RLBPs, 376 known RLBPs, and 9,109 non-RLBPs
to EpiFactors v2.1 database, classifies as writer/reader/eraser/remodeler,
and tests for enrichment.
"""

import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

BASE = Path(os.environ.get("RLBPPRED_ROOT", Path(__file__).resolve().parents[2]))
EPIFACTORS = Path(
    os.environ.get(
        "EPIFACTORS_CSV",
        BASE / "data" / "external" / "epifactors" / "v2.1" / "EpiGenes_main.csv",
    )
)
IDR_DIR = BASE / "data" / "idr_analysis"
OUT_DIR = BASE / "results" / "idr_analysis"

FONT_FAMILY = ["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"]
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": FONT_FAMILY,
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.linewidth": 0.8,
})

GROUPS = {
    "Novel predicted RLBPs": IDR_DIR / "candidates_342.fasta",
    "Known RLBPs":           IDR_DIR / "known_376.fasta",
    "Non-RLBPs":             IDR_DIR / "negative_9109.fasta",
}
GROUP_ORDER = ["Novel predicted RLBPs", "Known RLBPs", "Non-RLBPs"]

FUNCTION_CATEGORIES = {
    "Writer": ["Histone modification write", "DNA modification write",
               "RNA modification", "Histone modification write read"],
    "Reader": ["Histone modification read", "Histone modification write read",
               "DNA modification read"],
    "Eraser": ["Histone modification erase", "DNA modification erase"],
    "Remodeler": ["Chromatin remodeling"],
    "Other chromatin": [],
}

CATEGORY_COLORS = {
    "Writer": "#E69F00",
    "Reader": "#56B4E9",
    "Eraser": "#009E73",
    "Remodeler": "#D55E00",
    "Other chromatin": "#CC79A7",
    "Non-epigenetic": "#DDDDDD",
}


def load_epifactors():
    """Load EpiFactors database and classify by function."""
    epi_proteins = {}
    with open(EPIFACTORS, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uniprot = row["UniProt_AC"].strip()
            gene = row["HGNC_symbol"].strip()
            func = row["Function"].strip()
            modification = row["Modification"].strip()

            category = "Other chromatin"
            for cat, keywords in FUNCTION_CATEGORIES.items():
                if cat == "Other chromatin":
                    continue
                for kw in keywords:
                    if kw.lower() in func.lower():
                        category = cat
                        break
                if category != "Other chromatin":
                    break

            if "remodel" in func.lower():
                category = "Remodeler"

            epi_proteins[uniprot] = {
                "gene": gene,
                "function": func,
                "modification": modification,
                "category": category,
            }
    return epi_proteins


def load_group_ids():
    """Load UniProt IDs for each group."""
    from Bio import SeqIO
    group_ids = {}
    for group, fasta_path in GROUPS.items():
        ids = set()
        for rec in SeqIO.parse(fasta_path, "fasta"):
            ids.add(rec.id)
        group_ids[group] = ids
    return group_ids


def classify_groups(group_ids, epi_proteins):
    """Classify each group's proteins by epigenetic category."""
    results = {}
    for group in GROUP_ORDER:
        ids = group_ids[group]
        categories = Counter()
        epi_hits = {}
        for pid in ids:
            if pid in epi_proteins:
                cat = epi_proteins[pid]["category"]
                categories[cat] += 1
                epi_hits[pid] = epi_proteins[pid]
            else:
                categories["Non-epigenetic"] += 1
        results[group] = {
            "categories": dict(categories),
            "total": len(ids),
            "epi_count": sum(v for k, v in categories.items() if k != "Non-epigenetic"),
            "hits": epi_hits,
        }
    return results


def fisher_enrichment(results):
    """Fisher exact test for epigenetic factor enrichment."""
    tests = {}
    for group in ["Novel predicted RLBPs", "Known RLBPs"]:
        neg = results["Non-RLBPs"]
        grp = results[group]
        # 2x2: [epi_in_group, non_epi_in_group], [epi_in_neg, non_epi_in_neg]
        table = [
            [grp["epi_count"], grp["total"] - grp["epi_count"]],
            [neg["epi_count"], neg["total"] - neg["epi_count"]],
        ]
        odds, p = sp_stats.fisher_exact(table, alternative="greater")
        tests[group] = {"odds_ratio": odds, "p_value": p, "table": table}

        # Per-category enrichment
        cat_tests = {}
        for cat in ["Writer", "Reader", "Eraser", "Remodeler", "Other chromatin"]:
            g_cat = grp["categories"].get(cat, 0)
            g_rest = grp["total"] - g_cat
            n_cat = neg["categories"].get(cat, 0)
            n_rest = neg["total"] - n_cat
            if g_cat + n_cat == 0:
                continue
            odds_c, p_c = sp_stats.fisher_exact(
                [[g_cat, g_rest], [n_cat, n_rest]], alternative="greater"
            )
            cat_tests[cat] = {"count": g_cat, "odds_ratio": odds_c, "p_value": p_c}
        tests[group + "_categories"] = cat_tests
    return tests


def plot_stacked_bar(results, ax):
    """Stacked bar chart: epigenetic category proportions."""
    cats = ["Writer", "Reader", "Eraser", "Remodeler", "Other chromatin"]
    x = np.arange(len(GROUP_ORDER))
    width = 0.5
    bottom = np.zeros(len(GROUP_ORDER))

    for cat in cats:
        vals = []
        for g in GROUP_ORDER:
            count = results[g]["categories"].get(cat, 0)
            pct = 100.0 * count / results[g]["total"]
            vals.append(pct)
        ax.bar(x, vals, width, bottom=bottom, label=cat,
               color=CATEGORY_COLORS[cat], edgecolor="black", linewidth=0.3)
        for i, v in enumerate(vals):
            if v > 1.5:
                ax.text(x[i], bottom[i] + v / 2,
                        f"{results[GROUP_ORDER[i]]['categories'].get(cat, 0)}",
                        ha="center", va="center", fontsize=7, fontweight="bold")
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(["Novel predicted\nRLBPs (n=342)",
                         "Known RLBPs\n(n=376)",
                         "Non-RLBPs\n(n=9,109)"], fontsize=8)
    ax.set_ylabel("Percentage of proteins (%)")
    ax.set_title("Epigenetic factor composition", fontsize=10)
    ax.legend(frameon=False, fontsize=7, loc="upper right", ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_enrichment_dots(tests, ax):
    """Dot plot: per-category enrichment significance."""
    cats = ["Writer", "Reader", "Eraser", "Remodeler", "Other chromatin"]
    groups_to_plot = ["Novel predicted RLBPs", "Known RLBPs"]
    colors_g = {"Novel predicted RLBPs": "#E69F00", "Known RLBPs": "#56B4E9"}

    y_pos = 0
    y_labels = []
    y_ticks = []
    for cat in cats:
        for gi, g in enumerate(groups_to_plot):
            cat_tests = tests.get(g + "_categories", {})
            if cat not in cat_tests:
                continue
            ct = cat_tests[cat]
            neg_log_p = -np.log10(max(ct["p_value"], 1e-20))
            size = max(ct["count"] * 3, 10)
            ax.scatter(neg_log_p, y_pos, s=size, c=colors_g[g],
                       edgecolor="black", linewidth=0.4, alpha=0.85,
                       zorder=3)
            ax.text(neg_log_p + 0.3, y_pos,
                    f"n={ct['count']}, OR={ct['odds_ratio']:.1f}",
                    va="center", fontsize=6.5)
            y_labels.append(f"{cat} ({'Novel' if gi == 0 else 'Known'})")
            y_ticks.append(y_pos)
            y_pos += 1
        y_pos += 0.3

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=7.5)
    ax.set_xlabel("-log10(p-value)")
    ax.set_title("Enrichment vs Non-RLBPs\n(Fisher exact test)", fontsize=10)
    ax.axvline(-np.log10(0.05), color="red", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.text(-np.log10(0.05) + 0.1, y_pos - 1, "p=0.05", fontsize=6.5, color="red")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E69F00',
               markersize=8, label='Novel predicted'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#56B4E9',
               markersize=8, label='Known RLBPs'),
    ]
    ax.legend(handles=legend_elements, frameon=False, fontsize=7, loc="lower right")


def write_legend(results, tests, out_dir):
    md = """# Figure: Chromatin/Epigenetic Regulator Classification (Experiment 2.1)

## Method

Proteins were mapped to the EpiFactors v2.1 database (Medvedeva et al., NAR 2023;
DOI: 10.1093/nar/gkac989), which catalogs 801 human epigenetic factors classified
by function. Proteins were categorized as:
- **Writer**: Histone/DNA/RNA modification writers
- **Reader**: Histone/DNA modification readers
- **Eraser**: Histone/DNA modification erasers
- **Remodeler**: Chromatin remodeling factors
- **Other chromatin**: Other epigenetic factors not fitting above categories

Enrichment was assessed by Fisher exact test (one-sided, greater) comparing
each RLBP group against Non-RLBPs.

## Results

"""
    for g in GROUP_ORDER:
        r = results[g]
        md += f"**{g}** (n={r['total']}): {r['epi_count']} epigenetic factors "
        md += f"({100*r['epi_count']/r['total']:.1f}%)\n"
        for cat in ["Writer", "Reader", "Eraser", "Remodeler", "Other chromatin"]:
            count = r["categories"].get(cat, 0)
            if count > 0:
                md += f"  - {cat}: {count}\n"
        md += "\n"

    for g in ["Novel predicted RLBPs", "Known RLBPs"]:
        t = tests[g]
        md += f"**{g} vs Non-RLBPs**: OR={t['odds_ratio']:.2f}, p={t['p_value']:.2e}\n"
        cat_tests = tests.get(g + "_categories", {})
        for cat, ct in cat_tests.items():
            md += f"  - {cat}: n={ct['count']}, OR={ct['odds_ratio']:.1f}, p={ct['p_value']:.2e}\n"
        md += "\n"

    path = out_dir / "exp2_1_chromatin_classification_legend.md"
    with open(path, "w") as f:
        f.write(md)
    print(f"Saved legend: {path}")


def main():
    print("Loading EpiFactors v2.1...")
    epi_proteins = load_epifactors()
    print(f"  {len(epi_proteins)} epigenetic factors loaded")

    print("Loading protein groups...")
    group_ids = load_group_ids()

    print("Classifying...")
    results = classify_groups(group_ids, epi_proteins)
    for g in GROUP_ORDER:
        r = results[g]
        print(f"  {g}: {r['epi_count']}/{r['total']} epigenetic factors "
              f"({100*r['epi_count']/r['total']:.1f}%)")
        for cat, count in sorted(r["categories"].items()):
            if cat != "Non-epigenetic" and count > 0:
                print(f"    {cat}: {count}")

    print("Fisher enrichment tests...")
    tests = fisher_enrichment(results)
    for g in ["Novel predicted RLBPs", "Known RLBPs"]:
        t = tests[g]
        print(f"  {g} vs Non-RLBPs: OR={t['odds_ratio']:.2f}, p={t['p_value']:.2e}")

    # Save detailed results
    detail = {}
    for g in GROUP_ORDER:
        detail[g] = {
            "categories": results[g]["categories"],
            "total": results[g]["total"],
            "epi_count": results[g]["epi_count"],
            "hits": {pid: info for pid, info in results[g]["hits"].items()},
        }
    with open(OUT_DIR / "exp2_1_chromatin_classification.json", "w") as f:
        json.dump(detail, f, indent=2)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                    gridspec_kw={"width_ratios": [1, 1.3]})
    fig.subplots_adjust(wspace=0.45, left=0.08, right=0.95, top=0.90, bottom=0.12)

    plot_stacked_bar(results, ax1)
    plot_enrichment_dots(tests, ax2)

    for ext in ["svg", "pdf", "png"]:
        out = str(OUT_DIR / "exp2_1_chromatin_classification") + f".{ext}"
        fig.savefig(out, dpi=300 if ext == "png" else None, bbox_inches="tight")
    print(f"Saved figure: exp2_1_chromatin_classification.[svg|pdf|png]")
    plt.close(fig)

    write_legend(results, tests, OUT_DIR)

    # List notable Novel epigenetic factors
    print("\nNotable Novel predicted RLBP epigenetic factors:")
    for pid, info in sorted(results["Novel predicted RLBPs"]["hits"].items(),
                            key=lambda x: x[1]["category"]):
        print(f"  {info['gene']} ({pid}): {info['category']} - {info['function']}")


if __name__ == "__main__":
    main()
